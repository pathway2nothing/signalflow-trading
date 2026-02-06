# Quick Start Guide

Build your first algorithmic trading strategy with SignalFlow.

---

## Prerequisites

- Python 3.12 or higher
- Basic understanding of Polars DataFrames
- Familiarity with trading concepts (OHLCV data, signals)

---

## Installation

```bash
pip install signalflow-trading
```

For technical analysis indicators:
```bash
pip install signalflow-ta
```

---

## Your First Strategy: SMA Crossover

We'll build a classic Simple Moving Average crossover strategy step by step,
using synthetic data so everything works offline.

### Step 1: Generate Market Data

Use `VirtualDataProvider` for self-contained development - no API keys needed:

```python
from datetime import datetime
from pathlib import Path

from signalflow.data.source import VirtualDataProvider
from signalflow.data.raw_store import DuckDbSpotStore

# Create store and virtual data provider
store = DuckDbSpotStore(db_path=Path("tutorial.duckdb"))
provider = VirtualDataProvider(
    store=store,
    base_prices={"BTCUSDT": 40000.0, "ETHUSDT": 2200.0},
    volatility=0.02,
    seed=42,
)

# Generate 10,000 bars of 1-minute data per pair
provider.download(pairs=["BTCUSDT", "ETHUSDT"], n_bars=10_000)
```

??? tip "Production: Binance Data"
    For real market data, swap in `BinanceSpotLoader`:
    ```python
    import asyncio
    from signalflow.data.source import BinanceSpotLoader

    loader = BinanceSpotLoader(store=store)
    asyncio.run(loader.download(pairs=["BTCUSDT", "ETHUSDT"], days=30))
    ```

### Step 2: Load Data into Framework

Convert stored data to SignalFlow's `RawData` container:

```python
from signalflow.data import RawDataFactory

raw_data = RawDataFactory.from_duckdb_spot_store(
    spot_store_path=Path("tutorial.duckdb"),
    pairs=["BTCUSDT", "ETHUSDT"],
    start=datetime(2020, 1, 1),
    end=datetime(2030, 1, 1),
)

raw_data_view = raw_data.view()
df = raw_data_view.to_polars("spot")
print(f"Loaded {df.height} rows, pairs: {df['pair'].unique().to_list()}")
```

### Step 3: Extract Features

Create features for signal detection and ML validation:

```python
from signalflow.feature import FeaturePipeline, ExampleRsiFeature, ExampleSmaFeature

pipeline = FeaturePipeline(features=[
    ExampleRsiFeature(period=14),
    ExampleSmaFeature(period=20),
    ExampleSmaFeature(period=50),
])

features_df = pipeline.compute(df)
print(f"Feature columns: {pipeline.output_cols()}")
```

### Step 4: Detect Signals

Use the built-in SMA crossover detector:

```python
from signalflow.detector import ExampleSmaCrossDetector

detector = ExampleSmaCrossDetector(fast_period=20, slow_period=50)
signals = detector.run(raw_data_view)

signals_df = signals.value
active = signals_df.filter(signals_df["signal_type"] != "none")
print(f"Detected {active.height} active signals out of {signals_df.height} rows")
```

**Signal Types:**

- `rise` - Fast SMA crosses above slow SMA (bullish)
- `fall` - Fast SMA crosses below slow SMA (bearish)
- `none` - No crossover

### Step 5: Label Signals for ML

Add forward-looking labels to evaluate signal quality:

```python
from signalflow.target import FixedHorizonLabeler

labeler = FixedHorizonLabeler(
    price_col="close",
    horizon=60,        # look 60 bars ahead
    include_meta=True, # include t1 and log-return
)

# Extract signal timestamps for masking
signal_keys = signals.value.select(["pair", "timestamp"])

labeled_df = labeler.compute(
    df=raw_data_view.to_polars("spot"),
    signals=signals,
    data_context={"signal_keys": signal_keys},
)

# Only signal rows get meaningful labels
labeled_signals = labeled_df.filter(labeled_df["label"] != "none")
print(f"Labeled signals: {labeled_signals.height}")
print(labeled_signals.group_by("label").len().sort("label"))
```

!!! note "Signal Masking"
    Labels are computed on the full price series but **masked** to signal timestamps.
    Pass `data_context={"signal_keys": ...}` to enable masking. Non-signal rows
    receive `label="none"`.

### Step 6: Train Signal Validator (Meta-Labeling)

Filter signals using ML to predict success probability:

```python
import polars as pl
from signalflow.validator import SklearnSignalValidator
from signalflow.core import Signals

# Prepare training data - join features with labeled signals
feature_cols = pipeline.output_cols()
train_data = labeled_signals.join(
    features_df.select(["pair", "timestamp"] + feature_cols),
    on=["pair", "timestamp"],
    how="inner",
).drop_nulls(subset=feature_cols)

# Time-based train/test split (80/20)
split_idx = int(len(train_data) * 0.8)
train_df = train_data[:split_idx]
test_df = train_data[split_idx:]

# Create and train validator
validator = SklearnSignalValidator(model_type="random_forest")
validator.fit(
    X_train=train_df.select(["pair", "timestamp"] + feature_cols),
    y_train=train_df.select("label"),
)

# Predict on test set
test_signals = Signals(test_df.select(["pair", "timestamp", "signal_type", "signal"]))
validated = validator.validate_signals(
    test_signals,
    test_df.select(["pair", "timestamp"] + feature_cols),
)

print(validated.value.select(
    ["pair", "signal_type", "probability_rise", "probability_fall"]
).head())
```

!!! tip "Model Selection"
    `SklearnSignalValidator` supports:

    - `lightgbm` - fast gradient boosting (requires `pip install lightgbm`)
    - `xgboost` - gradient boosting (requires `pip install xgboost`)
    - `random_forest` - sklearn Random Forest
    - `logistic_regression` - linear baseline
    - `auto` - automatic selection via cross-validation

### Step 7: Backtest Strategy

Simulate trading with entry/exit rules and a broker:

```python
from signalflow.strategy.runner import BacktestRunner
from signalflow.strategy.component.entry.signal import SignalEntryRule
from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit
from signalflow.strategy.broker import BacktestBroker
from signalflow.strategy.broker.executor import VirtualSpotExecutor

# Configure components
entry_rule = SignalEntryRule(
    base_position_size=100.0,
    use_probability_sizing=False,
    max_positions_per_pair=1,
    max_total_positions=10,
)

exit_rule = TakeProfitStopLossExit(
    take_profit_pct=0.02,
    stop_loss_pct=0.01,
)

broker = BacktestBroker(executor=VirtualSpotExecutor(fee_rate=0.001))

# Create and run backtest
runner = BacktestRunner(
    strategy_id="sma_cross_quickstart",
    broker=broker,
    entry_rules=[entry_rule],
    exit_rules=[exit_rule],
    initial_capital=10_000.0,
)

state = runner.run(raw_data=raw_data, signals=signals)

# Results
print(f"Total trades: {len(runner.trades)}")
print(f"Final capital: ${state.capital:.2f}")

trades_df = runner.trades_df
if trades_df.height > 0:
    wins = trades_df.filter(pl.col("pnl") > 0).height
    print(f"Win rate: {wins / trades_df.height:.1%}")
```

---

## Complete Example

Here's the full workflow in one script:

```python
from datetime import datetime
from pathlib import Path
import polars as pl

from signalflow.data.source import VirtualDataProvider
from signalflow.data.raw_store import DuckDbSpotStore
from signalflow.data import RawDataFactory
from signalflow.feature import FeaturePipeline, ExampleRsiFeature, ExampleSmaFeature
from signalflow.detector import ExampleSmaCrossDetector
from signalflow.target import FixedHorizonLabeler
from signalflow.validator import SklearnSignalValidator
from signalflow.core import Signals
from signalflow.strategy.runner import BacktestRunner
from signalflow.strategy.component.entry.signal import SignalEntryRule
from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit
from signalflow.strategy.broker import BacktestBroker
from signalflow.strategy.broker.executor import VirtualSpotExecutor

# 1. Generate data
store = DuckDbSpotStore(db_path=Path("quickstart.duckdb"))
provider = VirtualDataProvider(store=store, seed=42)
provider.download(pairs=["BTCUSDT"], n_bars=10_000)

# 2. Load data
raw_data = RawDataFactory.from_duckdb_spot_store(
    spot_store_path=Path("quickstart.duckdb"),
    pairs=["BTCUSDT"],
    start=datetime(2020, 1, 1),
    end=datetime(2030, 1, 1),
)
view = raw_data.view()
df = view.to_polars("spot")

# 3. Features
pipeline = FeaturePipeline(features=[
    ExampleRsiFeature(period=14),
    ExampleSmaFeature(period=20),
    ExampleSmaFeature(period=50),
])
features_df = pipeline.compute(df)

# 4. Detect signals
detector = ExampleSmaCrossDetector(fast_period=20, slow_period=50)
signals = detector.run(view)

# 5. Backtest
runner = BacktestRunner(
    strategy_id="quickstart",
    broker=BacktestBroker(executor=VirtualSpotExecutor(fee_rate=0.001)),
    entry_rules=[SignalEntryRule(base_position_size=100.0, use_probability_sizing=False)],
    exit_rules=[TakeProfitStopLossExit(take_profit_pct=0.02, stop_loss_pct=0.01)],
    initial_capital=10_000.0,
)
state = runner.run(raw_data=raw_data, signals=signals)
print(f"Trades: {len(runner.trades)}, Final capital: ${state.capital:.2f}")
```

---

## Next Steps

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **[User Guide](guide/custom-data-types.md)**

    ---

    Custom data types, advanced features, and extension points

-   :material-code-braces:{ .lg .middle } **[API Reference](api/index.md)**

    ---

    Complete documentation for all classes and methods

-   :material-puzzle:{ .lg .middle } **[Ecosystem](ecosystem/index.md)**

    ---

    signalflow-ta (199 indicators) and signalflow-nn (deep learning validators)

-   :material-github:{ .lg .middle } **[GitHub](https://github.com/pathway2nothing/signalflow-trading)**

    ---

    Source code, issues, and contributions

</div>

---

## Common Patterns

### Using the Component Registry

Register and discover components dynamically:

```python
from signalflow.core import sf_component, default_registry, SfComponentType
from signalflow.detector import SignalDetector

@sf_component(name="my_detector")
class MyCustomDetector(SignalDetector):
    def detect(self, features, context=None):
        # Your signal detection logic
        ...

# List available detectors
detectors = default_registry.list(SfComponentType.DETECTOR)
print(f"Available: {detectors}")
```

### Working with Polars DataFrames

SignalFlow is Polars-first for performance:

```python
import polars as pl

# Filter active signals
active_signals = signals.value.filter(pl.col("signal_type") != "none")

# Group by pair
stats = signals.value.group_by("pair").agg(
    pl.col("signal_type").filter(pl.col("signal_type") != "none").len().alias("signal_count")
)
```

---

## Troubleshooting

### No signals detected?

Check detector parameters and data quality:

```python
# Verify data
print(df.select("close").describe())

# Try shorter periods for more signals
detector = ExampleSmaCrossDetector(fast_period=10, slow_period=20)
```

### Import errors?

```bash
# Core installation
pip install signalflow-trading

# With technical analysis indicators
pip install signalflow-ta

# With neural network validators
pip install signalflow-nn
```

---

!!! success "You're Ready!"
    You now have a working algorithmic trading strategy.
    Experiment with different detectors, validators, and parameters to improve performance.
    **Backtest thoroughly before live trading!**
