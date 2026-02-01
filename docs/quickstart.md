# Quick Start Guide

This guide walks you through building your first algorithmic trading strategy with SignalFlow in under 10 minutes.

---

## Prerequisites

Before starting, ensure you have:

- Python 3.12 or higher
- Basic understanding of pandas/Polars DataFrames
- Familiarity with trading concepts (OHLCV data, signals)

---

## Installation

Install SignalFlow via pip:
```bash
pip install signalflow-trading
```

For development installation with all extras:
```bash
pip install signalflow-trading[nn]
```

This includes:

- `dev`: Development tools (pytest, black, mypy)
- `nn`: Neural network support (PyTorch, Lightning)

---

## Your First Strategy: SMA Crossover

We'll build a classic Simple Moving Average (SMA) crossover strategy step by step.

### Step 1: Load Market Data

First, download historical data from Binance:
```python
import asyncio
from pathlib import Path
from datetime import datetime
from signalflow.data import BinanceSpotLoader

# Initialize loader
loader = BinanceSpotLoader(
    store_path=Path("data/binance_spot.duckdb"),
    timeframe="1m"  # 1-minute candles
)

# Download 30 days of data for BTC and ETH
async def download_data():
    await loader.download(
        pairs=["BTCUSDT", "ETHUSDT"],
        days=30,
        fill_gaps=True
    )

# Run download
asyncio.run(download_data())
```

!!! tip "Data Source"
    SignalFlow currently supports Binance Spot data. Support for Binance Futures and other exchanges is planned for future releases.

### Step 2: Load Data into Framework

Convert stored data to SignalFlow's `RawData` format:
```python
from signalflow.data import RawDataFactory

# Load data from DuckDB
raw_data = RawDataFactory.from_duckdb_spot_store(
    spot_store_path=Path("data/binance_spot.duckdb"),
    pairs=["BTCUSDT", "ETHUSDT"],
    start=datetime(2024, 11, 1),
    end=datetime(2024, 12, 1),
    data_types=["spot"]
)

# Inspect loaded data
print(f"Loaded {len(raw_data['spot'])} candles")
print(f"Pairs: {raw_data.pairs}")
print(f"Date range: {raw_data.datetime_start} to {raw_data.datetime_end}")
```

??? info "RawData Container"
    `RawData` is an immutable container that stores market data with metadata. It provides unified access to different data types (spot, futures, LOB) and validates data integrity.

### Step 3: Detect Signals

Use the built-in SMA crossover detector:
```python
from signalflow.detector import SmaCrossSignalDetector

# Create detector (20/50 SMA crossover)
detector = SmaCrossSignalDetector(
    fast_period=20,
    slow_period=50,
    price_col="close"
)

# Detect signals
signals = detector.run(raw_data)

# Inspect signals
signals_df = signals.value
print(f"Detected {len(signals_df)} signals")
print(signals_df.filter(pl.col("signal_type") != "none").head())
```

**Signal Types:**

- `RISE`: Fast SMA crosses above slow SMA (bullish signal)
- `FALL`: Fast SMA crosses below slow SMA (bearish signal)
- `NONE`: No crossover detected

### Step 4: Label Signals for ML

Add labels to signals using the Triple Barrier Method:
```python
from signalflow.labeler import TripleBarrierLabeler

# Create labeler
labeler = TripleBarrierLabeler(
    take_profit=0.02,   # 2% profit target
    stop_loss=0.01,     # 1% stop loss
    max_holding=1440    # 24 hours (1440 minutes)
)

# Label signals
labeled_signals = labeler.label(signals, raw_data)

# Check label distribution
label_counts = (
    labeled_signals.value
    .group_by("label")
    .count()
)
print(label_counts)
```

??? note "Triple Barrier Method"
    This labeling technique from Lopez de Prado's "Advances in Financial Machine Learning" assigns labels based on which barrier is hit first:
    
    - **Take Profit** → Label: RISE (profitable trade)
    - **Stop Loss** → Label: FALL (losing trade)
    - **Time Limit** → Label: NONE (inconclusive)

### Step 5: Extract Features

Create features for machine learning validation:
```python
from signalflow.feature import (
    FeaturePipeline,
    RsiFeature,
    BollingerBandsFeature,
    AtrFeature
)

# Define feature pipeline
pipeline = FeaturePipeline(features=[
    RsiFeature(period=14),
    BollingerBandsFeature(period=20, num_std=2),
    AtrFeature(period=14)
])

# Extract features
features = pipeline.run(raw_data_view)

# Check extracted features
print(features.columns)
# Output: ['pair', 'timestamp', 'rsi_14', 'bb_upper', 'bb_middle', 'bb_lower', 'atr_14']
```

### Step 6: Train Signal Validator (Optional)

Filter and validate signals using machine learning:
```python
import polars as pl
from signalflow.validator import SklearnSignalValidator

# Prepare training data (filter to active signals only)
active_signals = labeled_signals.value.filter(
    pl.col("signal_type") != "none"
)

# Join features
train_data = active_signals.join(
    features,
    on=["pair", "timestamp"],
    how="inner"
)

# Split train/test (80/20)
split_idx = int(len(train_data) * 0.8)
train_df = train_data[:split_idx]
test_df = train_data[split_idx:]

# Define feature columns
feature_cols = ["rsi_14", "bb_upper", "bb_middle", "bb_lower", "atr_14"]

# Create validator with LightGBM
validator = SklearnSignalValidator(
    model_type="lightgbm",
    model_params={
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 5
    }
)

# Train
validator.fit(
    X_train=train_df.select(["pair", "timestamp"] + feature_cols),
    y_train=train_df.select("label")
)

# Validate test signals
test_signals = Signals(test_df.select(["pair", "timestamp", "signal_type", "signal"]))
validated_signals = validator.validate_signals(test_signals, test_df.select(["pair", "timestamp"] + feature_cols))

# Filter high-confidence signals
high_confidence = validated_signals.value.filter(
    pl.col("probability_rise") > 0.7  # Only signals with >70% success probability
)
print(f"High confidence signals: {len(high_confidence)}")
```

!!! tip "Model Selection"
    `SklearnSignalValidator` supports multiple models:
    
    - `lightgbm` (recommended for speed)
    - `xgboost` (good accuracy)
    - `random_forest` (sklearn Random Forest)
    - `logistic_regression` (linear baseline)
    - `auto` (automatic model selection via cross-validation)

### Step 7: Backtest Strategy

Simulate trading with your signals:
```python
from signalflow.strategy import BacktestRunner, VirtualSpotExecutor
from signalflow.strategy import TakeProfitStopLossExit

# Configure execution
executor = VirtualSpotExecutor()

# Configure exit rules
exit_rule = TakeProfitStopLossExit(
    take_profit=0.02,  # 2% profit target
    stop_loss=0.01     # 1% stop loss
)

# Create backtest runner
runner = BacktestRunner(
    strategy_id="sma_cross_backtest",
    initial_capital=10000.0,
    executor=executor,
    exit_rule=exit_rule,
    data_key="spot"
)

# Run backtest
final_state = runner.run(
    raw_data=raw_data,
    signals=signals  # Or use validated_signals for filtered backtest
)

# Print results
print(f"Final Portfolio Value: ${final_state.portfolio.total_value():.2f}")
print(f"Total Trades: {len(final_state.trades)}")
print(f"Open Positions: {len(final_state.portfolio.open_positions())}")

# Compute metrics
from signalflow.strategy.metrics import PortfolioMetrics

metrics = PortfolioMetrics.compute(final_state)
print(f"\nBacktest Metrics:")
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
```

### Step 8: Analyze Results

Visualize backtest performance:
```python
import plotly.graph_objects as go
from signalflow.utils.visualization import plot_equity_curve

# Plot equity curve
fig = plot_equity_curve(final_state)
fig.show()

# Plot signals on price chart
from signalflow.utils.visualization import plot_signals_overlay

fig = plot_signals_overlay(
    raw_data=raw_data,
    signals=signals,
    pair="BTCUSDT",
    start=datetime(2024, 11, 1),
    end=datetime(2024, 11, 7)
)
fig.show()
```

---

## Complete Example

Here's the full workflow in one script:
```python
import asyncio
import polars as pl
from pathlib import Path
from datetime import datetime

from signalflow.data import BinanceSpotLoader, RawDataFactory
from signalflow.detector import SmaCrossSignalDetector
from signalflow.target import TripleBarrierLabeler
from signalflow.feature import FeaturePipeline, RsiFeature, BollingerBandsFeature
from signalflow.validator import SklearnSignalValidator
from signalflow.strategy import BacktestRunner, VirtualSpotExecutor, TakeProfitStopLossExit
from signalflow.core import Signals

# 1. Download data
async def main():
    loader = BinanceSpotLoader()
    await loader.download(pairs=["BTCUSDT"], days=30)

asyncio.run(main())

# 2. Load data
raw_data = RawDataFactory.from_duckdb_spot_store(
    spot_store_path=Path("raw_data.duckdb"),
    pairs=["BTCUSDT"],
    start=datetime(2024, 11, 1),
    end=datetime(2024, 12, 1)
)

# 3. Detect signals
detector = SmaCrossSignalDetector(fast_period=20, slow_period=50)
signals = detector.run(raw_data)

# 4. Label for ML (optional)
labeler = TripleBarrierLabeler(take_profit=0.02, stop_loss=0.01)
labeled_signals = labeler.label(signals, raw_data)

# 5. Extract features (optional)
features = FeaturePipeline(features=[
    RsiFeature(period=14),
    BollingerBandsFeature(period=20)
]).run(raw_data_view)

# 6. Train validator (optional)
train_data = labeled_signals.value.filter(
    pl.col("signal_type") != "none"
).join(features, on=["pair", "timestamp"])

validator = SklearnSignalValidator(model_type="lightgbm")
validator.fit(
    X_train=train_data.select(["pair", "timestamp", "rsi_14", "bb_upper", "bb_middle", "bb_lower"]),
    y_train=train_data.select("label")
)

# 7. Backtest
runner = BacktestRunner(
    strategy_id="quickstart",
    initial_capital=10000,
    executor=VirtualSpotExecutor(),
    exit_rule=TakeProfitStopLossExit(take_profit=0.02, stop_loss=0.01)
)

final_state = runner.run(raw_data=raw_data, signals=signals)
print(f"Final Value: ${final_state.portfolio.total_value():.2f}")
```

---

## Next Steps

Congratulations! You've built your first trading strategy with SignalFlow. Here's what to explore next:

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **[User Guide](../guide/overview.md)**

    ---

    Deep dive into core concepts: signals, features, validators, and strategies

-   :material-code-braces:{ .lg .middle } **[API Reference](../api/index.md)**

    ---

    Complete documentation for all classes and methods

-   :material-lightbulb:{ .lg .middle } **[Examples](../examples/index.md)**

    ---

    More advanced examples: ML integration, custom components, live trading

-   :material-cog:{ .lg .middle } **[Configuration](configuration.md)**

    ---

    Customize SignalFlow behavior and component parameters

</div>

---

## Common Patterns

### Using the Component Registry

Register and discover components dynamically:
```python
from signalflow.core import sf_component, default_registry, SfComponentType

# Register custom detector
@sf_component(name="my_detector")
class MyCustomDetector(SignalDetector):
    component_type = SfComponentType.DETECTOR
    
    def detect(self, features, context=None):
        # Your logic
        return Signals(...)

# List available detectors
detectors = default_registry.list(SfComponentType.DETECTOR)
print(f"Available: {detectors}")

# Create from registry
detector = default_registry.create(
    SfComponentType.DETECTOR,
    "my_detector",
    custom_param=42
)
```

### Working with Polars DataFrames

SignalFlow is Polars-first for performance:
```python
import polars as pl

# Filter signals
active_signals = signals.value.filter(
    pl.col("signal_type") != "none"
)

# Group by pair
signals_by_pair = signals.value.group_by("pair").agg([
    pl.count().alias("signal_count"),
    pl.col("signal_type").mode().alias("most_common_signal")
])

# Join with features
enriched = signals.value.join(
    features,
    on=["pair", "timestamp"],
    how="left"
)
```

### Pandas Compatibility

Convert to Pandas when needed:
```python
# Convert Polars to Pandas
pandas_df = signals.value.to_pandas()

# Convert Pandas to Polars
polars_df = pl.from_pandas(pandas_df)
```

---

## Troubleshooting

### No signals detected?

Check your detector parameters and data quality:
```python
# Verify data
print(raw_data['spot'].describe())

# Check for NaN in price columns
print(raw_data['spot'].select([
    pl.col("close").is_null().sum()
]))

# Lower detector thresholds
detector = SmaCrossSignalDetector(
    fast_period=10,  # Lower values = more signals
    slow_period=20
)
```

### Slow performance?

SignalFlow uses Polars for speed, but check:
```python
# Reduce data size
raw_data = RawDataFactory.from_duckdb_spot_store(
    pairs=["BTCUSDT"],  # Single pair
    start=datetime(2024, 12, 1),
    end=datetime(2024, 12, 7)  # One week
)

# Use numba-accelerated labelers
labeler = TripleBarrierLabeler(
    use_numba=True  # Enable JIT compilation
)
```

### Import errors?

Ensure all dependencies are installed:
```bash
# Full installation
pip install signalflow-trading[dev,nn]

# Or install missing packages individually
pip install lightgbm xgboost plotly
```

---

## Getting Help

- **Email**: [pathway2nothing@gmail.com](mailto:pathway2nothing@gmail.com)
- **GitHub Issues**: [Report bugs or request features](https://github.com/pathway2nothing/signalflow-trading/issues)
- **Documentation**: Explore the [User Guide](../guide/overview.md) and [API Reference](../api/index.md)

---

!!! success "You're Ready!"
    You now have a working algorithmic trading strategy. Experiment with different detectors, validators, and parameters to improve performance. Remember: **backtest thoroughly before live trading!**