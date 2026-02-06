# Strategy Module

The strategy module provides components for backtesting and live trading execution.

---

## Architecture

```mermaid
flowchart LR
    A[Signals] --> B[SignalAggregator]
    B --> C[EntryFilter]
    C --> D[PositionSizer]
    D --> E[Entry Rules]
    E --> F[Broker]
    F --> G[Exit Rules]

    style B fill:#16a34a,stroke:#22c55e,color:#fff
    style C fill:#ea580c,stroke:#f97316,color:#fff
    style D fill:#2563eb,stroke:#3b82f6,color:#fff
```

---

## Execution

::: signalflow.strategy.runner.backtest_runner.BacktestRunner
    options:
      show_root_heading: true
      show_source: true

---

## Entry Rules

::: signalflow.strategy.component.entry.signal.SignalEntryRule
    options:
      show_root_heading: true
      show_source: true

::: signalflow.strategy.component.entry.fixed_size.FixedSizeEntryRule
    options:
      show_root_heading: true
      show_source: true

---

## Exit Rules

::: signalflow.strategy.component.exit.tp_sl.TakeProfitStopLossExit
    options:
      show_root_heading: true
      show_source: true

---

## Position Sizing

Position sizers compute the notional value (in quote currency) for trades based on signal strength, portfolio state, and market conditions.

### Base Classes

::: signalflow.strategy.component.sizing.base.SignalContext
    options:
      show_root_heading: true
      show_source: true

::: signalflow.strategy.component.sizing.base.PositionSizer
    options:
      show_root_heading: true
      show_source: true

### Available Sizers

#### FixedFractionSizer

Allocate a fixed percentage of equity per trade.

::: signalflow.strategy.component.sizing.fixed_fraction.FixedFractionSizer
    options:
      show_root_heading: true
      show_source: true

#### SignalStrengthSizer

Scale position size by signal probability.

::: signalflow.strategy.component.sizing.signal_strength.SignalStrengthSizer
    options:
      show_root_heading: true
      show_source: true

#### KellyCriterionSizer

Optimal sizing using the Kelly Criterion formula.

::: signalflow.strategy.component.sizing.kelly.KellyCriterionSizer
    options:
      show_root_heading: true
      show_source: true

#### VolatilityTargetSizer

Size positions to achieve target volatility contribution.

::: signalflow.strategy.component.sizing.volatility_target.VolatilityTargetSizer
    options:
      show_root_heading: true
      show_source: true

#### RiskParitySizer

Equal risk contribution across positions.

::: signalflow.strategy.component.sizing.risk_parity.RiskParitySizer
    options:
      show_root_heading: true
      show_source: true

#### MartingaleSizer

Grid/DCA strategy with increasing position sizes.

::: signalflow.strategy.component.sizing.martingale.MartingaleSizer
    options:
      show_root_heading: true
      show_source: true

### Usage Example

```python
from signalflow.strategy.component.sizing import (
    FixedFractionSizer,
    KellyCriterionSizer,
    VolatilityTargetSizer,
    SignalContext,
)
from signalflow.core import StrategyState

# Create test state
state = StrategyState(strategy_id="demo")
state.portfolio.cash = 10_000.0
prices = {"BTCUSDT": 50000.0}

# Signal context
signal = SignalContext(
    pair="BTCUSDT",
    signal_type="rise",
    probability=0.75,
    price=50000.0,
)

# Fixed 2% of equity per trade
sizer = FixedFractionSizer(fraction=0.02)
notional = sizer.compute_size(signal, state, prices)  # $200

# Half-Kelly sizing
kelly = KellyCriterionSizer(kelly_fraction=0.5, default_win_rate=0.55)
notional = kelly.compute_size(signal, state, prices)

# Volatility targeting (requires ATR in state.runtime)
state.runtime["atr"] = {"BTCUSDT": 1000.0}  # 2% ATR
vol_sizer = VolatilityTargetSizer(target_volatility=0.01)
notional = vol_sizer.compute_size(signal, state, prices)
```

---

## Entry Filters

Entry filters provide pre-trade validation to improve signal quality.
All filters return `(allowed: bool, reason: str)` tuples.

### Base Classes

::: signalflow.strategy.component.entry.filters.EntryFilter
    options:
      show_root_heading: true
      show_source: true

::: signalflow.strategy.component.entry.filters.CompositeEntryFilter
    options:
      show_root_heading: true
      show_source: true

### Available Filters

#### RegimeFilter

Only enter when market regime matches signal direction.

::: signalflow.strategy.component.entry.filters.RegimeFilter
    options:
      show_root_heading: true
      show_source: true

#### VolatilityFilter

Skip entries in extreme volatility conditions.

::: signalflow.strategy.component.entry.filters.VolatilityFilter
    options:
      show_root_heading: true
      show_source: true

#### DrawdownFilter

Pause trading after significant drawdown.

::: signalflow.strategy.component.entry.filters.DrawdownFilter
    options:
      show_root_heading: true
      show_source: true

#### CorrelationFilter

Avoid concentrated positions in correlated assets.

::: signalflow.strategy.component.entry.filters.CorrelationFilter
    options:
      show_root_heading: true
      show_source: true

#### TimeOfDayFilter

Restrict trading to specific hours.

::: signalflow.strategy.component.entry.filters.TimeOfDayFilter
    options:
      show_root_heading: true
      show_source: true

#### PriceDistanceFilter

Filter entries based on price distance from existing positions.

::: signalflow.strategy.component.entry.filters.PriceDistanceFilter
    options:
      show_root_heading: true
      show_source: true

#### SignalAccuracyFilter

Pause trading when signal accuracy drops below threshold.

::: signalflow.strategy.component.entry.filters.SignalAccuracyFilter
    options:
      show_root_heading: true
      show_source: true

### Usage Example

```python
from signalflow.strategy.component.entry import (
    CompositeEntryFilter,
    DrawdownFilter,
    RegimeFilter,
    VolatilityFilter,
    TimeOfDayFilter,
)

# Combine multiple filters (AND logic)
composite = CompositeEntryFilter(
    filters=[
        DrawdownFilter(max_drawdown=0.10, recovery_threshold=0.05),
        RegimeFilter(),
        VolatilityFilter(max_volatility=0.03),
        TimeOfDayFilter(blocked_hours=[0, 1, 2, 3]),  # Skip overnight
    ],
    require_all=True,  # All must pass
)

allowed, reason = composite.allow_entry(signal, state, prices)
if not allowed:
    print(f"Entry rejected: {reason}")
```

---

## Signal Aggregation

Combine signals from multiple detectors using voting or weighting logic.

### VotingMode

::: signalflow.strategy.component.entry.aggregation.VotingMode
    options:
      show_root_heading: true

| Mode | Description |
|------|-------------|
| `MAJORITY` | Most common signal type wins (requires min_agreement) |
| `WEIGHTED` | Weighted average of probabilities |
| `UNANIMOUS` | All detectors must agree |
| `ANY` | Any non-NONE signal passes (highest probability wins) |
| `META_LABELING` | Detector direction × validator probability |

### SignalAggregator

::: signalflow.strategy.component.entry.aggregation.SignalAggregator
    options:
      show_root_heading: true
      show_source: true

### Usage Examples

```python
from signalflow.strategy.component.entry import SignalAggregator, VotingMode

# Majority voting: signal needs >50% agreement
aggregator = SignalAggregator(
    voting_mode=VotingMode.MAJORITY,
    min_agreement=0.5,
)
combined = aggregator.aggregate([signals_1, signals_2, signals_3])

# Weighted voting with custom weights
aggregator = SignalAggregator(
    voting_mode=VotingMode.WEIGHTED,
    weights=[2.0, 1.0, 1.0],  # First detector weighted 2x
    probability_threshold=0.6,
)
combined = aggregator.aggregate([primary_signals, secondary_1, secondary_2])

# Meta-labeling: detector direction × validator confidence
aggregator = SignalAggregator(
    voting_mode=VotingMode.META_LABELING,
    probability_threshold=0.5,
)
combined = aggregator.aggregate([detector_signals, validator_signals])
# Combined probability = detector_prob * validator_prob

# Unanimous: all must agree for high-conviction trades
aggregator = SignalAggregator(
    voting_mode=VotingMode.UNANIMOUS,
    probability_threshold=0.7,
)
combined = aggregator.aggregate([detector_1, detector_2, detector_3])
```

---

## Integration with SignalEntryRule

Position sizers and entry filters can be injected into `SignalEntryRule`:

```python
from signalflow.strategy.runner import BacktestRunner
from signalflow.strategy.component.entry import (
    SignalEntryRule,
    CompositeEntryFilter,
    DrawdownFilter,
    RegimeFilter,
)
from signalflow.strategy.component.sizing import KellyCriterionSizer
from signalflow.strategy.component.exit import TakeProfitStopLossExit
from signalflow.strategy.broker import BacktestBroker
from signalflow.strategy.broker.executor import VirtualSpotExecutor

# Create advanced entry rule
entry_rule = SignalEntryRule(
    position_sizer=KellyCriterionSizer(kelly_fraction=0.5),
    entry_filters=CompositeEntryFilter(
        filters=[
            DrawdownFilter(max_drawdown=0.10),
            RegimeFilter(),
        ],
    ),
)

# Run backtest
runner = BacktestRunner(
    strategy_id="advanced_strategy",
    broker=BacktestBroker(executor=VirtualSpotExecutor(fee_rate=0.001)),
    entry_rules=[entry_rule],
    exit_rules=[TakeProfitStopLossExit(take_profit_pct=0.02, stop_loss_pct=0.01)],
    initial_capital=10_000.0,
)

state = runner.run(raw_data=raw_data, signals=signals)
```

---

## Grid Trading Example

Combine `MartingaleSizer` with `PriceDistanceFilter` for grid strategies:

```python
from signalflow.strategy.component.entry import (
    SignalEntryRule,
    PriceDistanceFilter,
)
from signalflow.strategy.component.sizing import MartingaleSizer

# Grid strategy: buy more as price drops
entry_rule = SignalEntryRule(
    position_sizer=MartingaleSizer(
        base_size=100.0,      # Start with $100
        multiplier=1.5,       # Increase 50% per level
        max_grid_levels=5,    # Max 5 levels
    ),
    entry_filters=PriceDistanceFilter(
        min_distance_pct=0.02,  # 2% price drop between levels
        direction_aware=True,
    ),
    max_positions_per_pair=5,
)
```

---

## Data Sources

Components access data through `StrategyState`:

| Component | Data Source | Key |
|-----------|-------------|-----|
| `VolatilityTargetSizer` | ATR values | `state.runtime["atr"]` |
| `RiskParitySizer` | ATR values | `state.runtime["atr"]` |
| `DrawdownFilter` | Current drawdown | `state.metrics["current_drawdown"]` |
| `VolatilityFilter` | ATR values | `state.runtime["atr"]` |
| `RegimeFilter` | Market regime | `state.runtime["regime"]` |
| `CorrelationFilter` | Correlation matrix | `state.runtime["correlations"]` |
| `SignalAccuracyFilter` | Accuracy metrics | `state.runtime["signal_accuracy"]` |

Populate these during backtest:

```python
# Example: populate runtime data before entry rule
def on_bar_hook(state, timestamp, prices):
    state.runtime["atr"] = calculate_atr(prices)
    state.runtime["regime"] = detect_regime(prices)
```

---

## Full Backtest Example

Complete example using signal aggregation, entry filters, and position sizing:

```python
from datetime import datetime
from pathlib import Path

import polars as pl
from signalflow.data.raw_store import DuckDbRawStore
from signalflow.data.source import VirtualDataProvider
from signalflow.data import RawDataFactory
from signalflow.detector import ExampleSmaCrossDetector
from signalflow.strategy.broker import BacktestBroker
from signalflow.strategy.broker.executor import VirtualSpotExecutor
from signalflow.strategy.runner import OptimizedBacktestRunner
from signalflow.strategy.component.entry import (
    SignalEntryRule,
    SignalAggregator,
    VotingMode,
    CompositeEntryFilter,
    DrawdownFilter,
    TimeOfDayFilter,
)
from signalflow.strategy.component.sizing import VolatilityTargetSizer
from signalflow.strategy.component.exit import TakeProfitStopLossExit

# 1. Generate synthetic data
PAIRS = ["BTCUSDT", "ETHUSDT"]
START = datetime(2025, 1, 1)

spot_store = DuckDbRawStore(db_path=Path("backtest.duckdb"), timeframe="1m")
provider = VirtualDataProvider(store=spot_store, seed=42)
provider.download(pairs=PAIRS, n_bars=5000, start=START)

# 2. Load data
raw_data = RawDataFactory.from_duckdb_spot_store(
    spot_store_path=Path("backtest.duckdb"),
    pairs=PAIRS,
    start=START,
    end=datetime(2025, 1, 4),
)

# 3. Create multiple detectors
detector_fast = ExampleSmaCrossDetector(fast_period=10, slow_period=30)
detector_slow = ExampleSmaCrossDetector(fast_period=20, slow_period=50)

signals_fast = detector_fast.run(raw_data.view())
signals_slow = detector_slow.run(raw_data.view())

# 4. Aggregate signals (unanimous agreement)
aggregator = SignalAggregator(
    voting_mode=VotingMode.UNANIMOUS,
    probability_threshold=0.0,
)
signals = aggregator.aggregate([signals_fast, signals_slow])

# 5. Configure entry rule with sizer and filters
entry_rule = SignalEntryRule(
    position_sizer=VolatilityTargetSizer(
        target_volatility=0.015,
        default_volatility_pct=0.02,
        max_fraction=0.15,
    ),
    entry_filters=CompositeEntryFilter(
        filters=[
            DrawdownFilter(max_drawdown=0.10),
            TimeOfDayFilter(allowed_hours=list(range(6, 22))),
        ],
    ),
    max_positions_per_pair=1,
    max_total_positions=5,
)

# 6. Run backtest
runner = OptimizedBacktestRunner(
    strategy_id="advanced_strategy",
    broker=BacktestBroker(executor=VirtualSpotExecutor(fee_rate=0.001)),
    entry_rules=[entry_rule],
    exit_rules=[TakeProfitStopLossExit(take_profit_pct=0.02, stop_loss_pct=0.015)],
    initial_capital=10_000.0,
)

state = runner.run(raw_data, signals)
results = runner.get_results()

print(f"Total Return: {results.get('final_return', 0) * 100:.2f}%")
print(f"Max Drawdown: {results.get('max_drawdown', 0) * 100:.2f}%")
print(f"Win Rate: {results.get('win_rate', 0) * 100:.1f}%")

# Cleanup
spot_store.close()
```

---

## See Also

- **[Tutorial Notebook](../notebooks/strategy_tutorial.ipynb)**: Interactive examples
- **[Quick Start](../quickstart.md)**: Basic strategy setup
- **[Core API](core.md)**: `StrategyState`, `Portfolio`, `Position`
