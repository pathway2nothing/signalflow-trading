# Advanced Strategy Components

This guide covers advanced strategy components for position sizing, entry filtering, and signal aggregation.

---

## Overview

SignalFlow provides injectable components for customizing trade execution:

```
Signals (from detectors)
    |
    v
[SignalAggregator] -----> Aggregated Signals
    |
    v
SignalEntryRule.check_entries()
    |
    +---> [EntryFilter(s)] -----> allow_entry() -> bool
    |
    +---> [PositionSizer] -----> compute_size() -> notional
    |
    v
Order with computed qty
```

---

## Position Sizing

Position sizers determine how much capital to allocate per trade.

### Available Sizers

| Sizer | Strategy | Use Case |
|-------|----------|----------|
| `FixedFractionSizer` | Fixed % of equity | Simple, consistent sizing |
| `SignalStrengthSizer` | Scale by probability | More capital on high-confidence signals |
| `KellyCriterionSizer` | Optimal f* formula | Maximize long-term growth |
| `VolatilityTargetSizer` | Inverse volatility | Equal risk per position |
| `RiskParitySizer` | Equal risk budget | Portfolio diversification |
| `MartingaleSizer` | Grid/DCA scaling | Grid trading strategies |

### Quick Examples

```python
from signalflow.strategy.component.sizing import (
    FixedFractionSizer,
    KellyCriterionSizer,
    VolatilityTargetSizer,
    MartingaleSizer,
)

# 2% of equity per trade
sizer = FixedFractionSizer(fraction=0.02)

# Half-Kelly sizing (conservative)
sizer = KellyCriterionSizer(kelly_fraction=0.5)

# Target 1% volatility contribution
sizer = VolatilityTargetSizer(target_volatility=0.01)

# Grid trading: $100 -> $150 -> $225 -> ...
sizer = MartingaleSizer(base_size=100, multiplier=1.5, max_grid_levels=5)
```

### Kelly Criterion

The Kelly Criterion computes optimal position sizing:

$$f^* = \frac{p \cdot b - q}{b}$$

Where:

- $p$ = win probability
- $q$ = 1 - p (loss probability)
- $b$ = payoff ratio (avg win / avg loss)

```python
sizer = KellyCriterionSizer(
    kelly_fraction=0.5,          # Half-Kelly (safer)
    use_signal_probability=True, # Use signal.probability as p
    default_payoff_ratio=1.5,    # Expected win/loss ratio
    max_fraction=0.25,           # Never exceed 25%
)
```

!!! tip "Half-Kelly"
    Full Kelly can be volatile. Half-Kelly (kelly_fraction=0.5) captures most of the edge with less variance.

---

## Entry Filters

Entry filters validate signals before opening positions.

### Available Filters

| Filter | Blocks When | Key Params |
|--------|-------------|------------|
| `RegimeFilter` | Signal doesn't match regime | `allowed_regimes_rise/fall` |
| `VolatilityFilter` | Vol outside range | `min_volatility`, `max_volatility` |
| `DrawdownFilter` | Drawdown exceeds limit | `max_drawdown`, `recovery_threshold` |
| `CorrelationFilter` | Too many correlated positions | `max_correlation` |
| `TimeOfDayFilter` | Outside trading hours | `allowed_hours`, `blocked_hours` |
| `PriceDistanceFilter` | Price too close to last entry | `min_distance_pct` |
| `SignalAccuracyFilter` | Detector accuracy drops | `min_accuracy` |

### Composing Filters

Use `CompositeEntryFilter` to combine multiple filters:

```python
from signalflow.strategy.component.entry import (
    CompositeEntryFilter,
    DrawdownFilter,
    RegimeFilter,
    VolatilityFilter,
    TimeOfDayFilter,
)

# All must pass (AND logic)
composite = CompositeEntryFilter(
    filters=[
        DrawdownFilter(max_drawdown=0.10, recovery_threshold=0.05),
        RegimeFilter(),
        VolatilityFilter(max_volatility=0.03),
        TimeOfDayFilter(blocked_hours=[0, 1, 2, 3, 4, 5]),
    ],
    require_all=True,
)

# Check if entry is allowed
allowed, reason = composite.allow_entry(signal_ctx, state, prices)
if not allowed:
    print(f"Blocked: {reason}")
```

### Drawdown Protection

Pause trading after significant losses:

```python
filter_ = DrawdownFilter(
    max_drawdown=0.10,        # Pause at 10% drawdown
    recovery_threshold=0.05,  # Resume when back to 5%
)
```

The filter maintains state across calls - once paused, it stays paused until recovery.

---

## Signal Aggregation

Combine signals from multiple detectors using voting logic.

### Voting Modes

| Mode | Description |
|------|-------------|
| `MAJORITY` | Most common signal wins (with min agreement) |
| `WEIGHTED` | Probability-weighted average |
| `UNANIMOUS` | All must agree |
| `ANY` | Any non-NONE passes (highest prob wins) |
| `META_LABELING` | Detector direction × validator confidence |

### Examples

```python
from signalflow.strategy.component.entry import SignalAggregator, VotingMode

# Majority voting
agg = SignalAggregator(
    voting_mode=VotingMode.MAJORITY,
    min_agreement=0.6,  # Need 60% agreement
)
combined = agg.aggregate([signals_1, signals_2, signals_3])

# Weighted by custom weights
agg = SignalAggregator(
    voting_mode=VotingMode.WEIGHTED,
    weights=[2.0, 1.0, 1.0],  # First detector 2x weight
)

# Meta-labeling: detector * validator
agg = SignalAggregator(voting_mode=VotingMode.META_LABELING)
combined = agg.aggregate([detector_signals, validator_signals])
# probability = detector_prob * validator_prob
```

---

## Integration with SignalEntryRule

Inject sizers and filters into entry rules:

```python
from signalflow.strategy.component.entry import SignalEntryRule
from signalflow.strategy.component.sizing import VolatilityTargetSizer

entry_rule = SignalEntryRule(
    # Custom position sizer
    position_sizer=VolatilityTargetSizer(target_volatility=0.01),

    # Entry filters
    entry_filters=CompositeEntryFilter(
        filters=[
            DrawdownFilter(max_drawdown=0.10),
            TimeOfDayFilter(allowed_hours=list(range(8, 20))),
        ],
    ),

    # Standard parameters still work
    max_positions_per_pair=2,
    max_total_positions=10,
)
```

---

## Grid Trading Strategy

Complete grid trading setup using `MartingaleSizer` + `PriceDistanceFilter`:

```python
from signalflow.strategy.component.entry import (
    SignalEntryRule,
    CompositeEntryFilter,
    PriceDistanceFilter,
    RegimeFilter,
)
from signalflow.strategy.component.sizing import MartingaleSizer

# Grid configuration
grid_entry = SignalEntryRule(
    position_sizer=MartingaleSizer(
        base_size=200.0,      # First entry: $200
        multiplier=1.5,       # Each level: 1.5x
        max_grid_levels=5,    # Max 5 levels
    ),
    entry_filters=CompositeEntryFilter(
        filters=[
            # Only add when price drops 2%
            PriceDistanceFilter(min_distance_pct=0.02, direction_aware=True),
            RegimeFilter(),
        ],
    ),
    max_positions_per_pair=5,  # Allow grid
)
```

**Grid Progression:**

| Level | Price Drop | Position Size |
|-------|------------|---------------|
| 1 | Entry | $200 |
| 2 | -2% | $300 |
| 3 | -4% | $450 |
| 4 | -6% | $675 |
| 5 | -8% | $1,012 |

---

## Data Requirements

Components access runtime data through `StrategyState`:

```python
# Setup state with required data
state.runtime["atr"] = {"BTCUSDT": 1000.0}  # For VolatilityTargetSizer
state.runtime["regime"] = {"BTCUSDT": "trend_up"}  # For RegimeFilter
state.runtime["correlations"] = {("BTCUSDT", "ETHUSDT"): 0.85}  # For CorrelationFilter
state.metrics["current_drawdown"] = 0.05  # For DrawdownFilter
```

---

## Best Practices

### 1. Start Conservative

Use Half-Kelly and moderate position sizing:

```python
sizer = KellyCriterionSizer(
    kelly_fraction=0.5,
    max_fraction=0.15,  # Cap at 15%
)
```

### 2. Layer Filters

Combine multiple filters for defense in depth:

```python
filters = CompositeEntryFilter(
    filters=[
        DrawdownFilter(max_drawdown=0.10),  # Risk management
        VolatilityFilter(max_volatility=0.05),  # Market conditions
        TimeOfDayFilter(blocked_hours=[0, 1, 2, 3]),  # Liquidity
    ],
)
```

### 3. Test Components Individually

Validate each component before combining:

```python
# Test sizer
signal = SignalContext(pair="BTCUSDT", signal_type="rise", probability=0.8, price=50000)
size = sizer.compute_size(signal, state, prices)
assert size > 0

# Test filter
allowed, reason = filter_.allow_entry(signal, state, prices)
print(f"Allowed: {allowed}, Reason: {reason}")
```

---

## See Also

- **[API Reference](../api/strategy.md)**: Detailed component documentation
- **[Tutorial Notebook](../notebooks/strategy_tutorial.ipynb)**: Interactive examples
- **[Quick Start](../quickstart.md)**: Basic strategy setup
