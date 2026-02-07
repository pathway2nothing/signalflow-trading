# External Model Integration

This guide covers integrating external ML/RL models with SignalFlow for automated trading decisions.

---

## Overview

SignalFlow supports external model integration via a Protocol-based interface:

```
Backtest Bar
    |
    v
Build ModelContext (signals, metrics, positions)
    |
    v
model.decide(context) --> list[StrategyDecision]
    |
    +---> CLOSE/CLOSE_ALL --> ModelExitRule --> Exit Orders
    |
    +---> ENTER --> ModelEntryRule --> Entry Orders
```

**Design Principle**: Models receive signals and metrics, NOT raw OHLCV prices.

---

## Quick Start

### 1. Implement the Protocol

```python
from signalflow.strategy.model import (
    StrategyModel,
    StrategyAction,
    StrategyDecision,
    ModelContext,
)


class MyModel:
    """Your ML/RL model implementing StrategyModel protocol."""

    def decide(self, context: ModelContext) -> list[StrategyDecision]:
        decisions = []

        for row in context.signals.value.iter_rows(named=True):
            prob = row.get("probability", 0.5)

            if prob > 0.7:
                decisions.append(StrategyDecision(
                    action=StrategyAction.ENTER,
                    pair=row["pair"],
                    confidence=prob,
                ))

        return decisions
```

### 2. Create Rules

```python
from signalflow.strategy.model import ModelEntryRule, ModelExitRule

model = MyModel()

entry_rule = ModelEntryRule(
    model=model,
    base_position_size=0.02,
    max_positions=5,
    min_confidence=0.6,
)

exit_rule = ModelExitRule(
    model=model,
    min_confidence=0.7,
)
```

### 3. Run Backtest

```python
from signalflow.strategy.runner import BacktestRunner
from signalflow.strategy.broker import BacktestBroker
from signalflow.strategy.broker.executor import VirtualSpotExecutor

runner = BacktestRunner(
    strategy_id="model_strategy",
    broker=BacktestBroker(executor=VirtualSpotExecutor(fee_rate=0.001)),
    entry_rules=[entry_rule],
    exit_rules=[exit_rule],
    initial_capital=10_000.0,
)

state = runner.run(raw_data, signals)
```

---

## Decision Types

Models return `StrategyDecision` objects with these actions:

| Action | Description | Required Fields |
|--------|-------------|-----------------|
| `ENTER` | Open new position | `pair`, optionally `size_multiplier` |
| `SKIP` | Skip this signal | `pair` |
| `CLOSE` | Close specific position | `pair`, `position_id` |
| `CLOSE_ALL` | Close all positions for pair | `pair` |
| `HOLD` | Do nothing | - |

### Entry Decision

```python
StrategyDecision(
    action=StrategyAction.ENTER,
    pair="BTCUSDT",
    size_multiplier=1.5,  # 1.5x base position size
    confidence=0.85,
    meta={"reason": "high_confidence_signal"},
)
```

### Exit Decision

```python
StrategyDecision(
    action=StrategyAction.CLOSE,
    pair="BTCUSDT",
    position_id="pos_abc123",  # Required for CLOSE
    confidence=0.9,
    meta={"reason": "take_profit"},
)
```

### Close All Positions

```python
StrategyDecision(
    action=StrategyAction.CLOSE_ALL,
    pair="BTCUSDT",  # Close all BTC positions
    confidence=0.8,
    meta={"reason": "risk_off"},
)
```

---

## ModelContext

The context passed to `model.decide()` contains:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `datetime` | Current bar timestamp |
| `signals` | `Signals` | Current bar signals (from detectors) |
| `prices` | `dict[str, float]` | Current prices per pair |
| `positions` | `list[Position]` | Open positions |
| `metrics` | `dict[str, float]` | Portfolio metrics (equity, drawdown, etc.) |
| `runtime` | `dict[str, Any]` | Custom state (regime, ATR, etc.) |

### Accessing Context Data

```python
def decide(self, context: ModelContext) -> list[StrategyDecision]:
    # Check portfolio state
    equity = context.metrics.get("equity", 0)
    drawdown = context.metrics.get("max_drawdown", 0)

    # Risk check
    if drawdown > 0.15:
        return []  # No trading during high drawdown

    # Process signals
    for row in context.signals.value.iter_rows(named=True):
        pair = row["pair"]
        signal_type = row["signal_type"]
        prob = row["probability"]

        # Get current price
        price = context.prices.get(pair, 0)

        # Check existing positions
        pair_positions = [p for p in context.positions if p.pair == pair]

        # Your model logic here...
```

---

## Model-Based Exit Management

Models can manage exits by analyzing open positions:

```python
def decide(self, context: ModelContext) -> list[StrategyDecision]:
    decisions = []

    for pos in context.positions:
        price = context.prices.get(pos.pair, pos.entry_price)
        pnl_pct = (price - pos.entry_price) / pos.entry_price

        # Take profit
        if pnl_pct > 0.05:  # 5% profit
            decisions.append(StrategyDecision(
                action=StrategyAction.CLOSE,
                pair=pos.pair,
                position_id=pos.id,
                confidence=0.9,
                meta={"reason": "model_take_profit"},
            ))

        # Stop loss
        elif pnl_pct < -0.02:  # 2% loss
            decisions.append(StrategyDecision(
                action=StrategyAction.CLOSE,
                pair=pos.pair,
                position_id=pos.id,
                confidence=0.95,
                meta={"reason": "model_stop_loss"},
            ))

    return decisions
```

---

## Combining with Traditional Rules

Model rules work alongside traditional rules:

```python
from signalflow.strategy.component.exit import TakeProfitStopLossExit

runner = BacktestRunner(
    strategy_id="hybrid_strategy",
    broker=broker,
    entry_rules=[
        ModelEntryRule(model=model, base_position_size=0.02),
    ],
    exit_rules=[
        ModelExitRule(model=model),  # Model-based exits first
        TakeProfitStopLossExit(      # Fallback TP/SL
            take_profit_pct=0.05,
            stop_loss_pct=0.03,
        ),
    ],
    initial_capital=10_000.0,
)
```

Exit rules are processed in order - model exits run first, then traditional rules catch remaining positions.

---

## Decision Caching

The model is called **once per bar**. Decisions are cached in `state.runtime`:

```
Bar Start
    |
    v
ExitRule.check_exits()
    ├── Check cache for decisions
    ├── If empty: call model.decide(), cache result
    └── Process CLOSE/CLOSE_ALL decisions
    |
    v
EntryRule.check_entries()
    ├── Check cache (uses cached decisions)
    └── Process ENTER decisions
    |
    v
state.reset_tick_cache() clears cache for next bar
```

This ensures consistent decisions across entry and exit processing.

---

## Training Data Export

Export backtest results for model training:

```python
from pathlib import Path
from signalflow.strategy.exporter import BacktestExporter

exporter = BacktestExporter()

# During backtest
for ts in timestamps:
    # ... process bar ...
    exporter.export_bar(ts, signals, state.metrics, state)

# When positions close
exporter.export_position_close(position, exit_time, exit_price, "take_profit")

# Write to disk
exporter.finalize(Path("./training_data"))
```

### Exported Files

**bars.parquet** - Per-bar state:
```python
import polars as pl

bars = pl.read_parquet("./training_data/bars.parquet")
# Columns: timestamp, pair, signal_type, probability,
#          metric_equity, metric_max_drawdown, ...
```

**trades.parquet** - Completed trades:
```python
trades = pl.read_parquet("./training_data/trades.parquet")
# Columns: position_id, pair, entry_time, exit_time,
#          entry_price, exit_price, realized_pnl, exit_reason, ...
```

---

## RL Model Example

Complete example with reinforcement learning patterns:

```python
import numpy as np
from signalflow.strategy.model import (
    StrategyAction,
    StrategyDecision,
    ModelContext,
)


class RLTradingModel:
    """RL model for trading decisions."""

    def __init__(self, model_path: str, epsilon: float = 0.0):
        self.model = self._load_model(model_path)
        self.epsilon = epsilon  # Exploration rate

    def decide(self, context: ModelContext) -> list[StrategyDecision]:
        decisions = []

        # Build state features
        features = self._build_features(context)

        # Get Q-values from model
        q_values = self.model.predict(features)

        # Epsilon-greedy action selection (for training)
        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(len(q_values))
        else:
            action_idx = np.argmax(q_values)

        # Map to trading action
        action = self._map_action(action_idx, context)
        if action:
            decisions.append(action)

        return decisions

    def _build_features(self, context: ModelContext) -> np.ndarray:
        """Build feature vector from context."""
        features = []

        # Portfolio state
        features.append(context.metrics.get("equity", 10000) / 10000)
        features.append(context.metrics.get("max_drawdown", 0))

        # Position count
        features.append(len(context.positions) / 10)

        # Signal features
        for row in context.signals.value.head(5).iter_rows(named=True):
            features.append(row.get("probability", 0.5))
            features.append(1 if row.get("signal_type") == "rise" else -1)

        # Pad if fewer signals
        while len(features) < 15:
            features.append(0)

        return np.array(features[:15])

    def _map_action(
        self, action_idx: int, context: ModelContext
    ) -> StrategyDecision | None:
        """Map action index to StrategyDecision."""
        # 0: HOLD, 1: ENTER, 2: CLOSE_ALL
        if action_idx == 1 and context.signals.value.height > 0:
            row = context.signals.value.row(0, named=True)
            return StrategyDecision(
                action=StrategyAction.ENTER,
                pair=row["pair"],
                confidence=0.8,
            )
        elif action_idx == 2 and context.positions:
            return StrategyDecision(
                action=StrategyAction.CLOSE_ALL,
                pair=context.positions[0].pair,
                confidence=0.9,
            )
        return None
```

---

## Best Practices

### 1. Confidence Thresholds

Set appropriate confidence thresholds:

```python
entry_rule = ModelEntryRule(
    model=model,
    min_confidence=0.6,  # Only act on confident entries
)

exit_rule = ModelExitRule(
    model=model,
    min_confidence=0.7,  # Higher threshold for exits
)
```

### 2. Position Size Scaling

Use `size_multiplier` to scale by confidence:

```python
StrategyDecision(
    action=StrategyAction.ENTER,
    pair="BTCUSDT",
    size_multiplier=min(confidence, 1.5),  # Cap at 1.5x
    confidence=confidence,
)
```

### 3. Risk Management in Model

Check portfolio state before trading:

```python
def decide(self, context: ModelContext) -> list[StrategyDecision]:
    # Skip during high drawdown
    if context.metrics.get("max_drawdown", 0) > 0.15:
        return []

    # Limit position count
    if len(context.positions) >= 5:
        return []  # Or only return exit decisions

    # Your trading logic...
```

### 4. Meta in Decisions

Include debugging info in `meta`:

```python
StrategyDecision(
    action=StrategyAction.ENTER,
    pair="BTCUSDT",
    confidence=0.85,
    meta={
        "model_version": "v2.1",
        "signal_type": "rise",
        "features": {"rsi": 35, "trend": "up"},
    },
)
```

---

## See Also

- **[API Reference](../api/strategy.md#external-model-integration)**: Detailed component documentation
- **[Advanced Strategies](advanced-strategies.md)**: Position sizing and entry filters
- **[Quick Start](../quickstart.md)**: Basic strategy setup
