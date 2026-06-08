---
title: State Persistence
description: StateManager with Redis, DuckDB, and Memory backends for crash recovery
---

# State Persistence

!!! warning "Source of truth: the event log"
    The canonical state model is **event-sourced**: the portfolio changes only
    through fills, so the append-only trade log (`StrategyStore.append_trade` /
    `read_trades`) is the source of truth and the saved state is a derived
    snapshot cache — verify it with `StrategyStore.verify_snapshot`, replay it
    with `core.fold`.

    ```python
    from signalflow.core import fold
    portfolio = fold(store.read_trades("my_bot"), initial_cash=10_000.0)
    assert store.verify_snapshot("my_bot", initial_cash=10_000.0)
    ```

!!! note "StateManager is live-trading WIP"
    `StateManager` and its Redis/DuckDB backends below are **not yet wired** and
    have moved to `signalflow.strategy.live.state`
    (`from signalflow.strategy.live.state import StateManager`). They still use a
    legacy position model that will be migrated onto the canonical
    `signalflow.core.Position` when live trading lands. The reconciliation port
    that verifies internal vs exchange event logs lives in
    `signalflow.strategy.live.reconciliation`.

The `StateManager` provides async state persistence for live and paper trading.
It tracks open positions, pending orders, risk state, and signal deduplication
across three backends: **Redis**, **DuckDB**, and **Memory**.

---

## Quick Start

```python
from signalflow.strategy.live.state import StateManager  # live-trading WIP

config = {
    "backend": "duckdb",
    "duckdb": {"path": "state/{flow_id}.db"},
    "recovery": {"mode": "sync"},
}

async with StateManager.from_config(flow_id="my_bot", config=config) as mgr:
    # Save position
    await mgr.save_position(Position(
        id="pos_001", pair="BTC/USDT", side="long",
        size=0.5, entry_price=50000, entry_ts=datetime.now(),
    ))

    # Check positions
    positions = await mgr.get_positions()
```

---

## Backends

| Backend | Best For | Persistence | Multi-Process |
|---------|---------|-------------|---------------|
| **Redis** | Production, multi-bot | RDB/AOF snapshots | Yes |
| **DuckDB** | Single bot, embedded | Always on disk | No |
| **Memory** | Testing, development | None | No |

### Redis

```python
config = {
    "backend": "redis",
    "redis": {"url": "redis://localhost:6379"},
}
```

Key schema: `sf:{flow_id}:{category}:{type}`

- `sf:bot:positions:open` — Hash of open positions
- `sf:bot:risk:daily` — Daily PnL and trade count
- `sf:bot:signals:cooldowns` — Pair cooldown expiry times
- `sf:bot:execution:heartbeat` — Liveness timestamp

### DuckDB

```python
config = {
    "backend": "duckdb",
    "duckdb": {"path": "state/{flow_id}.db"},
}
```

Creates tables: `positions`, `pending_orders`, `risk_state`, `signal_state`, `heartbeat`.
The `{flow_id}` placeholder is replaced automatically.

### Memory

```python
config = {"backend": "memory"}
```

No persistence — state is lost on restart. Useful for testing.

---

## State Types

### Position

Tracks an open trading position:

```python
from signalflow.strategy import Position

pos = Position(
    id="pos_001",
    pair="BTC/USDT",
    side="long",        # "long" or "short"
    size=0.5,
    entry_price=50000,
    entry_ts=datetime.now(),
    tp=55000,           # Take profit (optional)
    sl=45000,           # Stop loss (optional)
    metadata={},        # Custom data
)
```

### RiskState

Circuit breaker and daily risk tracking:

```python
from signalflow.strategy import RiskState

risk = RiskState(
    daily_pnl=0.0,
    daily_trades=0,
    consecutive_losses=0,
    current_drawdown=0.0,
    peak_equity=0.0,
    circuit_breaker_active=False,
)
```

### SignalState

Signal deduplication and cooldown tracking:

```python
from signalflow.strategy import SignalState
# Tracks last_processed_ts, cooldowns per pair, recent_signal_ids
```

---

## Position Management

```python
async with StateManager.from_config(flow_id="bot", config=cfg) as mgr:
    # Save
    await mgr.save_position(position)

    # Get all
    positions = await mgr.get_positions()

    # Remove
    await mgr.remove_position("pos_001")
```

---

## Risk Management

### Daily PnL

```python
risk = await mgr.update_daily_pnl(pnl_change=100.50)
print(f"Daily PnL: {risk.daily_pnl}")
```

### Circuit Breaker

```python
from datetime import timedelta

# Trigger
await mgr.trigger_circuit_breaker(
    reason="Daily loss limit exceeded",
    duration=timedelta(hours=2),
)

# Check before trading
if await mgr.check_circuit_breaker():
    print("Trading paused — circuit breaker active")
    return
```

---

## Signal Deduplication

```python
# Mark signal as processed
await mgr.mark_signal_processed(
    signal_id="sig_789",
    pair="BTC/USDT",
    timestamp=datetime.now(),
)

# Check before processing
if await mgr.is_signal_processed("sig_789"):
    return  # Skip duplicate

# Set cooldown after entry
await mgr.set_cooldown("BTC/USDT", duration=timedelta(minutes=5))

if await mgr.is_on_cooldown("BTC/USDT"):
    return  # Pair on cooldown
```

---

## Heartbeat & Staleness

```python
# Update heartbeat (call periodically)
await mgr.heartbeat()

# Check if state is stale (e.g. after crash)
if await mgr.check_stale(max_age=timedelta(hours=24)):
    print("State is stale — recovery needed")
```

---

## Recovery Modes

Configure how the bot handles restarts:

| Mode | Behavior |
|------|----------|
| `sync` | Sync state with exchange after restart |
| `restore` | Restore from persistence only |
| `close_all` | Close all positions on restart |
| `manual` | Require manual intervention |

### Orphan Position Handling

When exchange has positions not tracked in state:

| Action | Behavior |
|--------|----------|
| `close` | Close orphaned positions |
| `adopt` | Adopt into state |
| `manual` | Manual intervention required |

```yaml
# YAML configuration
state:
  backend: redis
  redis:
    url: redis://localhost:6379
  recovery:
    mode: sync
    orphan_positions: close
    max_state_age: 24h
```

---

## Full Configuration Reference

| Option | Values | Default |
|--------|--------|---------|
| `backend` | `redis`, `duckdb`, `memory` | `memory` |
| `redis.url` | URL string | `redis://localhost:6379` |
| `redis.key_prefix` | string | `sf` |
| `duckdb.path` | path (supports `{flow_id}`) | `state/{flow_id}.db` |
| `recovery.mode` | `sync`, `restore`, `close_all`, `manual` | `sync` |
| `recovery.orphan_positions` | `close`, `adopt`, `manual` | `close` |
| `recovery.max_state_age` | duration string | `24h` |

---

## Imports

```python
from signalflow.strategy import (
    StateManager,
    StateConfig,
    StateBackend,
    Position,
    RiskState,
    SignalState,
)
```
