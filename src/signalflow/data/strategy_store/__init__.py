"""Strategy state persistence backends.

Provides storage implementations for strategy state, positions,
trades, and metrics during backtesting and live trading.

Backends:
    DuckDbStrategyStore: High-performance local DuckDB storage.
    SqliteStrategyStore: Lightweight SQLite storage.
    InMemoryStrategyStore: In-memory storage for testing.
    PgStrategyStore: PostgreSQL storage (lazy import, optional).

Features:
    - State snapshots with upsert semantics
    - Position history tracking
    - Trade log (immutable append)
    - Metrics time series

Example:
    ```python
    from signalflow.data.strategy_store import DuckDbStrategyStore
    from signalflow.core import StrategyState

    store = DuckDbStrategyStore("backtest.duckdb")
    store.init()

    state = StrategyState(strategy_id="my_strategy")
    store.save_state(state)
    loaded = store.load_state("my_strategy")
    ```
"""

from signalflow.data.strategy_store.base import StrategyStore
from signalflow.data.strategy_store.duckdb import DuckDbStrategyStore
from signalflow.data.strategy_store.sqlite import SqliteStrategyStore
from signalflow.data.strategy_store.memory import InMemoryStrategyStore

__all__ = [
    "StrategyStore",
    "DuckDbStrategyStore",
    "SqliteStrategyStore",
    "InMemoryStrategyStore",
]


def __getattr__(name: str):  # type: ignore[misc]
    if name == "PgStrategyStore":
        from signalflow.data.strategy_store.pg import PgStrategyStore

        return PgStrategyStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
