"""Strategy state persistence backends.

Provides storage implementations for strategy state, positions,
trades, and metrics during backtesting and live trading.

Built-in Backends:
    DuckDbStrategyStore: High-performance local DuckDB storage (default).
    InMemoryStrategyStore: In-memory storage for testing.

Extended Backends (via signalflow-data package):
    SqliteStrategyStore: Lightweight SQLite storage.
    PgStrategyStore: PostgreSQL storage (requires asyncpg/psycopg2).

Install extended backends:
    pip install git+https://github.com/yourorg/sf-data.git

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
from signalflow.data.strategy_store.memory import InMemoryStrategyStore

__all__ = [
    "DuckDbStrategyStore",
    "InMemoryStrategyStore",
    "PgStrategyStore",
    # Extended stores (lazy imports via __getattr__)
    "SqliteStrategyStore",
    "StrategyStore",
]


def __getattr__(name: str) -> object:
    """Lazy import extended stores from signalflow-data if available."""
    _EXTENDED_STORES = {
        "PgStrategyStore": "signalflow.data.strategy_store.pg",
        "SqliteStrategyStore": "signalflow.data.strategy_store.sqlite",
    }

    if name in _EXTENDED_STORES:
        try:
            module_path = _EXTENDED_STORES[name]
            module = __import__(module_path, fromlist=[name])
            return getattr(module, name)
        except ImportError as err:
            raise ImportError(
                f"{name} requires signalflow-data package. Install with:\n"
                f"  pip install git+https://github.com/yourorg/sf-data.git"
            ) from err

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
