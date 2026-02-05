from signalflow.data.strategy_store.base import StrategyStore
from signalflow.data.strategy_store.duckdb import DuckDbStrategyStore
from signalflow.data.strategy_store.sqlite import SqliteStrategyStore

__all__ = [
    "StrategyStore",
    "DuckDbStrategyStore",
    "SqliteStrategyStore",
]


def __getattr__(name: str):  # type: ignore[misc]
    if name == "PgStrategyStore":
        from signalflow.data.strategy_store.pg import PgStrategyStore

        return PgStrategyStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
