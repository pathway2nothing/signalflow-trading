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
