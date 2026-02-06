from signalflow.data.raw_store.base import RawDataStore
from signalflow.data.raw_store.duckdb_stores import DuckDbRawStore, DuckDbSpotStore
from signalflow.data.raw_store.sqlite_stores import SqliteRawStore, SqliteSpotStore
from signalflow.data.raw_store.memory_store import InMemoryRawStore, InMemorySpotStore

__all__ = [
    "RawDataStore",
    "DuckDbRawStore",
    "DuckDbSpotStore",
    "SqliteRawStore",
    "SqliteSpotStore",
    "InMemoryRawStore",
    "InMemorySpotStore",
]


def __getattr__(name: str):  # type: ignore[misc]
    if name in ("PgRawStore", "PgSpotStore"):
        from signalflow.data.raw_store.pg_stores import PgRawStore

        return PgRawStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
