from signalflow.data.raw_store.base import RawDataStore
from signalflow.data.raw_store.duckdb_stores import DuckDbSpotStore
from signalflow.data.raw_store.sqlite_stores import SqliteSpotStore

__all__ = [
    "RawDataStore",
    "DuckDbSpotStore",
    "SqliteSpotStore",
]


def __getattr__(name: str):  # type: ignore[misc]
    if name == "PgSpotStore":
        from signalflow.data.raw_store.pg_stores import PgSpotStore

        return PgSpotStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
