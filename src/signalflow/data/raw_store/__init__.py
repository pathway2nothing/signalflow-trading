"""Raw OHLCV data storage backends.

Provides storage implementations for historical market data
with support for multiple data types (spot, futures, perpetual).

Backends:
    DuckDbRawStore: High-performance local DuckDB storage.
    SqliteRawStore: Lightweight SQLite storage (zero dependencies).
    InMemoryRawStore: In-memory storage for testing.
    PgRawStore: PostgreSQL storage (lazy import, optional dependency).

Features:
    - Upsert semantics (INSERT OR REPLACE)
    - Gap detection and filling
    - Dynamic schema based on data type
    - Automatic schema migration

Example:
    ```python
    from signalflow.data.raw_store import DuckDbRawStore
    from pathlib import Path

    store = DuckDbRawStore(db_path=Path("data/spot.duckdb"))
    store.insert_klines("BTCUSDT", klines)
    df = store.load("BTCUSDT", hours=24)
    ```
"""

from signalflow.data.raw_store.base import RawDataStore
from signalflow.data.raw_store.duckdb_stores import DuckDbRawStore, DuckDbSpotStore
from signalflow.data.raw_store.memory_store import InMemoryRawStore, InMemorySpotStore
from signalflow.data.raw_store.sqlite_stores import SqliteRawStore, SqliteSpotStore

__all__ = [
    "DuckDbRawStore",
    "DuckDbSpotStore",
    "InMemoryRawStore",
    "InMemorySpotStore",
    "RawDataStore",
    "SqliteRawStore",
    "SqliteSpotStore",
]


def __getattr__(name: str):  # type: ignore[misc]
    if name in ("PgRawStore", "PgSpotStore"):
        from signalflow.data.raw_store.pg_stores import PgRawStore

        return PgRawStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
