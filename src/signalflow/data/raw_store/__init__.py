"""Raw OHLCV data storage backends.

Provides storage implementations for historical market data
with support for multiple data types (spot, futures, perpetual).

Built-in Backends:
    DuckDbRawStore: High-performance local DuckDB storage (default).
    InMemoryRawStore: In-memory storage for testing.

Extended Backends (via signalflow-data package):
    SqliteRawStore: Lightweight SQLite storage.
    PgRawStore: PostgreSQL storage (requires asyncpg/psycopg2).

Install extended backends:
    pip install git+https://github.com/yourorg/sf-data.git

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

__all__ = [
    "DuckDbRawStore",
    "DuckDbSpotStore",
    "InMemoryRawStore",
    "InMemorySpotStore",
    "RawDataStore",
    # Extended stores (lazy imports via __getattr__)
    "SqliteRawStore",
    "SqliteSpotStore",
    "PgRawStore",
    "PgSpotStore",
]


def __getattr__(name: str):  # type: ignore[misc]
    """Lazy import extended stores from signalflow-data if available."""
    _EXTENDED_STORES = {
        "PgRawStore": "signalflow.data.raw_store.pg_stores",
        "PgSpotStore": "signalflow.data.raw_store.pg_stores",
        "SqliteRawStore": "signalflow.data.raw_store.sqlite_stores",
        "SqliteSpotStore": "signalflow.data.raw_store.sqlite_stores",
    }

    if name in _EXTENDED_STORES:
        try:
            module_path = _EXTENDED_STORES[name]
            module = __import__(module_path, fromlist=[name])
            return getattr(module, name)
        except ImportError:
            raise ImportError(
                f"{name} requires signalflow-data package. Install with:\n"
                f"  pip install git+https://github.com/yourorg/sf-data.git"
            )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
