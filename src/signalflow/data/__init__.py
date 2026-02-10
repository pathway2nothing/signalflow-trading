"""SignalFlow Data Module.

Provides data infrastructure for market data and strategy persistence:

Submodules:
    source: Exchange data sources (Binance, Bybit, OKX) and loaders.
    raw_store: Raw OHLCV storage backends (DuckDB, SQLite, PostgreSQL).
    strategy_store: Strategy state persistence backends.

Factories:
    StoreFactory: Creates storage backends for different data types.
    RawDataFactory: Creates RawData containers from stores.

Example:
    ```python
    from signalflow.data import StoreFactory, RawDataFactory
    from datetime import datetime

    # Create a DuckDB store for spot data
    store = StoreFactory.create_raw_store(
        backend="duckdb",
        data_type="spot",
        db_path="data/spot.duckdb",
    )

    # Load data as RawData container
    raw_data = store.to_raw_data(
        pairs=["BTCUSDT", "ETHUSDT"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
    )
    ```
"""

import signalflow.data.source as source
import signalflow.data.raw_store as raw_store
import signalflow.data.strategy_store as strategy_store
from signalflow.data.raw_data_factory import RawDataFactory
from signalflow.data.store_factory import StoreFactory

__all__ = [
    "source",
    "raw_store",
    "strategy_store",
    "RawDataFactory",
    "StoreFactory",
]
