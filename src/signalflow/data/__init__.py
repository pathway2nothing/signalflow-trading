"""SignalFlow Data Module.

Provides data infrastructure for market data and strategy persistence:

Submodules:
    source: Exchange data sources (Binance, Bybit, OKX) and loaders.
    raw_store: Raw OHLCV storage backends (DuckDB, SQLite, PostgreSQL).
    strategy_store: Strategy state persistence backends.
    resample: OHLCV timeframe resampling and alignment utilities.

Factories:
    StoreFactory: Creates storage backends for different data types.
    RawDataFactory: Creates RawData containers from stores with
        optional auto-resampling via ``target_timeframe``.

Timeframe handling:
    Set ``timeframe`` once — data is auto-resampled at load time.
    Works through ``sf.load()``, ``sf.Backtest().data()``, and YAML config.

Example:
    ```python
    from signalflow.data import StoreFactory, RawDataFactory
    from datetime import datetime

    # Load with auto-resample to 1h
    raw_data = RawDataFactory.from_stores(
        stores=[store],
        pairs=["BTCUSDT", "ETHUSDT"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
        target_timeframe="1h",  # auto-resample from source tf
    )

    # Or use resample utilities directly
    from signalflow.data import align_to_timeframe, select_best_timeframe

    df_4h = align_to_timeframe(raw_df, "4h")
    best_tf = select_best_timeframe("bybit", "8h")  # → "4h"
    ```
"""

import signalflow.data.raw_store as raw_store
import signalflow.data.source as source
import signalflow.data.strategy_store as strategy_store
from signalflow.data.raw_data_factory import RawDataFactory
from signalflow.data.resample import (
    EXCHANGE_TIMEFRAMES,
    TIMEFRAME_MINUTES,
    align_to_timeframe,
    can_resample,
    detect_timeframe,
    resample_ohlcv,
    select_best_timeframe,
)
from signalflow.data.store_factory import StoreFactory

__all__ = [
    "EXCHANGE_TIMEFRAMES",
    "TIMEFRAME_MINUTES",
    "RawDataFactory",
    "StoreFactory",
    "align_to_timeframe",
    "can_resample",
    "detect_timeframe",
    "raw_store",
    "resample_ohlcv",
    "select_best_timeframe",
    "source",
    "strategy_store",
]
