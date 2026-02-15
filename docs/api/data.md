# Data Module

## Data Sources

### Binance

::: signalflow.data.source.binance.BinanceClient
    options:
      show_root_heading: true

::: signalflow.data.source.binance.BinanceSpotLoader
    options:
      show_root_heading: true

::: signalflow.data.source.binance.BinanceFuturesUsdtLoader
    options:
      show_root_heading: true

::: signalflow.data.source.binance.BinanceFuturesCoinLoader
    options:
      show_root_heading: true

### Bybit

::: signalflow.data.source.bybit.BybitClient
    options:
      show_root_heading: true

::: signalflow.data.source.bybit.BybitSpotLoader
    options:
      show_root_heading: true

::: signalflow.data.source.bybit.BybitFuturesLoader
    options:
      show_root_heading: true

::: signalflow.data.source.bybit.BybitFuturesInverseLoader
    options:
      show_root_heading: true

### OKX

::: signalflow.data.source.okx.OkxClient
    options:
      show_root_heading: true

::: signalflow.data.source.okx.OkxSpotLoader
    options:
      show_root_heading: true

::: signalflow.data.source.okx.OkxFuturesLoader
    options:
      show_root_heading: true

### Deribit

::: signalflow.data.source.deribit.DeribitClient
    options:
      show_root_heading: true

::: signalflow.data.source.deribit.DeribitFuturesLoader
    options:
      show_root_heading: true

### Kraken

::: signalflow.data.source.kraken.KrakenClient
    options:
      show_root_heading: true

::: signalflow.data.source.kraken.KrakenSpotLoader
    options:
      show_root_heading: true

::: signalflow.data.source.kraken.KrakenFuturesLoader
    options:
      show_root_heading: true

### Hyperliquid

::: signalflow.data.source.hyperliquid.HyperliquidClient
    options:
      show_root_heading: true

::: signalflow.data.source.hyperliquid.HyperliquidFuturesLoader
    options:
      show_root_heading: true

### WhiteBIT

::: signalflow.data.source.whitebit.WhitebitClient
    options:
      show_root_heading: true

::: signalflow.data.source.whitebit.WhitebitSpotLoader
    options:
      show_root_heading: true

::: signalflow.data.source.whitebit.WhitebitFuturesLoader
    options:
      show_root_heading: true

### Virtual Data

::: signalflow.data.source.virtual.VirtualDataProvider
    options:
      show_root_heading: true

## Storage

::: signalflow.data.raw_store.duckdb_stores.DuckDbSpotStore
    options:
      show_root_heading: true
      show_source: true

## Factory

::: signalflow.data.raw_data_factory.RawDataFactory
    options:
      show_root_heading: true
      show_source: true

## Resampling

Unified OHLCV timeframe resampling with exchange-aware timeframe selection.

```python
from signalflow.data.resample import (
    resample_ohlcv,
    align_to_timeframe,
    detect_timeframe,
    select_best_timeframe,
    can_resample,
)

# Auto-detect and resample to 1h
df_1h = align_to_timeframe(raw_df, target_tf="1h")

# Find best exchange timeframe for download
best = select_best_timeframe("bybit", target_tf="8h")  # "4h"
```

::: signalflow.data.resample.resample_ohlcv
    options:
      show_root_heading: true
      show_source: true

::: signalflow.data.resample.align_to_timeframe
    options:
      show_root_heading: true
      show_source: true

::: signalflow.data.resample.detect_timeframe
    options:
      show_root_heading: true

::: signalflow.data.resample.can_resample
    options:
      show_root_heading: true

::: signalflow.data.resample.select_best_timeframe
    options:
      show_root_heading: true

::: signalflow.data.resample.timeframe_to_minutes
    options:
      show_root_heading: true