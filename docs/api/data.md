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