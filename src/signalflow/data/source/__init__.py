"""Exchange data sources and loaders.

Provides async clients and loaders for downloading historical OHLCV data
from cryptocurrency exchanges.

Supported Exchanges:
    - Binance: Spot, USDT-M Futures, COIN-M Futures
    - Bybit: Spot, Linear Futures
    - OKX: Spot, Perpetual Swaps
    - Deribit: Futures/Perpetuals
    - Kraken: Spot, Futures
    - Hyperliquid: Perpetuals (DEX)
    - WhiteBIT: Spot

Base Classes:
    RawDataSource: Abstract base for exchange API clients.
    RawDataLoader: Abstract base for data loaders with download/sync.

Virtual Data:
    VirtualDataProvider: Synthetic OHLCV data for testing.
    generate_ohlcv: Random walk price series.
    generate_crossover_data: SMA crossover patterns.

Example:
    ```python
    from signalflow.data.source import BinanceClient
    from datetime import datetime

    async with BinanceClient() as client:
        klines = await client.get_klines("BTCUSDT", "1h")
    ```
"""

from signalflow.data.source.base import RawDataSource, RawDataLoader
from signalflow.data.source.binance import (
    BinanceClient,
    BinanceSpotLoader,
    BinanceFuturesUsdtLoader,
    BinanceFuturesCoinLoader,
)
from signalflow.data.source.bybit import (
    BybitClient,
    BybitSpotLoader,
    BybitFuturesLoader,
)
from signalflow.data.source.okx import (
    OkxClient,
    OkxSpotLoader,
    OkxFuturesLoader,
)
from signalflow.data.source.deribit import (
    DeribitClient,
    DeribitFuturesLoader,
)
from signalflow.data.source.kraken import (
    KrakenClient,
    KrakenSpotLoader,
    KrakenFuturesLoader,
)
from signalflow.data.source.hyperliquid import (
    HyperliquidClient,
    HyperliquidFuturesLoader,
)
from signalflow.data.source.whitebit import (
    WhitebitClient,
    WhitebitSpotLoader,
)
from signalflow.data.source.virtual import VirtualDataProvider, generate_ohlcv, generate_crossover_data


__all__ = [
    "RawDataSource",
    "RawDataLoader",
    # Binance
    "BinanceClient",
    "BinanceSpotLoader",
    "BinanceFuturesUsdtLoader",
    "BinanceFuturesCoinLoader",
    # Bybit
    "BybitClient",
    "BybitSpotLoader",
    "BybitFuturesLoader",
    # OKX
    "OkxClient",
    "OkxSpotLoader",
    "OkxFuturesLoader",
    # Deribit
    "DeribitClient",
    "DeribitFuturesLoader",
    # Kraken
    "KrakenClient",
    "KrakenSpotLoader",
    "KrakenFuturesLoader",
    # Hyperliquid
    "HyperliquidClient",
    "HyperliquidFuturesLoader",
    # WhiteBIT
    "WhitebitClient",
    "WhitebitSpotLoader",
    # Virtual
    "VirtualDataProvider",
    "generate_ohlcv",
    "generate_crossover_data",
]
