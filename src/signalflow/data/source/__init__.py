"""Exchange data sources and loaders.

Provides async clients and loaders for downloading historical OHLCV data
from cryptocurrency exchanges.

Supported Exchanges:
    - Binance: Spot, USDT-M Futures, COIN-M Futures
    - Bybit: Spot, Linear Futures, Inverse Futures
    - OKX: Spot, Perpetual Swaps
    - Deribit: Futures/Perpetuals
    - Kraken: Spot, Futures
    - Hyperliquid: Perpetuals (DEX)
    - WhiteBIT: Spot, Futures

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

from signalflow.data.source.base import RawDataLoader, RawDataSource
from signalflow.data.source.binance import (
    BinanceClient,
    BinanceFuturesCoinLoader,
    BinanceFuturesUsdtLoader,
    BinanceSpotLoader,
)
from signalflow.data.source.bybit import (
    BybitClient,
    BybitFuturesInverseLoader,
    BybitFuturesLoader,
    BybitSpotLoader,
)
from signalflow.data.source.deribit import (
    DeribitClient,
    DeribitFuturesLoader,
)
from signalflow.data.source.hyperliquid import (
    HyperliquidClient,
    HyperliquidFuturesLoader,
)
from signalflow.data.source.kraken import (
    KrakenClient,
    KrakenFuturesLoader,
    KrakenSpotLoader,
)
from signalflow.data.source.okx import (
    OkxClient,
    OkxFuturesLoader,
    OkxSpotLoader,
)
from signalflow.data.source.virtual import VirtualDataProvider, generate_crossover_data, generate_ohlcv
from signalflow.data.source.whitebit import (
    WhitebitClient,
    WhitebitFuturesLoader,
    WhitebitSpotLoader,
)

__all__ = [
    # Binance
    "BinanceClient",
    "BinanceFuturesCoinLoader",
    "BinanceFuturesUsdtLoader",
    "BinanceSpotLoader",
    # Bybit
    "BybitClient",
    "BybitFuturesInverseLoader",
    "BybitFuturesLoader",
    "BybitSpotLoader",
    # Deribit
    "DeribitClient",
    "DeribitFuturesLoader",
    # Hyperliquid
    "HyperliquidClient",
    "HyperliquidFuturesLoader",
    # Kraken
    "KrakenClient",
    "KrakenFuturesLoader",
    "KrakenSpotLoader",
    # OKX
    "OkxClient",
    "OkxFuturesLoader",
    "OkxSpotLoader",
    "RawDataLoader",
    "RawDataSource",
    # Virtual
    "VirtualDataProvider",
    # WhiteBIT
    "WhitebitClient",
    "WhitebitFuturesLoader",
    "WhitebitSpotLoader",
    "generate_crossover_data",
    "generate_ohlcv",
]
