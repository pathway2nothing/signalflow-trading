"""Exchange data sources and loaders.

Provides async clients and loaders for downloading historical OHLCV data
from cryptocurrency exchanges.

Built-in Exchanges:
    - Binance: Spot, USDT-M Futures, COIN-M Futures (reference implementation)
    - Virtual: Synthetic data for testing

Extended Exchanges (via signalflow-data package):
    - Bybit: Spot, Linear Futures, Inverse Futures
    - OKX: Spot, Perpetual Swaps
    - Deribit: Futures/Perpetuals
    - Kraken: Spot, Futures
    - Hyperliquid: Perpetuals (DEX)
    - WhiteBIT: Spot, Futures

Install extended exchanges:
    pip install git+https://github.com/yourorg/sf-data.git

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
from signalflow.data.source.virtual import VirtualDataProvider, generate_crossover_data, generate_ohlcv

__all__ = [
    # Base classes
    "RawDataLoader",
    "RawDataSource",
    # Binance (built-in reference implementation)
    "BinanceClient",
    "BinanceFuturesCoinLoader",
    "BinanceFuturesUsdtLoader",
    "BinanceSpotLoader",
    # Virtual (testing)
    "VirtualDataProvider",
    "generate_crossover_data",
    "generate_ohlcv",
    # Extended exchanges (lazy imports via __getattr__)
    "BinanceStocksClient",
    "BinanceStocksLoader",
    "BybitClient",
    "BybitFuturesInverseLoader",
    "BybitFuturesLoader",
    "BybitSpotLoader",
    "DeribitClient",
    "DeribitFuturesLoader",
    "HyperliquidClient",
    "HyperliquidFuturesLoader",
    "KrakenClient",
    "KrakenFuturesLoader",
    "KrakenSpotLoader",
    "OkxClient",
    "OkxFuturesLoader",
    "OkxSpotLoader",
    "WhitebitClient",
    "WhitebitFuturesLoader",
    "WhitebitSpotLoader",
]


def __getattr__(name: str):
    """Lazy import extended exchanges from signalflow-data if available."""
    _EXTENDED_EXCHANGES = {
        # Binance Stocks
        "BinanceStocksClient": "signalflow.data.source.binance_stocks",
        "BinanceStocksLoader": "signalflow.data.source.binance_stocks",
        # OKX
        "OkxClient": "signalflow.data.source.okx",
        "OkxSpotLoader": "signalflow.data.source.okx",
        "OkxFuturesLoader": "signalflow.data.source.okx",
        # Bybit
        "BybitClient": "signalflow.data.source.bybit",
        "BybitSpotLoader": "signalflow.data.source.bybit",
        "BybitFuturesLoader": "signalflow.data.source.bybit",
        "BybitFuturesInverseLoader": "signalflow.data.source.bybit",
        # Kraken
        "KrakenClient": "signalflow.data.source.kraken",
        "KrakenSpotLoader": "signalflow.data.source.kraken",
        "KrakenFuturesLoader": "signalflow.data.source.kraken",
        # Deribit
        "DeribitClient": "signalflow.data.source.deribit",
        "DeribitFuturesLoader": "signalflow.data.source.deribit",
        # Hyperliquid
        "HyperliquidClient": "signalflow.data.source.hyperliquid",
        "HyperliquidFuturesLoader": "signalflow.data.source.hyperliquid",
        # WhiteBIT
        "WhitebitClient": "signalflow.data.source.whitebit",
        "WhitebitSpotLoader": "signalflow.data.source.whitebit",
        "WhitebitFuturesLoader": "signalflow.data.source.whitebit",
    }

    if name in _EXTENDED_EXCHANGES:
        try:
            module_path = _EXTENDED_EXCHANGES[name]
            module = __import__(module_path, fromlist=[name])
            return getattr(module, name)
        except ImportError:
            raise ImportError(
                f"{name} requires signalflow-data package. Install with:\n"
                f"  pip install git+https://github.com/yourorg/sf-data.git"
            )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
