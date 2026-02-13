"""Shared utilities for exchange data sources.

Common datetime conversion functions and constants used across
Binance, Bybit, OKX, Deribit, Kraken, Hyperliquid, WhiteBIT, and other exchange data loaders.
"""

from datetime import datetime, timezone
from typing import Final

# Standard timeframe to milliseconds mapping.
TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def dt_to_ms_utc(dt: datetime) -> int:
    """Convert datetime to UNIX milliseconds in UTC.

    Accepts naive (assumed UTC) or aware (converted to UTC) datetimes.

    Args:
        dt: Input datetime.

    Returns:
        UNIX timestamp in milliseconds (UTC).

    Example:
        >>> from datetime import datetime
        >>> dt_to_ms_utc(datetime(2024, 1, 1))
        1704067200000
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


def ms_to_dt_utc_naive(ms: int) -> datetime:
    """Convert UNIX milliseconds to UTC-naive datetime.

    Args:
        ms: UNIX timestamp in milliseconds.

    Returns:
        UTC datetime without timezone info.

    Example:
        >>> ms_to_dt_utc_naive(1704067200000)
        datetime.datetime(2024, 1, 1, 0, 0)
    """
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None)


def ensure_utc_naive(dt: datetime) -> datetime:
    """Normalize to UTC-naive datetime.

    Args:
        dt: Input datetime (naive or aware).

    Returns:
        UTC-naive datetime.

    Example:
        >>> from datetime import timezone
        >>> dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        >>> ensure_utc_naive(dt)
        datetime.datetime(2024, 1, 1, 0, 0)
    """
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Seconds-based timestamp helpers (for Kraken)
# ---------------------------------------------------------------------------


def sec_to_dt_utc_naive(sec: int) -> datetime:
    """Convert UNIX seconds to UTC-naive datetime.

    Args:
        sec: UNIX timestamp in seconds.

    Returns:
        UTC datetime without timezone info.

    Example:
        >>> sec_to_dt_utc_naive(1704067200)
        datetime.datetime(2024, 1, 1, 0, 0)
    """
    return datetime.fromtimestamp(sec, tz=timezone.utc).replace(tzinfo=None)


def dt_to_sec_utc(dt: datetime) -> int:
    """Convert datetime to UNIX seconds in UTC.

    Args:
        dt: Input datetime (naive=UTC or aware).

    Returns:
        UNIX timestamp in seconds.

    Example:
        >>> dt_to_sec_utc(datetime(2024, 1, 1))
        1704067200
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp())


# ---------------------------------------------------------------------------
# Kraken pair normalization
# ---------------------------------------------------------------------------

# Kraken uses non-standard symbol names with X/Z prefixes
KRAKEN_SPOT_MAP: Final[dict[str, str]] = {
    "XXBTZUSD": "BTCUSD",
    "XETHZUSD": "ETHUSD",
    "XXBTZEUR": "BTCEUR",
    "XETHZEUR": "ETHEUR",
    "XXBTZGBP": "BTCGBP",
    "XETHZGBP": "ETHGBP",
    "XXRPZUSD": "XRPUSD",
    "XLTCZUSD": "LTCUSD",
    "XETHXXBT": "ETHBTC",
    "XXLMZUSD": "XLMUSD",
    "XZECZUSD": "ZECUSD",
    "XXMRZUSD": "XMRUSD",
    "XETCZUSD": "ETCUSD",
    "XXDGZUSD": "DOGEUSD",
    "ADAUSD": "ADAUSD",
    "SOLUSD": "SOLUSD",
    "DOTUSD": "DOTUSD",
    "LINKUSD": "LINKUSD",
    "MATICUSD": "MATICUSD",
    "UNIUSD": "UNIUSD",
    "AVAXUSD": "AVAXUSD",
    "ATOMUSD": "ATOMUSD",
}

KRAKEN_SPOT_REVERSE_MAP: Final[dict[str, str]] = {v: k for k, v in KRAKEN_SPOT_MAP.items()}


def normalize_kraken_spot_pair(symbol: str) -> str:
    """Convert Kraken spot symbol to internal compact format.

    Args:
        symbol: Kraken symbol (e.g., "XXBTZUSD", "XETHZUSD").

    Returns:
        Internal format (e.g., "BTCUSD", "ETHUSD").

    Example:
        >>> normalize_kraken_spot_pair("XXBTZUSD")
        'BTCUSD'
        >>> normalize_kraken_spot_pair("XETHZEUR")
        'ETHEUR'
    """
    symbol = symbol.upper()
    if symbol in KRAKEN_SPOT_MAP:
        return KRAKEN_SPOT_MAP[symbol]
    # Fallback: remove X/Z prefixes if present
    if symbol.startswith("X") and len(symbol) > 4:
        symbol = symbol[1:]
    if "Z" in symbol and len(symbol) > 5:
        symbol = symbol.replace("Z", "", 1)
    return symbol


def to_kraken_spot_symbol(pair: str) -> str:
    """Convert internal format to Kraken spot symbol.

    Args:
        pair: Internal format (e.g., "BTCUSD").

    Returns:
        Kraken symbol (e.g., "XXBTZUSD").

    Example:
        >>> to_kraken_spot_symbol("BTCUSD")
        'XXBTZUSD'
    """
    pair = pair.upper()
    return KRAKEN_SPOT_REVERSE_MAP.get(pair, pair)


def normalize_kraken_futures_pair(symbol: str) -> str:
    """Convert Kraken futures symbol to internal compact format.

    Args:
        symbol: Kraken futures symbol (e.g., "pi_xbtusd", "PI_ETHUSD").

    Returns:
        Internal format (e.g., "BTCUSD", "ETHUSD").

    Example:
        >>> normalize_kraken_futures_pair("pi_xbtusd")
        'BTCUSD'
        >>> normalize_kraken_futures_pair("PI_ETHUSD")
        'ETHUSD'
    """
    symbol = symbol.upper()
    # Remove prefixes: PI_, PF_, FI_
    for prefix in ("PI_", "PF_", "FI_"):
        if symbol.startswith(prefix):
            symbol = symbol[len(prefix):]
            break
    # XBT -> BTC
    symbol = symbol.replace("XBT", "BTC")
    return symbol


def to_kraken_futures_symbol(pair: str, prefix: str = "PI_") -> str:
    """Convert internal format to Kraken futures symbol.

    Args:
        pair: Internal format (e.g., "BTCUSD").
        prefix: Kraken prefix (default "PI_" for perpetuals).

    Returns:
        Kraken futures symbol (e.g., "pi_xbtusd").

    Example:
        >>> to_kraken_futures_symbol("BTCUSD")
        'pi_xbtusd'
    """
    pair = pair.upper().replace("BTC", "XBT")
    return f"{prefix.lower()}{pair.lower()}"


# ---------------------------------------------------------------------------
# Deribit pair normalization
# ---------------------------------------------------------------------------


def normalize_deribit_pair(instrument: str) -> str:
    """Convert Deribit instrument to internal compact format.

    Args:
        instrument: Deribit instrument (e.g., "BTC-PERPETUAL", "BTC-USDC-PERPETUAL").

    Returns:
        Internal format (e.g., "BTCUSD", "BTCUSDC").

    Example:
        >>> normalize_deribit_pair("BTC-PERPETUAL")
        'BTCUSD'
        >>> normalize_deribit_pair("ETH-USDC-PERPETUAL")
        'ETHUSDC'
        >>> normalize_deribit_pair("BTC-27DEC24")
        'BTCUSD'
    """
    parts = instrument.upper().split("-")
    base = parts[0]

    if "PERPETUAL" in parts:
        if "USDC" in parts:
            return f"{base}USDC"
        return f"{base}USD"

    # Dated futures: BTC-27DEC24 -> BTCUSD
    if len(parts) == 2 and parts[1][0].isdigit():
        return f"{base}USD"

    return instrument


def to_deribit_instrument(pair: str, suffix: str = "-PERPETUAL") -> str:
    """Convert internal format to Deribit instrument.

    Args:
        pair: Internal format (e.g., "BTCUSD", "ETHUSDC").
        suffix: Instrument suffix (default "-PERPETUAL").

    Returns:
        Deribit instrument (e.g., "BTC-PERPETUAL", "ETH-USDC-PERPETUAL").

    Example:
        >>> to_deribit_instrument("BTCUSD")
        'BTC-PERPETUAL'
        >>> to_deribit_instrument("ETHUSDC")
        'ETH-USDC-PERPETUAL'
    """
    pair = pair.upper()
    for quote in ("USDC", "USD"):
        if pair.endswith(quote):
            base = pair[: -len(quote)]
            if quote == "USDC":
                return f"{base}-USDC{suffix}"
            return f"{base}{suffix}"
    return pair


# ---------------------------------------------------------------------------
# Hyperliquid pair normalization
# ---------------------------------------------------------------------------


def normalize_hyperliquid_pair(coin: str) -> str:
    """Convert Hyperliquid coin to internal compact format.

    Args:
        coin: Hyperliquid coin symbol (e.g., "BTC", "ETH").

    Returns:
        Internal format (e.g., "BTCUSD", "ETHUSD").

    Example:
        >>> normalize_hyperliquid_pair("BTC")
        'BTCUSD'
        >>> normalize_hyperliquid_pair("ETH")
        'ETHUSD'
    """
    return f"{coin.upper()}USD"


def to_hyperliquid_coin(pair: str) -> str:
    """Convert internal format to Hyperliquid coin.

    Args:
        pair: Internal format (e.g., "BTCUSD", "BTCUSDT").

    Returns:
        Hyperliquid coin (e.g., "BTC").

    Example:
        >>> to_hyperliquid_coin("BTCUSD")
        'BTC'
        >>> to_hyperliquid_coin("ETHUSDT")
        'ETH'
    """
    pair = pair.upper()
    for suffix in ("USDT", "USDC", "USD"):
        if pair.endswith(suffix):
            return pair[: -len(suffix)]
    return pair


# ---------------------------------------------------------------------------
# WhiteBIT pair normalization
# ---------------------------------------------------------------------------


def normalize_whitebit_pair(symbol: str) -> str:
    """Convert WhiteBIT symbol to internal compact format.

    WhiteBIT uses underscore-separated pairs: BTC_USDT, ETH_USDT.

    Args:
        symbol: WhiteBIT symbol (e.g., "BTC_USDT", "ETH_USDT").

    Returns:
        Internal format (e.g., "BTCUSDT", "ETHUSDT").

    Example:
        >>> normalize_whitebit_pair("BTC_USDT")
        'BTCUSDT'
        >>> normalize_whitebit_pair("eth_usdt")
        'ETHUSDT'
    """
    return symbol.upper().replace("_", "")


def to_whitebit_symbol(pair: str) -> str:
    """Convert internal format to WhiteBIT symbol.

    Args:
        pair: Internal format (e.g., "BTCUSDT", "ETHUSDT").

    Returns:
        WhiteBIT symbol (e.g., "BTC_USDT", "ETH_USDT").

    Example:
        >>> to_whitebit_symbol("BTCUSDT")
        'BTC_USDT'
        >>> to_whitebit_symbol("ETHUSDC")
        'ETH_USDC'
    """
    pair = pair.upper()
    for quote in ("USDT", "USDC", "USD", "UAH", "EUR", "BTC", "ETH"):
        if pair.endswith(quote):
            base = pair[: -len(quote)]
            return f"{base}_{quote}"
    return pair
