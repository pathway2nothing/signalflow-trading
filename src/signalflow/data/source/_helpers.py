"""Shared utilities for exchange data sources.

Common datetime conversion functions and constants used across
Binance, Bybit, OKX, and other exchange data loaders.
"""

from datetime import datetime, timezone

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
