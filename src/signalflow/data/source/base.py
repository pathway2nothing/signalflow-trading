"""Source plugin contract."""

from typing import Protocol, runtime_checkable

import polars as pl

CANONICAL_COLUMNS = ["pair", "ts", "open", "high", "low", "close", "volume"]
"""Columns every source must return (spot-first; extra columns are allowed)."""

INTERVAL_SECONDS: dict[str, int] = {
    "1s": 1,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}
"""Bar intervals the built-in sources accept: the Binance kline set minus calendar-month ``1M``.

Every entry is a fixed number of seconds, which is what the fetch pagination, the
disk cache, and the live polling feed rely on.
"""


def interval_seconds(interval: str) -> int:
    """Seconds per bar for a supported ``interval``; ``ValueError`` names the supported set otherwise."""
    try:
        return INTERVAL_SECONDS[interval]
    except KeyError:
        supported = ", ".join(INTERVAL_SECONDS)
        raise ValueError(
            f"unsupported interval {interval!r}; use one of: {supported} (calendar-month '1M' is not supported)"
        ) from None


@runtime_checkable
class Source(Protocol):
    """Fetches market data as canonical OHLCV rows."""

    name: str

    def fetch(
        self,
        pairs: list[str],
        start: str,
        end: str | None = None,
        interval: str = "1h",
    ) -> pl.DataFrame:
        """Return a frame with at least :data:`CANONICAL_COLUMNS`, sorted by (pair, ts)."""
        ...


def validate_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Check canonical columns are present and the frame is sorted by (pair, ts)."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"source frame missing columns: {missing}")
    return df.sort(["pair", "ts"])


def parse_time(value: str | int) -> int:
    """Parse an ISO date/datetime or epoch-seconds into epoch seconds."""
    if isinstance(value, int):
        return value
    from datetime import datetime

    txt = str(value)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(txt, fmt).timestamp())
        except ValueError:
            continue
    raise ValueError(f"cannot parse datetime {value!r}")
