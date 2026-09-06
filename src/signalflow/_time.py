"""Time helpers shared by sources, the model, targets and runs - one parser per concept.

* :data:`INTERVAL_SECONDS` / :func:`interval_seconds` - the bar intervals the
  built-in sources, the disk cache and the polling feed accept.
* :func:`parse_duration` - ``"30m"``/``"12h"``/``"365d"`` -> ``timedelta``;
  :func:`advance` / :func:`retreat` also understand calendar months (``"3mo"``).
* :func:`parse_datetime` / :func:`to_epoch` - ISO dates and epoch seconds, always UTC.
* :func:`bar_seconds` - the median spacing of a timestamp series (the bar width).
"""

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

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

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}

_DATETIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")


def interval_seconds(interval: str) -> int:
    """Seconds per bar for a supported ``interval``; ``ValueError`` names the supported set otherwise."""
    try:
        return INTERVAL_SECONDS[interval]
    except KeyError:
        supported = ", ".join(INTERVAL_SECONDS)
        raise ValueError(
            f"unsupported interval {interval!r}; use one of: {supported} (calendar-month '1M' is not supported)"
        ) from None


def parse_duration(span: str) -> timedelta:
    """Parse a fixed-width duration like ``1d``, ``365d``, ``12h``, ``30m`` into a ``timedelta``.

    Calendar months (``"1mo"``) are not fixed-width; use :func:`advance` / :func:`retreat`.
    """
    text = str(span).strip().lower()
    if len(text) < 2 or text[-1] not in _UNIT_SECONDS:
        raise ValueError(f"unsupported duration {span!r}; use <number><s|m|h|d|w>, e.g. '30m', '12h', '365d'")
    try:
        amount = float(text[:-1])
    except ValueError:
        raise ValueError(f"unsupported duration {span!r}; use <number><s|m|h|d|w>") from None
    return timedelta(seconds=amount * _UNIT_SECONDS[text[-1]])


def _days_in_month(year: int, month: int) -> int:
    nxt = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    return (nxt - datetime(year, month, 1)).days


def _add_months(moment: datetime, months: int) -> datetime:
    """Calendar-month arithmetic clamping the day into the target month."""
    total = moment.month - 1 + months
    year = moment.year + total // 12
    month = total % 12 + 1
    day = min(moment.day, _days_in_month(year, month))
    return moment.replace(year=year, month=month, day=day)


def advance(moment: datetime, span: str) -> datetime:
    """Step ``moment`` forward by a duration string; ``"1mo"``/``"3mo"`` are calendar months."""
    text = str(span).strip().lower()
    if text.endswith("mo"):
        return _add_months(moment, int(text[:-2]))
    return moment + parse_duration(text)


def retreat(moment: datetime, span: str) -> datetime:
    """Step ``moment`` backward by a duration string (calendar months for ``"Nmo"``)."""
    text = str(span).strip().lower()
    if text.endswith("mo"):
        return _add_months(moment, -int(text[:-2]))
    return moment - parse_duration(text)


def parse_datetime(value: str | int | float | datetime) -> datetime:
    """Parse an ISO date/datetime string or epoch seconds into a naive UTC ``datetime``."""
    if isinstance(value, datetime):
        return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo is not None else value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(float(value), UTC).replace(tzinfo=None)
    text = str(value).strip().replace("T", " ")
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt.replace("T", " "))
        except ValueError:
            continue
    raise ValueError(f"cannot parse datetime {value!r}; use YYYY-MM-DD or YYYY-MM-DD HH:MM[:SS]")


def to_epoch(value: str | int | float | datetime) -> int:
    """Epoch seconds (UTC) for an ISO date/datetime string, epoch seconds, or ``datetime``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return int(parse_datetime(value).replace(tzinfo=UTC).timestamp())


def bar_seconds(ts: pl.Series | Iterable) -> float:
    """Median spacing in seconds between consecutive unique timestamps; ``0.0`` when unknown."""
    series = ts if isinstance(ts, pl.Series) else pl.Series(list(ts))
    if series.len() < 2:
        return 0.0
    uniq = series.unique().sort()
    if uniq.len() < 2:
        return 0.0
    if uniq.dtype.is_temporal():
        secs = uniq.diff().drop_nulls().dt.total_seconds()
    else:
        secs = uniq.diff().drop_nulls().cast(pl.Float64)
    positive = secs.filter(secs > 0)
    return float(np.median(positive.to_numpy())) if positive.len() else 0.0


__all__ = [
    "INTERVAL_SECONDS",
    "advance",
    "bar_seconds",
    "interval_seconds",
    "parse_datetime",
    "parse_duration",
    "retreat",
    "to_epoch",
]
