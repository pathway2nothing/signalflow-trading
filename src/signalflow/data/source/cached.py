"""Disk-backed OHLCV cache that wraps any Source and fetches only missing spans."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from loguru import logger

from signalflow.data.source.base import Source, validate_frame

_OVERLAP = timedelta(days=1)

_STEP_S = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def _interval_step(interval: str) -> timedelta:
    return timedelta(seconds=_STEP_S.get(interval, 0))


def _parse_dt(value: str) -> datetime:
    """Parse an ISO date or datetime string into a naive datetime."""
    text = str(value).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognised date {value!r}; use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")


def _fmt(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


class CachedSource(Source):
    """Wrap ``inner``, serving repeated requests from one growing parquet per (pair, interval)."""

    def __init__(self, inner: Source, root: "str | Path") -> None:
        self.inner = inner
        self.root = Path(root)
        self.name = getattr(inner, "name", "cached")

    def fetch(
        self,
        pairs: list[str],
        start: str,
        end: "str | None" = None,
        interval: str = "1m",
    ) -> pl.DataFrame:
        frames = [self._fetch_pair(pair, start, end, interval) for pair in pairs]
        frames = [f for f in frames if f.height > 0]
        if not frames:
            return validate_frame(self.inner.fetch(pairs, start, end, interval))
        return validate_frame(pl.concat(frames))

    def _path(self, pair: str, interval: str) -> Path:
        return self.root / interval / f"{pair}.parquet"

    def _read_cache(self, path: Path) -> "pl.DataFrame | None":
        if not path.exists():
            return None
        try:
            return pl.read_parquet(path)
        except Exception as exc:
            logger.warning(f"CachedSource: discarding unreadable cache {path}: {exc}")
            return None

    def _missing_spans(
        self, start: str, end: "str | None", cached: "pl.DataFrame | None", interval: str
    ) -> "list[tuple[str, str | None]]":
        if cached is None or cached.height == 0:
            return [(start, end)]
        have_min = cached.get_column("ts").min()
        have_max = cached.get_column("ts").max()
        want_start = _parse_dt(start)
        spans: list[tuple[str, str | None]] = []
        if want_start < have_min:
            spans.append((start, _fmt(have_min + _OVERLAP)))
        if end is None:
            spans.append((_fmt(have_max - _OVERLAP), None))
        elif _parse_dt(end) > have_max + _interval_step(interval):
            spans.append((_fmt(have_max - _OVERLAP), end))
        return spans

    def _fetch_pair(self, pair: str, start: str, end: "str | None", interval: str) -> pl.DataFrame:
        path = self._path(pair, interval)
        cached = self._read_cache(path)
        spans = self._missing_spans(start, end, cached, interval)

        parts = [cached] if cached is not None else []
        for span_start, span_end in spans:
            parts.append(self.inner.fetch([pair], span_start, span_end, interval))
        parts = [p for p in parts if p is not None and p.height > 0]
        if not parts:
            return cached if cached is not None else self.inner.fetch([pair], start, end, interval)

        merged = pl.concat(parts).unique(subset=["pair", "ts"], keep="last").sort(["pair", "ts"])
        if spans and merged.height > 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".parquet.tmp")
            merged.write_parquet(tmp)
            tmp.replace(path)

        lo = _parse_dt(start)
        frame = merged.filter(pl.col("ts") >= lo)
        if end is not None:
            frame = frame.filter(pl.col("ts") <= _parse_dt(end))
        return frame
