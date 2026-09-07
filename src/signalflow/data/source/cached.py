"""Disk-backed OHLCV cache that wraps any Source and fetches only missing spans."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from loguru import logger

from signalflow._time import INTERVAL_SECONDS, parse_datetime
from signalflow.data.source.base import Source, validate_frame

_OVERLAP = timedelta(days=1)


_MARKER = ".ts_convention"


def _ensure_close_time(root: Path) -> None:
    """Migrate a cache written when ``ts`` was the open time (one shift per file, then a marker).

    Bars keep their day partition: a bar that opened in a day closes in it too
    (the last one exactly at midnight, which the ``> day .. <= day_end`` bounds keep).
    """
    marker = root / _MARKER
    if marker.exists():
        return
    files = sorted(root.glob("*/**/*.parquet")) if root.exists() else []
    if files:
        logger.warning(f"CachedSource: migrating {len(files)} file(s) under {root} from open-time to close-time ts")
        for path in files:
            step = _interval_step(path.relative_to(root).parts[0])
            shifted = pl.read_parquet(path).with_columns((pl.col("ts") + step).alias("ts"))
            tmp = path.with_suffix(".parquet.tmp")
            shifted.write_parquet(tmp)
            tmp.replace(path)
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text("close\n", encoding="utf-8")


def _interval_step(interval: str) -> timedelta:
    """Bar width for cache-completeness checks; zero (never complete) for an unknown interval."""
    return timedelta(seconds=INTERVAL_SECONDS.get(interval, 0))


def _fmt(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


class CachedSource(Source):
    """Wrap ``inner``, serving repeated requests from parquet cache under ``root``.

    Layouts:

    * flat (default) - one growing ``<root>/<interval>/<PAIR>.parquet``, written
      only after the whole missing span is fetched;
    * ``partition="day"`` - one ``<root>/<interval>/<PAIR>/<YYYY-MM-DD>.parquet``
      per day, written as soon as that day is fetched, so an interrupted fetch
      keeps every completed day. An incomplete trailing day (last bar short of
      midnight) is re-fetched and merged on the next request.
    """

    def __init__(self, inner: Source, root: "str | Path", partition: "str | None" = None) -> None:
        if partition not in (None, "day"):
            raise ValueError(f"unsupported cache partition {partition!r}; use None or 'day'")
        self.inner = inner
        self.root = Path(root)
        self.partition = partition
        self.name = getattr(inner, "name", "cached")
        _ensure_close_time(self.root)

    def fetch(
        self,
        pairs: list[str],
        start: str,
        end: "str | None" = None,
        interval: str = "1h",
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
        want_start = parse_datetime(start)
        spans: list[tuple[str, str | None]] = []
        if want_start < have_min:
            spans.append((start, _fmt(have_min + _OVERLAP)))
        if end is None:
            spans.append((_fmt(have_max - _OVERLAP), None))
        elif parse_datetime(end) > have_max + _interval_step(interval):
            spans.append((_fmt(have_max - _OVERLAP), end))
        return spans

    def _fetch_pair(self, pair: str, start: str, end: "str | None", interval: str) -> pl.DataFrame:
        if self.partition == "day":
            return self._fetch_pair_daily(pair, start, end, interval)
        path = self._path(pair, interval)
        cached = self._read_cache(path)
        spans = self._missing_spans(start, end, cached, interval)
        if spans:
            logger.info(f"CachedSource: {pair} {interval}: fetching missing span(s) {spans}")
        else:
            logger.debug(f"CachedSource: {pair} {interval}: served from cache ({path})")

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

        lo = parse_datetime(start)
        frame = merged.filter(pl.col("ts") >= lo)
        if end is not None:
            frame = frame.filter(pl.col("ts") <= parse_datetime(end))
        return frame

    def _fetch_pair_daily(self, pair: str, start: str, end: "str | None", interval: str) -> pl.DataFrame:
        step = _interval_step(interval)
        start_dt = parse_datetime(start)
        end_dt = parse_datetime(end) if end is not None else datetime.now(UTC).replace(tzinfo=None)
        day_dir = self.root / interval / pair
        parts: list[pl.DataFrame] = []
        day = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        while day <= end_dt:
            day_end = day + timedelta(days=1)
            path = day_dir / f"{day:%Y-%m-%d}.parquet"
            cached = self._read_cache(path)
            complete = (
                cached is not None
                and cached.height > 0
                and step > timedelta(0)
                and cached.get_column("ts").max() >= day_end  # the day's last bar closes at day_end
            )
            if not complete:
                fetched = self.inner.fetch([pair], _fmt(day), _fmt(day_end), interval)
                pieces = [p for p in (cached, fetched) if p is not None and p.height > 0]
                merged = (
                    pl.concat(pieces)
                    .unique(subset=["pair", "ts"], keep="last")
                    .sort(["pair", "ts"])
                    .filter((pl.col("ts") > day) & (pl.col("ts") <= day_end))
                    if pieces
                    else None
                )
                if merged is not None and merged.height > 0:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".parquet.tmp")
                    merged.write_parquet(tmp)
                    tmp.replace(path)
                    logger.debug(f"CachedSource: {pair} {interval}: wrote {path.name} ({merged.height} rows)")
                cached = merged
            else:
                logger.debug(f"CachedSource: {pair} {interval}: {path.name} served from cache")
            if cached is not None and cached.height > 0:
                parts.append(cached)
            day = day_end
        if not parts:
            return self.inner.fetch([pair], start, end, interval)
        frame = pl.concat(parts).filter(pl.col("ts") >= start_dt)
        if end is not None:
            frame = frame.filter(pl.col("ts") <= end_dt)
        return frame
