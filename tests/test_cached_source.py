"""CachedSource disk-cache tests."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import polars as pl

from signalflow.data import CachedSource, data

_EPOCH = datetime(2000, 1, 1)


def _parse(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    raise ValueError(value)


@dataclass
class TsSource:
    """Source whose bar values depend only on absolute ts, so any span is consistent.

    Timestamps are naive wall-clock (matching the cache's boundary parsing), so
    merged cache output is byte-identical to a single wide fetch.
    """

    name: str = "ts_source"
    step_seconds: int = 3600

    def fetch(self, pairs, start, end=None, interval="1h"):
        lo = _parse(start)
        hi = _parse(end) if end else lo + timedelta(seconds=5000 * self.step_seconds)
        n = max(0, int((hi - lo).total_seconds() // self.step_seconds))
        stamps = [lo + timedelta(seconds=self.step_seconds * i) for i in range(n)]
        close = [float((t - _EPOCH).total_seconds()) for t in stamps]
        frames = []
        for pair in pairs:
            frames.append(
                pl.DataFrame(
                    {
                        "pair": [pair] * n,
                        "ts": stamps,
                        "open": close,
                        "high": close,
                        "low": close,
                        "close": close,
                        "volume": [1.0] * n,
                    }
                ).with_columns(pl.col("ts").cast(pl.Datetime("ms")))
            )
        return pl.concat(frames).sort(["pair", "ts"])


@dataclass
class CountingSource:
    """Wrap a source, recording every fetch span."""

    inner: object
    name: str = "counting"
    calls: list = field(default_factory=list)

    def fetch(self, pairs, start, end=None, interval="1h"):
        self.calls.append((start, end))
        return self.inner.fetch(pairs, start, end, interval)


def test_first_call_populates_parquet(tmp_path):
    src = CachedSource(TsSource(), tmp_path)
    src.fetch(["BTCUSDT"], "2023-01-01", "2023-01-10", "1h")
    assert (tmp_path / "1h" / "BTCUSDT.parquet").exists()


def test_wider_span_fetches_only_missing_tail(tmp_path):
    counting = CountingSource(TsSource())
    cached = CachedSource(counting, tmp_path)

    cached.fetch(["BTCUSDT"], "2023-01-01", "2023-01-10", "1h")
    cached.fetch(["BTCUSDT"], "2023-01-01", "2023-01-20", "1h")

    assert len(counting.calls) == 2
    tail_start = _parse(counting.calls[1][0])
    assert tail_start > _parse("2023-01-01")


def test_cached_data_identical_to_uncached(tmp_path):
    cached = CachedSource(TsSource(), tmp_path)
    cached.fetch(["BTCUSDT"], "2023-01-01", "2023-01-10", "1h")
    got = cached.fetch(["BTCUSDT"], "2023-01-01", "2023-01-20", "1h").sort(["pair", "ts"])

    want = TsSource().fetch(["BTCUSDT"], "2023-01-01", "2023-01-20", "1h").sort(["pair", "ts"])
    assert got.select(["pair", "ts", "close"]).equals(want.select(["pair", "ts", "close"]))


def test_corrupt_cache_self_heals(tmp_path):
    cached = CachedSource(TsSource(), tmp_path)
    cached.fetch(["BTCUSDT"], "2023-01-01", "2023-01-10", "1h")

    path = tmp_path / "1h" / "BTCUSDT.parquet"
    path.write_bytes(b"not a parquet file")

    got = cached.fetch(["BTCUSDT"], "2023-01-01", "2023-01-10", "1h").sort(["pair", "ts"])
    want = TsSource().fetch(["BTCUSDT"], "2023-01-01", "2023-01-10", "1h").sort(["pair", "ts"])
    assert got.select(["pair", "ts", "close"]).equals(want.select(["pair", "ts", "close"]))
    assert pl.read_parquet(path).height > 0


def test_cache_dir_param_on_sf_data(tmp_path):
    src = TsSource()
    uncached = data(src, pairs=["BTCUSDT"], start="2023-01-01", end="2023-01-05", interval="1h")
    cached = data(src, pairs=["BTCUSDT"], start="2023-01-01", end="2023-01-05", interval="1h", cache_dir=tmp_path)
    assert (tmp_path / "1h" / "BTCUSDT.parquet").exists()
    assert cached.frame.sort(["pair", "ts"]).equals(uncached.frame.sort(["pair", "ts"]))
