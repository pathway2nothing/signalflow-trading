"""Bar-interval support shared by the built-in sources, the disk cache, and the live feed."""

import importlib
from datetime import timedelta

import pytest

import signalflow as sf
from signalflow.data.source.base import INTERVAL_SECONDS, interval_seconds
from signalflow.data.source.binance import BinanceSource

BINANCE_KLINE_INTERVALS = ["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"]


def test_interval_table_matches_binance_fixed_width_set():
    assert list(INTERVAL_SECONDS) == BINANCE_KLINE_INTERVALS
    assert "1M" not in INTERVAL_SECONDS


@pytest.mark.parametrize("bad", ["1M", "2m", "10m", "", "1H"])
def test_unsupported_interval_names_the_supported_set(bad):
    with pytest.raises(ValueError, match="unsupported interval"):
        interval_seconds(bad)


@pytest.mark.parametrize("interval", BINANCE_KLINE_INTERVALS)
def test_synthetic_source_spacing_matches_interval(interval):
    step = INTERVAL_SECONDS[interval]
    span = max(step * 4, 3600)  # at least four bars
    src = sf.SyntheticSource()
    frame = src.fetch(["BTCUSDT"], start=0, end=span, interval=interval)
    assert frame.height == span // step
    diffs = frame.get_column("ts").diff().drop_nulls().dt.total_seconds().unique().to_list()
    assert diffs == [step]


class _FakeBinance(BinanceSource):
    """Serve klines from a local clock instead of the network."""

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit
        self.calls: list[tuple[str, int, int]] = []

    def _request(self, pair, interval, start_ms, end_ms):
        step = interval_seconds(interval) * 1000
        self.calls.append((interval, start_ms, end_ms))
        rows = []
        t = start_ms
        while t < end_ms and len(rows) < self.limit:
            rows.append([t, 1.0, 1.0, 1.0, 1.0, 1.0])
            t += step
        return rows


@pytest.mark.parametrize("interval", ["3m", "30m", "2h", "6h", "8h", "12h", "1w"])
def test_binance_pagination_uses_new_interval_widths(interval, monkeypatch):
    # `sf.data` (the function) shadows the `signalflow.data` package attribute, so neither
    # `import signalflow.data.source.binance as m` nor a dotted monkeypatch path resolves.
    binance_module = importlib.import_module("signalflow.data.source.binance")
    monkeypatch.setattr(binance_module, "_LIMIT", 4)
    src = _FakeBinance(limit=4)
    step = INTERVAL_SECONDS[interval]
    n_bars = 10
    frame = src.fetch(["BTCUSDT"], start=0, end=n_bars * step, interval=interval)
    assert frame.height == n_bars
    diffs = frame.get_column("ts").diff().drop_nulls().dt.total_seconds().unique().to_list()
    assert diffs == [step]
    # every page starts one bar after the previous page's last open time
    starts = [c[1] for c in src.calls]
    assert starts == [0, 4 * step * 1000, 8 * step * 1000]


def test_binance_rejects_calendar_month():
    with pytest.raises(ValueError, match="unsupported interval '1M'"):
        BinanceSource().fetch(["BTCUSDT"], start="2024-01-01", end="2024-03-01", interval="1M")


def test_cached_source_uses_shared_interval_widths():
    from signalflow.data.source.cached import _interval_step

    assert _interval_step("30m") == timedelta(minutes=30)
    assert _interval_step("1w") == timedelta(weeks=1)
    assert _interval_step("weird") == timedelta(0)


def test_polling_feed_accepts_extended_intervals():
    feed = sf.PollingFeed(source=sf.SyntheticSource(), pairs=["BTCUSDT"], interval="30m", warmup_bars=3)
    ds = feed.warmup()
    assert ds.frame.height >= 3
    diffs = ds.frame.get_column("ts").diff().drop_nulls().dt.total_seconds().unique().to_list()
    assert diffs == [1800]
