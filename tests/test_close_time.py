"""``ts`` is the bar's close time - in sources, the closed-bar filter, and the disk cache."""

from datetime import UTC, datetime, timedelta

import polars as pl

import signalflow as sf
from signalflow.data.source.cached import _MARKER, CachedSource
from signalflow.flow.live import _close_epoch, _closed_only


def test_synthetic_stamps_bars_at_their_close():
    ds = sf.dataset("synthetic", pairs=["BTCUSDT"], start="2024-01-01", end="2024-01-02", interval="1h")
    ts = ds.frame.get_column("ts")
    assert ts.min() == datetime(2024, 1, 1, 1)  # the first bar opened at start and closed an hour later
    assert ts.max() == datetime(2024, 1, 2)  # the last bar closes exactly at end
    assert ds.height == 24


def test_closed_only_keeps_a_bar_that_closed_right_now():
    frame = pl.DataFrame({"pair": ["X", "X"], "ts": [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2)]}).with_columns(
        pl.col("ts").cast(pl.Datetime("ms"))
    )
    now = datetime(2024, 1, 1, 1, tzinfo=UTC).timestamp()  # the first bar's close
    kept = _closed_only(frame, 3600, now)
    assert kept.get_column("ts").to_list() == [datetime(2024, 1, 1, 1)]
    assert _close_epoch(datetime(2024, 1, 1, 1), 3600) == now


def test_cache_migrates_open_time_files_once(tmp_path):
    root = tmp_path / "cache"
    (root / "1h").mkdir(parents=True)
    old = pl.DataFrame(
        {
            "pair": ["BTCUSDT"] * 3,
            "ts": [datetime(2024, 1, 1, h) for h in range(3)],
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [1.0, 1.0, 1.0],
        }
    ).with_columns(pl.col("ts").cast(pl.Datetime("ms")))
    old.write_parquet(root / "1h" / "BTCUSDT.parquet")

    CachedSource(sf.SyntheticSource(), root)
    migrated = pl.read_parquet(root / "1h" / "BTCUSDT.parquet").get_column("ts").to_list()
    assert migrated == [datetime(2024, 1, 1, h) + timedelta(hours=1) for h in range(3)]
    assert (root / _MARKER).read_text().strip() == "close"

    CachedSource(sf.SyntheticSource(), root)  # the marker prevents a second shift
    assert pl.read_parquet(root / "1h" / "BTCUSDT.parquet").get_column("ts").to_list() == migrated


def test_cache_round_trip_keeps_close_time(tmp_path):
    src = sf.SyntheticSource()
    direct = src.fetch(["BTCUSDT"], "2024-01-01", "2024-01-03", "1h")
    cached = CachedSource(src, tmp_path / "c", partition="day").fetch(["BTCUSDT"], "2024-01-01", "2024-01-03", "1h")
    assert cached.get_column("ts").to_list() == direct.get_column("ts").to_list()
    assert cached.get_column("ts").max() == datetime(2024, 1, 3)
