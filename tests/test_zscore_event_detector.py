"""Tests for ZScoreEventDetector."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.core.enums import SfComponentType
from signalflow.detector.event.base import EventDetectorBase
from signalflow.detector.event.zscore_detector import ZScoreEventDetector
from signalflow.target.multi_target_generator import HorizonConfig


def _make_normal_market(n: int = 300, n_pairs: int = 10) -> pl.DataFrame:
    """Generate multi-pair data with random independent price movements."""
    np.random.seed(42)
    base = datetime(2024, 1, 1)
    rows = []
    for p in range(n_pairs):
        price = 100.0
        for i in range(n):
            price *= np.exp(np.random.randn() * 0.005)
            rows.append(
                {
                    "pair": f"PAIR{p}",
                    "timestamp": base + timedelta(minutes=i),
                    "close": price,
                }
            )
    return pl.DataFrame(rows)


def _make_extreme_shock(
    n: int = 300,
    n_pairs: int = 10,
    event_idx: int = 200,
    shock_size: float = -0.08,
) -> pl.DataFrame:
    """Generate data where all pairs experience a large shock at event_idx."""
    np.random.seed(42)
    base = datetime(2024, 1, 1)
    rows = []
    for p in range(n_pairs):
        price = 100.0
        for i in range(n):
            if i == event_idx:
                change = shock_size
            else:
                change = np.random.randn() * 0.003
            price *= np.exp(change)
            rows.append(
                {
                    "pair": f"PAIR{p}",
                    "timestamp": base + timedelta(minutes=i),
                    "close": price,
                }
            )
    return pl.DataFrame(rows)


class TestZScoreEventDetector:
    def test_inherits_base(self):
        assert isinstance(ZScoreEventDetector(), EventDetectorBase)

    def test_component_type(self):
        assert ZScoreEventDetector.component_type == SfComponentType.EVENT_DETECTOR

    def test_detects_extreme_shock(self):
        df = _make_extreme_shock(n=300, n_pairs=10, event_idx=200, shock_size=-0.08)
        detector = ZScoreEventDetector(z_threshold=3.0, rolling_window=50, min_pairs=5)
        events = detector.detect(df)

        assert "_is_global_event" in events.columns
        assert "timestamp" in events.columns
        n_events = events.filter(pl.col("_is_global_event")).height
        assert n_events >= 1

    def test_normal_market_few_events(self):
        df = _make_normal_market(n=500, n_pairs=10)
        detector = ZScoreEventDetector(z_threshold=3.0, rolling_window=50, min_pairs=5)
        events = detector.detect(df)

        n_events = events.filter(pl.col("_is_global_event")).height
        event_rate = n_events / max(events.height, 1)
        assert event_rate < 0.05

    def test_higher_threshold_fewer_events(self):
        df = _make_extreme_shock(n=300, n_pairs=10, event_idx=200, shock_size=-0.08)

        det_low = ZScoreEventDetector(z_threshold=2.0, rolling_window=50, min_pairs=5)
        det_high = ZScoreEventDetector(z_threshold=4.0, rolling_window=50, min_pairs=5)

        events_low = det_low.detect(df).filter(pl.col("_is_global_event")).height
        events_high = det_high.detect(df).filter(pl.col("_is_global_event")).height

        assert events_high <= events_low

    def test_min_pairs_filter(self):
        df = _make_extreme_shock(n=300, n_pairs=3, event_idx=200)
        detector = ZScoreEventDetector(min_pairs=5)
        events = detector.detect(df)

        # 3 pairs < min_pairs=5 -> all timestamps filtered out
        assert events.height == 0

    def test_mask_targets_inherited(self):
        base = datetime(2024, 1, 1)
        n = 200
        timestamps = [base + timedelta(minutes=i) for i in range(n)]

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "pair": ["BTCUSDT"] * n,
                "target_direction_short": ["RISE"] * n,
            }
        )

        event_ts = timestamps[100]
        event_mask = pl.DataFrame(
            {
                "timestamp": [event_ts],
                "_is_global_event": [True],
            }
        )

        horizon = 30
        cooldown = 10
        detector = ZScoreEventDetector(cooldown_bars=cooldown)
        h_config = HorizonConfig(name="short", horizon=horizon)

        result = detector.mask_targets(
            df=df,
            event_timestamps=event_mask,
            horizon_configs=[h_config],
            target_columns_by_horizon={"short": ["target_direction_short"]},
        )

        col = result.get_column("target_direction_short")
        for i in range(n):
            if 70 <= i <= 110:
                assert col[i] is None, f"Expected null at index {i}"
            else:
                assert col[i] == "RISE", f"Expected RISE at index {i}"

    def test_missing_columns_raises(self):
        df = pl.DataFrame({"a": [1]})
        detector = ZScoreEventDetector()
        with pytest.raises(ValueError, match="Missing required columns"):
            detector.detect(df)

    def test_output_schema(self):
        df = _make_normal_market(n=200, n_pairs=10)
        detector = ZScoreEventDetector(rolling_window=30, min_pairs=5)
        events = detector.detect(df)

        assert set(events.columns) == {"timestamp", "_is_global_event"}
        assert events.schema["_is_global_event"] == pl.Boolean
