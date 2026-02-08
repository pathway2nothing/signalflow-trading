"""Tests for CusumEventDetector."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.core.enums import SfComponentType
from signalflow.detector.event.cusum_detector import CusumEventDetector
from signalflow.detector.event.base import EventDetectorBase
from signalflow.target.multi_target_generator import HorizonConfig


def _make_normal_market(n: int = 300, n_pairs: int = 10) -> pl.DataFrame:
    """Generate multi-pair data with random independent price movements."""
    np.random.seed(42)
    base = datetime(2024, 1, 1)
    rows = []
    for p in range(n_pairs):
        price = 100.0
        for i in range(n):
            price *= np.exp(np.random.randn() * 0.003)
            rows.append(
                {
                    "pair": f"PAIR{p}",
                    "timestamp": base + timedelta(minutes=i),
                    "close": price,
                }
            )
    return pl.DataFrame(rows)


def _make_regime_shift(
    n: int = 300,
    n_pairs: int = 10,
    shift_start: int = 150,
    shift_duration: int = 30,
    drift_per_bar: float = -0.015,
) -> pl.DataFrame:
    """Generate data with a sustained downward drift starting at shift_start."""
    np.random.seed(42)
    base = datetime(2024, 1, 1)
    rows = []
    for p in range(n_pairs):
        price = 100.0
        for i in range(n):
            if shift_start <= i < shift_start + shift_duration:
                change = drift_per_bar + np.random.randn() * 0.002
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


class TestCusumEventDetector:
    def test_inherits_base(self):
        assert isinstance(CusumEventDetector(), EventDetectorBase)

    def test_component_type(self):
        assert CusumEventDetector.component_type == SfComponentType.EVENT_DETECTOR

    def test_detects_regime_shift(self):
        df = _make_regime_shift(n=300, n_pairs=10, shift_start=150, shift_duration=30, drift_per_bar=-0.015)
        detector = CusumEventDetector(
            drift=0.003,
            cusum_threshold=0.03,
            rolling_window=50,
            min_pairs=5,
        )
        events = detector.detect(df)

        assert "_is_global_event" in events.columns
        n_events = events.filter(pl.col("_is_global_event")).height
        assert n_events >= 1

    def test_normal_market_few_events(self):
        df = _make_normal_market(n=500, n_pairs=10)
        detector = CusumEventDetector(
            drift=0.005,
            cusum_threshold=0.05,
            rolling_window=50,
            min_pairs=5,
        )
        events = detector.detect(df)

        n_events = events.filter(pl.col("_is_global_event")).height
        event_rate = n_events / max(events.height, 1)
        assert event_rate < 0.05

    def test_cusum_reset_after_event(self):
        """After event detection, CUSUM counters reset to 0."""
        df = _make_regime_shift(n=300, n_pairs=10, shift_start=150, shift_duration=30, drift_per_bar=-0.015)
        detector = CusumEventDetector(
            drift=0.001,
            cusum_threshold=0.02,
            rolling_window=50,
            min_pairs=5,
        )
        events = detector.detect(df)

        event_rows = events.filter(pl.col("_is_global_event"))
        if event_rows.height >= 2:
            # If multiple events, they shouldn't be consecutive bars
            # (reset means accumulation restarts)
            event_indices = []
            all_ts = events.get_column("timestamp").to_list()
            for ts in event_rows.get_column("timestamp").to_list():
                event_indices.append(all_ts.index(ts))
            gaps = [event_indices[i + 1] - event_indices[i] for i in range(len(event_indices) - 1)]
            assert all(g > 1 for g in gaps), "Events should not be on consecutive bars after reset"

    def test_higher_threshold_fewer_events(self):
        df = _make_regime_shift(n=300, n_pairs=10, shift_start=150)

        det_low = CusumEventDetector(drift=0.003, cusum_threshold=0.02, rolling_window=50, min_pairs=5)
        det_high = CusumEventDetector(drift=0.003, cusum_threshold=0.10, rolling_window=50, min_pairs=5)

        events_low = det_low.detect(df).filter(pl.col("_is_global_event")).height
        events_high = det_high.detect(df).filter(pl.col("_is_global_event")).height

        assert events_high <= events_low

    def test_min_pairs_filter(self):
        df = _make_regime_shift(n=300, n_pairs=3, shift_start=150)
        detector = CusumEventDetector(min_pairs=5)
        events = detector.detect(df)

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
        detector = CusumEventDetector(cooldown_bars=cooldown)
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
        detector = CusumEventDetector()
        with pytest.raises(ValueError, match="Missing required columns"):
            detector.detect(df)

    def test_output_schema(self):
        df = _make_normal_market(n=200, n_pairs=10)
        detector = CusumEventDetector(rolling_window=30, min_pairs=5)
        events = detector.detect(df)

        assert set(events.columns) == {"timestamp", "_is_global_event"}
        assert events.schema["_is_global_event"] == pl.Boolean
