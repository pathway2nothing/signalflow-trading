"""Tests for GlobalEventDetector."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.core.enums import SfComponentType
from signalflow.detector.event.base import EventDetectorBase
from signalflow.detector.event.global_detector import GlobalEventDetector
from signalflow.target.multi_target_generator import HorizonConfig


def _make_normal_market(n: int = 200, n_pairs: int = 10) -> pl.DataFrame:
    """Generate multi-pair data with random independent price movements."""
    np.random.seed(42)
    base = datetime(2024, 1, 1)
    rows = []
    for p in range(n_pairs):
        price = 100.0
        for i in range(n):
            price *= np.exp(np.random.randn() * 0.01)
            rows.append(
                {
                    "pair": f"PAIR{p}",
                    "timestamp": base + timedelta(minutes=i),
                    "close": price,
                }
            )
    return pl.DataFrame(rows)


def _make_synchronized_drop(
    n: int = 200,
    n_pairs: int = 10,
    event_idx: int = 100,
) -> pl.DataFrame:
    """Generate data where all pairs drop at event_idx."""
    np.random.seed(42)
    base = datetime(2024, 1, 1)
    rows = []
    for p in range(n_pairs):
        price = 100.0
        for i in range(n):
            if i == event_idx:
                change = -0.05  # All pairs drop 5%
            else:
                change = np.random.randn() * 0.005
            price *= np.exp(change)
            rows.append(
                {
                    "pair": f"PAIR{p}",
                    "timestamp": base + timedelta(minutes=i),
                    "close": price,
                }
            )
    return pl.DataFrame(rows)


class TestGlobalEventDetector:
    def test_inherits_base(self):
        assert isinstance(GlobalEventDetector(), EventDetectorBase)

    def test_component_type(self):
        assert GlobalEventDetector.component_type == SfComponentType.EVENT_DETECTOR

    def test_detects_synchronized_drop(self):
        df = _make_synchronized_drop(n=200, n_pairs=10, event_idx=100)
        detector = GlobalEventDetector(agreement_threshold=0.8, min_pairs=5)
        events = detector.detect(df)

        assert "_is_global_event" in events.columns
        n_events = events.filter(pl.col("_is_global_event")).height
        assert n_events >= 1  # Should detect at least the synchronized drop

    def test_normal_market_few_events(self):
        df = _make_normal_market(n=500, n_pairs=10)
        detector = GlobalEventDetector(agreement_threshold=0.9, min_pairs=5)
        events = detector.detect(df)

        n_events = events.filter(pl.col("_is_global_event")).height
        # With random movements, very few timestamps should have >90% agreement
        # Allow some by chance, but should be rare relative to total
        event_rate = n_events / events.height
        assert event_rate < 0.15

    def test_min_pairs_threshold(self):
        """Should not detect events with fewer pairs than min_pairs."""
        df = _make_synchronized_drop(n=200, n_pairs=3, event_idx=100)
        detector = GlobalEventDetector(min_pairs=5)
        events = detector.detect(df)

        # Should have no events because we have only 3 pairs < min_pairs=5
        n_events = events.filter(pl.col("_is_global_event")).height
        assert n_events == 0

    def test_mask_targets_nullifies_range(self):
        """mask_targets should null target columns in [T-horizon, T+cooldown]."""
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

        # Event at index 100
        event_ts = timestamps[100]
        event_mask = pl.DataFrame(
            {
                "timestamp": [event_ts],
                "_is_global_event": [True],
            }
        )

        horizon = 30
        cooldown = 10
        detector = GlobalEventDetector(cooldown_bars=cooldown)
        h_config = HorizonConfig(name="short", horizon=horizon)

        result = detector.mask_targets(
            df=df,
            event_timestamps=event_mask,
            horizon_configs=[h_config],
            target_columns_by_horizon={"short": ["target_direction_short"]},
        )

        col = result.get_column("target_direction_short")
        # Expected null range: [100-30, 100+10] = [70, 110]
        for i in range(n):
            if 70 <= i <= 110:
                assert col[i] is None, f"Expected null at index {i}, got {col[i]}"
            else:
                assert col[i] == "RISE", f"Expected RISE at index {i}, got {col[i]}"

    def test_mask_no_events(self):
        """No events -> no masking."""
        base = datetime(2024, 1, 1)
        n = 100
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(minutes=i) for i in range(n)],
                "pair": ["BTCUSDT"] * n,
                "target_direction_short": ["RISE"] * n,
            }
        )
        event_mask = pl.DataFrame(
            {
                "timestamp": [base],
                "_is_global_event": [False],
            }
        )

        detector = GlobalEventDetector()
        result = detector.mask_targets(
            df=df,
            event_timestamps=event_mask,
            horizon_configs=[HorizonConfig(name="short", horizon=30)],
            target_columns_by_horizon={"short": ["target_direction_short"]},
        )

        nulls = result.get_column("target_direction_short").null_count()
        assert nulls == 0

    def test_missing_columns_raises(self):
        df = pl.DataFrame({"a": [1]})
        detector = GlobalEventDetector()
        with pytest.raises(ValueError, match="Missing required columns"):
            detector.detect(df)
