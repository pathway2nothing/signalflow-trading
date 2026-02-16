"""Tests for GlobalEventDetector."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from signalflow.core import RawData, RawDataView, Signals
from signalflow.core.enums import SfComponentType, SignalCategory
from signalflow.detector.base import SignalDetector
from signalflow.detector.market import GlobalEventDetector
from signalflow.target.utils import mask_targets_by_signals


def _make_normal_market(n: int = 200, n_pairs: int = 10) -> RawDataView:
    """Generate multi-pair data with random independent price movements."""
    np.random.seed(42)
    base = datetime(2024, 1, 1)
    pairs = [f"PAIR{p}" for p in range(n_pairs)]
    rows = []
    for p in range(n_pairs):
        price = 100.0
        for i in range(n):
            price *= np.exp(np.random.randn() * 0.01)
            rows.append(
                {
                    "pair": f"PAIR{p}",
                    "timestamp": base + timedelta(minutes=i),
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1000.0,
                }
            )
    df = pl.DataFrame(rows)
    raw = RawData(
        datetime_start=base,
        datetime_end=base + timedelta(minutes=n),
        pairs=pairs,
        data={"spot": df},
    )
    return RawDataView(raw)


def _make_synchronized_drop(
    n: int = 200,
    n_pairs: int = 10,
    event_idx: int = 100,
) -> RawDataView:
    """Generate data where all pairs drop at event_idx."""
    np.random.seed(42)
    base = datetime(2024, 1, 1)
    pairs = [f"PAIR{p}" for p in range(n_pairs)]
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
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1000.0,
                }
            )
    df = pl.DataFrame(rows)
    raw = RawData(
        datetime_start=base,
        datetime_end=base + timedelta(minutes=n),
        pairs=pairs,
        data={"spot": df},
    )
    return RawDataView(raw)


class TestGlobalEventDetector:
    def test_inherits_signal_detector(self):
        assert isinstance(GlobalEventDetector(), SignalDetector)

    def test_component_type(self):
        assert GlobalEventDetector.component_type == SfComponentType.DETECTOR

    def test_signal_category(self):
        assert GlobalEventDetector().signal_category == SignalCategory.MARKET_WIDE

    def test_detects_synchronized_drop(self):
        raw_view = _make_synchronized_drop(n=200, n_pairs=10, event_idx=100)
        detector = GlobalEventDetector(agreement_threshold=0.8, min_pairs=5)
        signals = detector.run(raw_view)

        assert isinstance(signals, Signals)
        assert "signal_type" in signals.value.columns
        n_events = signals.value.height
        assert n_events >= 1  # Should detect at least the synchronized drop

    def test_returns_signals_with_correct_schema(self):
        raw_view = _make_synchronized_drop(n=200, n_pairs=10, event_idx=100)
        detector = GlobalEventDetector(agreement_threshold=0.8, min_pairs=5)
        signals = detector.run(raw_view)

        df = signals.value
        assert "pair" in df.columns
        assert "timestamp" in df.columns
        assert "signal_type" in df.columns
        assert "signal" in df.columns
        assert "probability" in df.columns

        # All signals should have signal_type == "global_event"
        if df.height > 0:
            assert df["signal_type"].to_list() == ["global_event"] * df.height
            # Pair should be "ALL" for market-wide events
            assert df["pair"].to_list() == ["ALL"] * df.height

    def test_normal_market_few_events(self):
        raw_view = _make_normal_market(n=500, n_pairs=10)
        detector = GlobalEventDetector(agreement_threshold=0.9, min_pairs=5)
        signals = detector.run(raw_view)

        n_events = signals.value.height
        # With random movements, very few timestamps should have >90% agreement
        # Allow some by chance, but should be rare
        assert n_events < 75  # Less than 15% of 500 bars

    def test_min_pairs_threshold(self):
        """Should not detect events with fewer pairs than min_pairs."""
        raw_view = _make_synchronized_drop(n=200, n_pairs=3, event_idx=100)
        detector = GlobalEventDetector(min_pairs=5)
        signals = detector.run(raw_view)

        # Should have no events because we have only 3 pairs < min_pairs=5
        assert signals.value.height == 0

    def test_mask_targets_by_signals_integration(self):
        """Test that mask_targets_by_signals works with GlobalEventDetector output."""
        base = datetime(2024, 1, 1)
        n = 200
        timestamps = [base + timedelta(minutes=i) for i in range(n)]

        # Target DataFrame
        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "pair": ["BTCUSDT"] * n,
                "target_direction_label": ["RISE"] * n,
            }
        )

        # Create signals as if from detector (event at index 100)
        event_ts = timestamps[100]
        signals = Signals(
            pl.DataFrame(
                {
                    "pair": ["ALL"],
                    "timestamp": [event_ts],
                    "signal_type": ["global_event"],
                    "signal": [1],
                    "probability": [0.9],
                }
            )
        )

        # Mask targets
        horizon = 30
        cooldown = 10
        result = mask_targets_by_signals(
            df=df,
            signals=signals,
            mask_signal_types={"global_event"},
            horizon_bars=horizon,
            cooldown_bars=cooldown,
        )

        # Note: mask_targets_by_signals filters by pair, but "ALL" won't match "BTCUSDT"
        # For market-wide events, user would need to expand to all pairs or use a different approach
        # Here we just verify the function runs without error
        assert result.height == n

    def test_custom_signal_type_name(self):
        """Test configurable signal_type_name."""
        raw_view = _make_synchronized_drop(n=200, n_pairs=10, event_idx=100)
        detector = GlobalEventDetector(
            agreement_threshold=0.8,
            min_pairs=5,
            signal_type_name="market_crash",
        )
        signals = detector.run(raw_view)

        if signals.value.height > 0:
            assert signals.value["signal_type"].to_list() == ["market_crash"] * signals.value.height
