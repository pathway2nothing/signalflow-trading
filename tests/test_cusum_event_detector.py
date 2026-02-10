"""Tests for CusumEventDetector."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from signalflow.core import RawData, RawDataView, Signals
from signalflow.core.enums import SfComponentType, SignalCategory
from signalflow.detector.base import SignalDetector
from signalflow.detector.market import CusumEventDetector


def _make_normal_market(n: int = 300, n_pairs: int = 10) -> RawDataView:
    """Generate multi-pair data with random independent price movements."""
    np.random.seed(42)
    base = datetime(2024, 1, 1)
    pairs = [f"PAIR{p}" for p in range(n_pairs)]
    rows = []
    for p in range(n_pairs):
        price = 100.0
        for i in range(n):
            price *= np.exp(np.random.randn() * 0.003)
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


def _make_regime_shift(
    n: int = 300,
    n_pairs: int = 10,
    shift_start: int = 150,
    shift_duration: int = 30,
    drift_per_bar: float = -0.015,
) -> RawDataView:
    """Generate data with a sustained downward drift starting at shift_start."""
    np.random.seed(42)
    base = datetime(2024, 1, 1)
    pairs = [f"PAIR{p}" for p in range(n_pairs)]
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


class TestCusumEventDetector:
    def test_inherits_signal_detector(self):
        assert isinstance(CusumEventDetector(), SignalDetector)

    def test_component_type(self):
        assert CusumEventDetector.component_type == SfComponentType.DETECTOR

    def test_signal_category(self):
        assert CusumEventDetector().signal_category == SignalCategory.MARKET_WIDE

    def test_detects_regime_shift(self):
        raw_view = _make_regime_shift(n=300, n_pairs=10, shift_start=150, shift_duration=30, drift_per_bar=-0.015)
        detector = CusumEventDetector(
            drift=0.003,
            cusum_threshold=0.03,
            rolling_window=50,
            min_pairs=5,
        )
        signals = detector.run(raw_view)

        assert isinstance(signals, Signals)
        assert "signal_type" in signals.value.columns
        n_events = signals.value.height
        assert n_events >= 1

    def test_returns_signals_with_correct_schema(self):
        raw_view = _make_regime_shift(n=300, n_pairs=10, shift_start=150, shift_duration=30, drift_per_bar=-0.015)
        detector = CusumEventDetector(
            drift=0.003,
            cusum_threshold=0.03,
            rolling_window=50,
            min_pairs=5,
        )
        signals = detector.run(raw_view)

        df = signals.value
        assert "pair" in df.columns
        assert "timestamp" in df.columns
        assert "signal_type" in df.columns
        assert "signal" in df.columns
        assert "probability" in df.columns

        if df.height > 0:
            assert df["signal_type"].to_list() == ["global_event"] * df.height
            assert df["pair"].to_list() == ["ALL"] * df.height

    def test_normal_market_few_events(self):
        raw_view = _make_normal_market(n=500, n_pairs=10)
        detector = CusumEventDetector(
            drift=0.005,
            cusum_threshold=0.05,
            rolling_window=50,
            min_pairs=5,
        )
        signals = detector.run(raw_view)

        n_events = signals.value.height
        # With random movements, few events should be detected
        assert n_events < 25  # Less than 5% of 500 bars

    def test_cusum_reset_after_event(self):
        """After event detection, CUSUM counters reset to 0."""
        raw_view = _make_regime_shift(n=300, n_pairs=10, shift_start=150, shift_duration=30, drift_per_bar=-0.015)
        detector = CusumEventDetector(
            drift=0.001,
            cusum_threshold=0.02,
            rolling_window=50,
            min_pairs=5,
        )
        signals = detector.run(raw_view)

        if signals.value.height >= 2:
            # If multiple events, they shouldn't be consecutive bars
            # (reset means accumulation restarts)
            event_ts = signals.value.get_column("timestamp").to_list()
            # Check that there are gaps between events
            for i in range(len(event_ts) - 1):
                gap = (event_ts[i + 1] - event_ts[i]).total_seconds() / 60  # minutes
                assert gap > 1, "Events should not be on consecutive bars after reset"

    def test_higher_threshold_fewer_events(self):
        raw_view = _make_regime_shift(n=300, n_pairs=10, shift_start=150)

        det_low = CusumEventDetector(drift=0.003, cusum_threshold=0.02, rolling_window=50, min_pairs=5)
        det_high = CusumEventDetector(drift=0.003, cusum_threshold=0.10, rolling_window=50, min_pairs=5)

        events_low = det_low.run(raw_view).value.height
        events_high = det_high.run(raw_view).value.height

        assert events_high <= events_low

    def test_min_pairs_filter(self):
        raw_view = _make_regime_shift(n=300, n_pairs=3, shift_start=150)
        detector = CusumEventDetector(min_pairs=5)
        signals = detector.run(raw_view)

        assert signals.value.height == 0

    def test_custom_signal_type_name(self):
        """Test configurable signal_type_name."""
        raw_view = _make_regime_shift(n=300, n_pairs=10, shift_start=150, shift_duration=30, drift_per_bar=-0.015)
        detector = CusumEventDetector(
            drift=0.003,
            cusum_threshold=0.03,
            rolling_window=50,
            min_pairs=5,
            signal_type_name="cusum_event",
        )
        signals = detector.run(raw_view)

        if signals.value.height > 0:
            assert signals.value["signal_type"].to_list() == ["cusum_event"] * signals.value.height
