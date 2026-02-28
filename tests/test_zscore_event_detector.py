"""Tests for ZScoreEventDetector."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from signalflow.core import RawData, RawDataView, Signals
from signalflow.core.enums import SfComponentType, SignalCategory
from signalflow.detector.base import SignalDetector
from signalflow.detector.market import ZScoreEventDetector


def _make_normal_market(n: int = 300, n_pairs: int = 10) -> RawDataView:
    """Generate multi-pair data with random independent price movements."""
    np.random.seed(42)
    base = datetime(2024, 1, 1)
    pairs = [f"PAIR{p}" for p in range(n_pairs)]
    rows = []
    for p in range(n_pairs):
        price = 100.0
        for i in range(n):
            price *= np.exp(np.random.randn() * 0.005)
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


def _make_extreme_shock(
    n: int = 300,
    n_pairs: int = 10,
    event_idx: int = 200,
    shock_size: float = -0.08,
) -> RawDataView:
    """Generate data where all pairs experience a large shock at event_idx."""
    np.random.seed(42)
    base = datetime(2024, 1, 1)
    pairs = [f"PAIR{p}" for p in range(n_pairs)]
    rows = []
    for p in range(n_pairs):
        price = 100.0
        for i in range(n):
            change = shock_size if i == event_idx else np.random.randn() * 0.003
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


class TestZScoreEventDetector:
    def test_inherits_signal_detector(self):
        assert isinstance(ZScoreEventDetector(), SignalDetector)

    def test_component_type(self):
        assert ZScoreEventDetector.component_type == SfComponentType.DETECTOR

    def test_signal_category(self):
        assert ZScoreEventDetector().signal_category == SignalCategory.MARKET_WIDE

    def test_detects_extreme_shock(self):
        raw_view = _make_extreme_shock(n=300, n_pairs=10, event_idx=200, shock_size=-0.08)
        detector = ZScoreEventDetector(z_threshold=3.0, rolling_window=50, min_pairs=5)
        signals = detector.run(raw_view)

        assert isinstance(signals, Signals)
        assert "signal_type" in signals.value.columns
        n_events = signals.value.height
        assert n_events >= 1

    def test_returns_signals_with_correct_schema(self):
        raw_view = _make_extreme_shock(n=300, n_pairs=10, event_idx=200, shock_size=-0.08)
        detector = ZScoreEventDetector(z_threshold=3.0, rolling_window=50, min_pairs=5)
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
        detector = ZScoreEventDetector(z_threshold=3.0, rolling_window=50, min_pairs=5)
        signals = detector.run(raw_view)

        n_events = signals.value.height
        # With random movements, z-score > 3 should be rare
        assert n_events < 25  # Less than 5% of 500 bars

    def test_higher_threshold_fewer_events(self):
        raw_view = _make_extreme_shock(n=300, n_pairs=10, event_idx=200, shock_size=-0.08)

        det_low = ZScoreEventDetector(z_threshold=2.0, rolling_window=50, min_pairs=5)
        det_high = ZScoreEventDetector(z_threshold=4.0, rolling_window=50, min_pairs=5)

        events_low = det_low.run(raw_view).value.height
        events_high = det_high.run(raw_view).value.height

        assert events_high <= events_low

    def test_min_pairs_filter(self):
        raw_view = _make_extreme_shock(n=300, n_pairs=3, event_idx=200)
        detector = ZScoreEventDetector(min_pairs=5)
        signals = detector.run(raw_view)

        # 3 pairs < min_pairs=5 -> all timestamps filtered out
        assert signals.value.height == 0

    def test_custom_signal_type_name(self):
        """Test configurable signal_type_name."""
        raw_view = _make_extreme_shock(n=300, n_pairs=10, event_idx=200, shock_size=-0.08)
        detector = ZScoreEventDetector(
            z_threshold=3.0,
            rolling_window=50,
            min_pairs=5,
            signal_type_name="zscore_event",
        )
        signals = detector.run(raw_view)

        if signals.value.height > 0:
            assert signals.value["signal_type"].to_list() == ["zscore_event"] * signals.value.height
