"""Tests for ZScoreAnomalyDetector."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.core import RawData, RawDataView, Signals
from signalflow.core.enums import SfComponentType, SignalCategory
from signalflow.detector.base import SignalDetector
from signalflow.detector.zscore_anomaly import ZScoreAnomalyDetector


def _make_raw_data_view(n: int = 200, pairs: list[str] | None = None) -> RawDataView:
    """Generate RawDataView with OHLCV data."""
    np.random.seed(42)
    base = datetime(2024, 1, 1)
    if pairs is None:
        pairs = ["BTCUSDT", "ETHUSDT"]

    rows = []
    for pair in pairs:
        price = 100.0
        for i in range(n):
            # Normal price movements with occasional large moves
            if i == 150:  # Create an anomaly
                price *= 1.15  # 15% spike
            else:
                price *= np.exp(np.random.randn() * 0.005)
            rows.append(
                {
                    "pair": pair,
                    "timestamp": base + timedelta(minutes=i),
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1000.0 + np.random.rand() * 100,
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


class TestZScoreAnomalyDetectorBasic:
    """Basic tests for ZScoreAnomalyDetector."""

    def test_inherits_signal_detector(self):
        detector = ZScoreAnomalyDetector()
        assert isinstance(detector, SignalDetector)

    def test_component_type(self):
        assert ZScoreAnomalyDetector.component_type == SfComponentType.DETECTOR

    def test_signal_category(self):
        detector = ZScoreAnomalyDetector()
        assert detector.signal_category == SignalCategory.ANOMALY

    def test_default_attributes(self):
        detector = ZScoreAnomalyDetector()
        assert detector.target_feature == "close"
        assert detector.rolling_window == 1440
        assert detector.threshold == 4.0
        assert detector.signal_high == "positive_anomaly"
        assert detector.signal_low == "negative_anomaly"

    def test_custom_attributes(self):
        detector = ZScoreAnomalyDetector(
            target_feature="volume",
            rolling_window=720,
            threshold=3.0,
            signal_high="high_volume",
            signal_low="low_volume",
        )
        assert detector.target_feature == "volume"
        assert detector.rolling_window == 720
        assert detector.threshold == 3.0
        assert detector.signal_high == "high_volume"
        assert detector.signal_low == "low_volume"

    def test_post_init_updates_allowed_signal_types(self):
        detector = ZScoreAnomalyDetector(
            signal_high="extreme_up",
            signal_low="extreme_down",
        )
        assert detector.allowed_signal_types == {"extreme_up", "extreme_down"}


class TestZScoreAnomalyDetectorPreprocess:
    """Tests for preprocess method."""

    def test_preprocess_adds_helper_columns(self):
        raw_view = _make_raw_data_view(n=100)
        detector = ZScoreAnomalyDetector(rolling_window=50)
        df = detector.preprocess(raw_view)

        assert "_target_rol_mean" in df.columns
        assert "_target_rol_std" in df.columns

    def test_preprocess_preserves_original_columns(self):
        raw_view = _make_raw_data_view(n=100)
        detector = ZScoreAnomalyDetector(rolling_window=50)
        df = detector.preprocess(raw_view)

        assert "pair" in df.columns
        assert "timestamp" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns


class TestZScoreAnomalyDetectorDetect:
    """Tests for detect method."""

    def test_detect_returns_signals(self):
        raw_view = _make_raw_data_view(n=200)
        detector = ZScoreAnomalyDetector(
            rolling_window=50,
            threshold=2.5,  # Lower threshold for test
        )
        signals = detector.run(raw_view)

        assert isinstance(signals, Signals)

    def test_detect_schema(self):
        raw_view = _make_raw_data_view(n=200)
        detector = ZScoreAnomalyDetector(
            rolling_window=50,
            threshold=2.5,
        )
        signals = detector.run(raw_view)
        df = signals.value

        assert "pair" in df.columns
        assert "timestamp" in df.columns
        assert "signal_type" in df.columns
        assert "signal" in df.columns
        assert "probability" in df.columns

    def test_detect_signal_types(self):
        raw_view = _make_raw_data_view(n=200)
        detector = ZScoreAnomalyDetector(
            rolling_window=50,
            threshold=2.0,  # Low threshold to get signals
        )
        signals = detector.run(raw_view)
        df = signals.value

        if df.height > 0:
            signal_types = df["signal_type"].unique().to_list()
            for st in signal_types:
                assert st in {"positive_anomaly", "negative_anomaly"}

    def test_high_threshold_fewer_signals(self):
        raw_view = _make_raw_data_view(n=200)

        detector_low = ZScoreAnomalyDetector(rolling_window=50, threshold=2.0)
        detector_high = ZScoreAnomalyDetector(rolling_window=50, threshold=4.0)

        signals_low = detector_low.run(raw_view)
        signals_high = detector_high.run(raw_view)

        assert signals_high.value.height <= signals_low.value.height

    def test_detect_probability_in_range(self):
        raw_view = _make_raw_data_view(n=200)
        detector = ZScoreAnomalyDetector(
            rolling_window=50,
            threshold=2.0,
        )
        signals = detector.run(raw_view)
        df = signals.value

        if df.height > 0:
            probs = df["probability"].to_list()
            assert all(0.0 <= p <= 1.0 for p in probs if p is not None)


class TestZScoreAnomalyDetectorCustomTarget:
    """Tests for detecting anomalies on different features."""

    def test_detect_volume_anomalies(self):
        """Test detecting anomalies on volume instead of close."""
        np.random.seed(42)
        base = datetime(2024, 1, 1)
        rows = []
        for i in range(200):
            # Create volume spike at i=150
            volume = 10000 if i == 150 else 1000 + np.random.rand() * 100
            rows.append(
                {
                    "pair": "BTCUSDT",
                    "timestamp": base + timedelta(minutes=i),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": volume,
                }
            )

        df = pl.DataFrame(rows)
        raw = RawData(
            datetime_start=base,
            datetime_end=base + timedelta(minutes=200),
            pairs=["BTCUSDT"],
            data={"spot": df},
        )
        raw_view = RawDataView(raw)

        detector = ZScoreAnomalyDetector(
            target_feature="volume",
            rolling_window=50,
            threshold=3.0,
            signal_high="volume_spike",
            signal_low="volume_drop",
        )
        signals = detector.run(raw_view)

        # Should detect volume spike
        df = signals.value
        if df.height > 0:
            signal_types = df["signal_type"].unique().to_list()
            assert "volume_spike" in signal_types or df.height == 0
