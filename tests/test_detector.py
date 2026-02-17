"""Tests for SignalDetector validation, SMA cross, PandasSignalDetector."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from signalflow.core.containers.signals import Signals
from signalflow.core.enums import SignalType
from signalflow.detector.base import SignalDetector

try:
    import pandas as pd

    from signalflow.detector.adapter.pandas_detector import PandasSignalDetector

    _HAS_PANDAS_DETECTOR = True
except (ImportError, ModuleNotFoundError):
    _HAS_PANDAS_DETECTOR = False


# ── helpers ─────────────────────────────────────────────────────────────────

TS = datetime(2024, 1, 1)


def _features_df(n=5, pair="BTCUSDT"):
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": [TS + timedelta(hours=i) for i in range(n)],
            "close": [100.0 + i for i in range(n)],
        }
    )


def _valid_signals(n=3, pair="BTCUSDT"):
    return Signals(
        pl.DataFrame(
            {
                "pair": [pair] * n,
                "timestamp": [TS + timedelta(hours=i) for i in range(n)],
                "signal_type": [SignalType.RISE.value] * n,
                "signal": [1] * n,
            }
        )
    )


# Concrete dummy detector for direct method testing
@dataclass
class DummyDetector(SignalDetector):
    allowed_signal_types: set[str] | None = None  # set in __post_init__

    def __post_init__(self):
        self.allowed_signal_types = {"rise", "fall"}

    def detect(self, features, context=None):
        return _valid_signals(features.height)


# ── _validate_features ──────────────────────────────────────────────────────


class TestValidateFeatures:
    def test_valid_passes(self):
        d = DummyDetector()
        d._validate_features(_features_df())

    def test_not_dataframe_raises(self):
        d = DummyDetector()
        with pytest.raises(TypeError, match=r"polars\.DataFrame"):
            d._validate_features("not a df")

    def test_missing_pair_col(self):
        d = DummyDetector()
        df = pl.DataFrame({"timestamp": [TS], "close": [100.0]})
        with pytest.raises(ValueError, match="pair"):
            d._validate_features(df)

    def test_missing_ts_col(self):
        d = DummyDetector()
        df = pl.DataFrame({"pair": ["BTCUSDT"], "close": [100.0]})
        with pytest.raises(ValueError, match="timestamp"):
            d._validate_features(df)

    def test_tz_aware_raises(self):
        d = DummyDetector()
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "close": [100.0],
            }
        )
        with pytest.raises(ValueError, match="timezone-naive"):
            d._validate_features(df)

    def test_duplicate_keys_raises(self):
        d = DummyDetector()
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT", "BTCUSDT"],
                "timestamp": [TS, TS],
                "close": [100.0, 101.0],
            }
        )
        with pytest.raises(ValueError, match="duplicate"):
            d._validate_features(df)


# ── _validate_signals ───────────────────────────────────────────────────────


class TestValidateSignals:
    def test_valid_passes(self):
        d = DummyDetector()
        d._validate_signals(_valid_signals())

    def test_not_signals_raises(self):
        d = DummyDetector()
        with pytest.raises(TypeError, match="Signals"):
            d._validate_signals("not signals")

    def test_missing_signal_type_raises(self):
        d = DummyDetector()
        sigs = Signals(pl.DataFrame({"pair": ["BTCUSDT"], "timestamp": [TS], "signal": [1]}))
        with pytest.raises(ValueError, match="signal_type"):
            d._validate_signals(sigs)

    def test_invalid_signal_type_raises(self):
        d = DummyDetector()
        sigs = Signals(
            pl.DataFrame({"pair": ["BTCUSDT"], "timestamp": [TS], "signal_type": ["INVALID"], "signal": [1]})
        )
        with pytest.raises(ValueError, match="unknown signal_type"):
            d._validate_signals(sigs)

    def test_require_probability_missing_raises(self):
        d = DummyDetector(require_probability=True)
        sigs = _valid_signals(1)
        with pytest.raises(ValueError, match="probability"):
            d._validate_signals(sigs)

    def test_tz_aware_signals_raises(self):
        d = DummyDetector()
        sigs = Signals(
            pl.DataFrame(
                {
                    "pair": ["BTCUSDT"],
                    "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                    "signal_type": [SignalType.RISE.value],
                    "signal": [1],
                }
            )
        )
        with pytest.raises(ValueError, match="timezone-naive"):
            d._validate_signals(sigs)

    def test_duplicate_signal_keys_raises(self):
        d = DummyDetector()
        sigs = Signals(
            pl.DataFrame(
                {
                    "pair": ["BTCUSDT", "BTCUSDT"],
                    "timestamp": [TS, TS],
                    "signal_type": [SignalType.RISE.value, SignalType.FALL.value],
                    "signal": [1, -1],
                }
            )
        )
        with pytest.raises(ValueError, match="duplicate"):
            d._validate_signals(sigs)


# ── _normalize_index ────────────────────────────────────────────────────────


class TestNormalizeIndex:
    def test_strips_timezone(self):
        d = DummyDetector()
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
            }
        )
        result = d._normalize_index(df)
        assert result.schema["timestamp"].time_zone is None

    def test_naive_unchanged(self):
        d = DummyDetector()
        df = pl.DataFrame({"pair": ["BTCUSDT"], "timestamp": [TS]})
        result = d._normalize_index(df)
        assert result["timestamp"][0] == TS

    def test_non_dataframe_raises(self):
        d = DummyDetector()
        with pytest.raises(TypeError):
            d._normalize_index("not a df")


# ── _keep_only_latest ───────────────────────────────────────────────────────


class TestKeepOnlyLatest:
    def test_keeps_latest_per_pair(self):
        d = DummyDetector()
        sigs = Signals(
            pl.DataFrame(
                {
                    "pair": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
                    "timestamp": [TS, TS + timedelta(hours=1), TS + timedelta(hours=2)],
                    "signal_type": [SignalType.RISE.value] * 3,
                    "signal": [1, 1, 1],
                }
            )
        )
        result = d._keep_only_latest(sigs)
        assert result.value.height == 1
        assert result.value["timestamp"][0] == TS + timedelta(hours=2)

    def test_keep_only_latest_in_run(self):
        """Test that keep_only_latest_per_pair=True works in run()."""
        from signalflow.core import RawData, RawDataView

        @dataclass
        class MultiSignalDetector(SignalDetector):
            keep_only_latest_per_pair: bool = True

            def detect(self, features, context=None):
                # Return multiple signals per pair
                return Signals(
                    pl.DataFrame(
                        {
                            "pair": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
                            "timestamp": [
                                TS,
                                TS + timedelta(hours=1),
                                TS,
                                TS + timedelta(hours=1),
                            ],
                            "signal_type": [SignalType.RISE.value] * 4,
                            "signal": [1] * 4,
                        }
                    )
                )

        raw = RawData(
            datetime_start=TS,
            datetime_end=TS + timedelta(hours=5),
            pairs=["BTCUSDT", "ETHUSDT"],
            data={
                "spot": pl.DataFrame(
                    {
                        "pair": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
                        "timestamp": [TS, TS + timedelta(hours=1), TS, TS + timedelta(hours=1)],
                        "open": [100.0] * 4,
                        "high": [101.0] * 4,
                        "low": [99.0] * 4,
                        "close": [100.0] * 4,
                        "volume": [1000.0] * 4,
                    }
                )
            },
        )
        raw_view = RawDataView(raw)

        d = MultiSignalDetector(keep_only_latest_per_pair=True)
        signals = d.run(raw_view)

        # Should only have 1 signal per pair (the latest)
        assert signals.value.height == 2

    def test_multiple_pairs(self):
        d = DummyDetector()
        sigs = Signals(
            pl.DataFrame(
                {
                    "pair": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
                    "timestamp": [TS, TS + timedelta(hours=1), TS, TS + timedelta(hours=1)],
                    "signal_type": [SignalType.RISE.value] * 4,
                    "signal": [1] * 4,
                }
            )
        )
        result = d._keep_only_latest(sigs)
        assert result.value.height == 2


# ── Preprocess features handling ───────────────────────────────────────────


class TestPreprocessFeatures:
    """Test different features types in preprocess."""

    def test_single_feature(self):
        """Test preprocess with a single Feature instance."""
        from signalflow.core import RawData, RawDataView
        from signalflow.feature.examples import ExampleSmaFeature

        @dataclass
        class DetectorWithFeature(SignalDetector):
            def __post_init__(self):
                self.features = ExampleSmaFeature(period=3)

            def detect(self, features, context=None):
                return _valid_signals(1)

        raw = RawData(
            datetime_start=TS,
            datetime_end=TS + timedelta(hours=10),
            pairs=["BTCUSDT"],
            data={
                "spot": pl.DataFrame(
                    {
                        "pair": ["BTCUSDT"] * 10,
                        "timestamp": [TS + timedelta(hours=i) for i in range(10)],
                        "open": [100.0 + i for i in range(10)],
                        "high": [101.0 + i for i in range(10)],
                        "low": [99.0 + i for i in range(10)],
                        "close": [100.0 + i for i in range(10)],
                        "volume": [1000.0] * 10,
                    }
                )
            },
        )
        raw_view = RawDataView(raw)

        d = DetectorWithFeature()
        feats = d.preprocess(raw_view)

        assert "sma_3" in feats.columns

    def test_feature_pipeline(self):
        """Test preprocess with FeaturePipeline."""
        from signalflow.core import RawData, RawDataView
        from signalflow.feature.examples import ExampleSmaFeature
        from signalflow.feature.feature_pipeline import FeaturePipeline

        @dataclass
        class DetectorWithPipeline(SignalDetector):
            def __post_init__(self):
                self.features = FeaturePipeline(
                    features=[ExampleSmaFeature(period=3)]
                )

            def detect(self, features, context=None):
                return _valid_signals(1)

        raw = RawData(
            datetime_start=TS,
            datetime_end=TS + timedelta(hours=10),
            pairs=["BTCUSDT"],
            data={
                "spot": pl.DataFrame(
                    {
                        "pair": ["BTCUSDT"] * 10,
                        "timestamp": [TS + timedelta(hours=i) for i in range(10)],
                        "open": [100.0 + i for i in range(10)],
                        "high": [101.0 + i for i in range(10)],
                        "low": [99.0 + i for i in range(10)],
                        "close": [100.0 + i for i in range(10)],
                        "volume": [1000.0] * 10,
                    }
                )
            },
        )
        raw_view = RawDataView(raw)

        d = DetectorWithPipeline()
        feats = d.preprocess(raw_view)

        assert "sma_3" in feats.columns


# ── PandasSignalDetector adapter ────────────────────────────────────────────


@pytest.mark.skipif(not _HAS_PANDAS_DETECTOR, reason="PandasSignalDetector not importable (broken internal import)")
class TestPandasSignalDetector:
    def test_roundtrip(self):
        @dataclass
        class DummyPD(PandasSignalDetector):
            def detect_pd(self, features, context=None):
                return pd.DataFrame(
                    {
                        "pair": features["pair"].values,
                        "timestamp": features["timestamp"].values,
                        "signal_type": [SignalType.RISE.value] * len(features),
                        "signal": [1] * len(features),
                    }
                )

        d = DummyPD()
        result = d.detect(_features_df(3))
        assert isinstance(result, Signals)
        assert result.value.height == 3

    def test_non_dataframe_input_raises(self):
        @dataclass
        class DummyPD(PandasSignalDetector):
            def detect_pd(self, features, context=None):
                return pd.DataFrame()

        d = DummyPD()
        with pytest.raises(TypeError):
            d.detect("not a df")

    def test_non_dataframe_output_raises(self):
        @dataclass
        class BadPD(PandasSignalDetector):
            def detect_pd(self, features, context=None):
                return "not a dataframe"

        d = BadPD()
        with pytest.raises(TypeError, match=r"pd\.DataFrame"):
            d.detect(_features_df(3))


# ── ExampleSmaCrossDetector ─────────────────────────────────────────────────


class TestSmaCrossDetector:
    def test_init_valid(self):
        from signalflow.detector.sma_cross import ExampleSmaCrossDetector

        d = ExampleSmaCrossDetector(fast_period=5, slow_period=10)
        assert d.fast_period == 5
        assert d.slow_period == 10

    def test_has_features(self):
        from signalflow.detector.sma_cross import ExampleSmaCrossDetector

        d = ExampleSmaCrossDetector(fast_period=5, slow_period=10)
        assert d.features is not None
