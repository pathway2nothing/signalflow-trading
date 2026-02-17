"""Integration tests for FeatureInformativenessAnalyzer."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.detector.market import GlobalEventDetector
from signalflow.feature.informativeness import (
    FeatureInformativenessAnalyzer,
    InformativenessReport,
    RollingMIConfig,
)
from signalflow.target.multi_target_generator import (
    HorizonConfig,
    MultiTargetGenerator,
    TargetType,
)


def _make_ohlcv_with_features(
    n: int = 2000,
    pairs: list[str] | None = None,
) -> pl.DataFrame:
    """Generate OHLCV with an informative and a noise feature."""
    if pairs is None:
        pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"]

    np.random.seed(42)
    base = datetime(2024, 1, 1)
    rows = []

    for pair in pairs:
        price = 100.0
        for i in range(n):
            # Trend component that features could detect
            trend = 0.002 * math.sin(i / 100.0)
            noise = np.random.randn() * 0.005
            ret = trend + noise
            price *= math.exp(ret)

            # Informative feature: leading indicator of trend
            feat_informative = math.sin((i + 10) / 100.0) + np.random.randn() * 0.3

            # Noise feature: completely random
            feat_noise = np.random.randn()

            rows.append(
                {
                    "pair": pair,
                    "timestamp": base + timedelta(minutes=i),
                    "open": price * 0.999,
                    "high": price * 1.005,
                    "low": price * 0.995,
                    "close": price,
                    "volume": 1000.0 + 500 * abs(math.sin(i / 30.0)) + np.random.rand() * 200,
                    "feat_informative": feat_informative,
                    "feat_noise": feat_noise,
                }
            )

    return pl.DataFrame(rows)


class TestFeatureInformativenessAnalyzer:
    @pytest.fixture
    def small_analyzer(self):
        """Analyzer with small horizons for fast tests."""
        return FeatureInformativenessAnalyzer(
            target_generator=MultiTargetGenerator(
                horizons=[HorizonConfig(name="short", horizon=30)],
                target_types=[TargetType(name="direction", kind="discrete")],
            ),
            event_detector=None,
            rolling_mi=RollingMIConfig(window_size=500),
            bins=15,
        )

    def test_full_pipeline_produces_report(self, small_analyzer):
        """Smoke test: full pipeline produces report without errors."""
        df = _make_ohlcv_with_features(n=1500, pairs=["BTCUSDT", "ETHUSDT"])
        report = small_analyzer.analyze(df, feature_columns=["feat_informative", "feat_noise"])

        assert isinstance(report, InformativenessReport)
        assert report.composite_scores.height == 2
        assert report.raw_mi.height > 0
        assert report.metadata["n_features"] == 2

    def test_informative_feature_ranks_higher(self):
        """Feature correlated with target should rank above noise."""
        analyzer = FeatureInformativenessAnalyzer(
            target_generator=MultiTargetGenerator(
                horizons=[HorizonConfig(name="short", horizon=30)],
                target_types=[
                    TargetType(name="direction", kind="discrete"),
                    TargetType(name="return_magnitude", kind="continuous"),
                ],
            ),
            event_detector=None,
            rolling_mi=RollingMIConfig(window_size=500),
            bins=15,
        )
        df = _make_ohlcv_with_features(n=2000, pairs=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        report = analyzer.analyze(df, feature_columns=["feat_informative", "feat_noise"])

        scores = report.composite_scores
        informative_score = scores.filter(pl.col("feature") == "feat_informative").get_column("composite_score")[0]
        noise_score = scores.filter(pl.col("feature") == "feat_noise").get_column("composite_score")[0]
        assert informative_score > noise_score

    def test_empty_feature_columns_raises(self, small_analyzer):
        df = _make_ohlcv_with_features(n=200)
        with pytest.raises(ValueError, match="feature_columns must not be empty"):
            small_analyzer.analyze(df, feature_columns=[])

    def test_missing_feature_columns_raises(self, small_analyzer):
        df = _make_ohlcv_with_features(n=200)
        with pytest.raises(ValueError, match="Feature columns not found"):
            small_analyzer.analyze(df, feature_columns=["nonexistent_column"])

    def test_missing_ohlcv_columns_raises(self, small_analyzer):
        df = pl.DataFrame({"pair": ["A"], "timestamp": [datetime(2024, 1, 1)], "feat": [1.0]})
        with pytest.raises(ValueError):
            small_analyzer.analyze(df, feature_columns=["feat"])

    def test_report_top_features(self, small_analyzer):
        df = _make_ohlcv_with_features(n=1500, pairs=["BTCUSDT", "ETHUSDT"])
        report = small_analyzer.analyze(df, ["feat_informative", "feat_noise"])
        top = report.top_features(1)
        assert top.height == 1

    def test_report_feature_detail(self, small_analyzer):
        df = _make_ohlcv_with_features(n=1500, pairs=["BTCUSDT", "ETHUSDT"])
        report = small_analyzer.analyze(df, ["feat_informative", "feat_noise"])
        detail = report.feature_detail("feat_informative")
        assert detail.height > 0
        assert "horizon" in detail.columns
        assert "target_type" in detail.columns

    def test_score_matrix_shape(self):
        analyzer = FeatureInformativenessAnalyzer(
            target_generator=MultiTargetGenerator(
                horizons=[
                    HorizonConfig(name="short", horizon=20),
                    HorizonConfig(name="mid", horizon=60),
                ],
                target_types=[
                    TargetType(name="direction", kind="discrete"),
                    TargetType(name="return_magnitude", kind="continuous"),
                ],
            ),
            event_detector=None,
            rolling_mi=RollingMIConfig(window_size=300),
            bins=10,
        )
        df = _make_ohlcv_with_features(n=1000, pairs=["BTCUSDT", "ETHUSDT"])
        report = analyzer.analyze(df, ["feat_informative", "feat_noise"])
        matrix = report.score_matrix

        # 2 features as rows
        assert matrix.height == 2
        # 2 horizons x 2 targets = 4 pivot columns + 1 feature column
        assert matrix.width == 5

    def test_with_global_event_detection(self):
        analyzer = FeatureInformativenessAnalyzer(
            target_generator=MultiTargetGenerator(
                horizons=[HorizonConfig(name="short", horizon=30)],
                target_types=[TargetType(name="direction", kind="discrete")],
            ),
            event_detector=GlobalEventDetector(
                agreement_threshold=0.8,
                min_pairs=3,
            ),
            rolling_mi=RollingMIConfig(window_size=500),
            bins=10,
        )
        df = _make_ohlcv_with_features(n=1500, pairs=["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"])
        report = analyzer.analyze(df, ["feat_informative", "feat_noise"])

        assert report.global_events is not None
        assert "signal_type" in report.global_events.columns
        assert report.global_events.height > 0
        assert report.composite_scores.height == 2

    def test_disable_global_event_detection(self):
        analyzer = FeatureInformativenessAnalyzer(
            target_generator=MultiTargetGenerator(
                horizons=[HorizonConfig(name="short", horizon=30)],
                target_types=[TargetType(name="direction", kind="discrete")],
            ),
            event_detector=None,
            rolling_mi=RollingMIConfig(window_size=500),
            bins=10,
        )
        df = _make_ohlcv_with_features(n=1000, pairs=["BTCUSDT", "ETHUSDT"])
        report = analyzer.analyze(df, ["feat_informative"])
        assert report.global_events is None
