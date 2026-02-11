"""Z-score based anomaly detector.

Detects anomalies on any feature using z-score method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from signalflow.core import RawDataView, Signals, sf_component
from signalflow.core.enums import SignalCategory
from signalflow.detector.base import SignalDetector


@dataclass
@sf_component(name="zscore_anomaly_detector")
class ZScoreAnomalyDetector(SignalDetector):
    """Z-score based anomaly detector on any feature.

    Detects anomalies when a feature value deviates significantly from its
    rolling mean, measured in standard deviations (z-score).

    Algorithm:
        1. Compute rolling mean of target_feature over rolling_window
        2. Compute rolling std of target_feature over rolling_window
        3. Calculate z-score: (value - rolling_mean) / rolling_std
        4. Signal if |z-score| > threshold

    Attributes:
        target_feature: Column name to analyze for anomalies.
        rolling_window: Window size for rolling mean/std calculation.
        threshold: Z-score threshold for anomaly detection.
        signal_high: Signal type when z-score > threshold.
        signal_low: Signal type when z-score < -threshold.

    Example:
        ```python
        from signalflow.detector import ZScoreAnomalyDetector
        from signalflow.feature import RsiExtractor

        # Detect anomalies on RSI
        detector = ZScoreAnomalyDetector(
            features=[RsiExtractor(period=14)],
            target_feature="RSI_14",
            threshold=3.0,
        )
        signals = detector.run(raw_data_view)

        # Detect anomalies on log returns
        detector = ZScoreAnomalyDetector(
            target_feature="close",  # Will compute on raw close prices
            threshold=4.0,
            signal_high="extreme_positive_anomaly",
            signal_low="extreme_negative_anomaly",
        )
        ```

    Note:
        This detector overrides preprocess() to add helper columns
        (_target_rol_mean, _target_rol_std) for z-score calculation.
    """

    signal_category: SignalCategory = SignalCategory.ANOMALY
    allowed_signal_types: set[str] | None = field(default_factory=lambda: {"positive_anomaly", "negative_anomaly"})

    # Target feature to analyze
    target_feature: str = "close"

    # Z-score parameters
    rolling_window: int = 1440
    threshold: float = 4.0

    # Signal type names
    signal_high: str = "positive_anomaly"  # value > mean + threshold * std
    signal_low: str = "negative_anomaly"  # value < mean - threshold * std

    def __post_init__(self) -> None:
        # Update allowed_signal_types based on configured signal names
        self.allowed_signal_types = {self.signal_high, self.signal_low}

    def preprocess(
        self,
        raw_data_view: RawDataView,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Run features + compute rolling mean/std for target_feature.

        Args:
            raw_data_view: View to raw market data.
            context: Additional context (unused).

        Returns:
            DataFrame with original columns plus _target_rol_mean, _target_rol_std.
        """
        # 1. Base preprocessing (OHLCV + features)
        df = super().preprocess(raw_data_view, context)

        # 2. Compute helper columns for z-score
        min_samples = max(2, self.rolling_window // 4)
        df = df.with_columns(
            [
                pl.col(self.target_feature)
                .rolling_mean(window_size=self.rolling_window, min_samples=min_samples)
                .over(self.pair_col)
                .alias("_target_rol_mean"),
                pl.col(self.target_feature)
                .rolling_std(window_size=self.rolling_window, min_samples=min_samples)
                .over(self.pair_col)
                .alias("_target_rol_std"),
            ]
        )

        return df

    def detect(
        self,
        features: pl.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> Signals:
        """Detect anomalies using z-score method.

        Args:
            features: Preprocessed DataFrame with _target_rol_mean, _target_rol_std.
            context: Additional context (unused).

        Returns:
            Signals with positive_anomaly/negative_anomaly signal types.
        """
        target = pl.col(self.target_feature)
        mean = pl.col("_target_rol_mean")
        std = pl.col("_target_rol_std")

        # Z-score calculation
        z_score = (target - mean) / std

        # Classify anomalies
        is_high = (std > 0) & (z_score > self.threshold)
        is_low = (std > 0) & (z_score < -self.threshold)

        signal_type_expr = (
            pl.when(is_high)
            .then(pl.lit(self.signal_high))
            .when(is_low)
            .then(pl.lit(self.signal_low))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias("signal_type")
        )

        # Probability: how far beyond threshold (clipped to [0, 1])
        probability_expr = (
            pl.when(is_high | is_low)
            .then((z_score.abs() / self.threshold).clip(0.0, 1.0))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("probability")
        )

        signals_df = (
            features.with_columns([signal_type_expr, probability_expr])
            .filter(pl.col("signal_type").is_not_null())
            .select(
                [
                    self.pair_col,
                    self.ts_col,
                    "signal_type",
                    pl.lit(1).alias("signal"),
                    "probability",
                ]
            )
        )

        return Signals(signals_df)
