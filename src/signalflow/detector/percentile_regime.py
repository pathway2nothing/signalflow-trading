"""Percentile-based regime detector.

Detects regime shifts on any feature using rolling percentile method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from signalflow.core import RawDataView, Signals, sf_component
from signalflow.core.enums import SignalCategory
from signalflow.detector.base import SignalDetector


@dataclass
@sf_component(name="percentile_regime_detector")
class PercentileRegimeDetector(SignalDetector):
    """Percentile-based regime detector on any feature.

    Classifies the current regime by computing the rolling percentile of
    a target feature within a lookback window.

    Algorithm:
        1. For each bar, compute percentile of target_feature within lookback_window
        2. If percentile > upper_quantile -> signal_high
        3. If percentile < lower_quantile -> signal_low
        4. Otherwise -> no signal

    Attributes:
        target_feature: Column name to analyze for regime.
        lookback_window: Window size for percentile calculation.
        upper_quantile: Upper threshold (signal if percentile > this).
        lower_quantile: Lower threshold (signal if percentile < this).
        signal_high: Signal type when percentile > upper_quantile.
        signal_low: Signal type when percentile < lower_quantile.

    Example:
        ```python
        from signalflow.detector import PercentileRegimeDetector
        from signalflow.feature import FeaturePipeline, RealizedVolExtractor

        # Volatility regime detection
        detector = PercentileRegimeDetector(
            features_pipe=FeaturePipeline([RealizedVolExtractor()]),
            target_feature="_realized_vol",
            upper_quantile=0.67,
            lower_quantile=0.33,
            signal_high="vol_high",
            signal_low="vol_low",
        )
        signals = detector.run(raw_data_view)
        ```

    Note:
        Uses numpy for percentile calculation per group since Polars
        doesn't have native rolling percentile.
    """

    signal_category: SignalCategory = SignalCategory.VOLATILITY
    allowed_signal_types: set[str] | None = field(default_factory=lambda: {"regime_high", "regime_low"})

    # Target feature to analyze
    target_feature: str = "_realized_vol"

    # Percentile parameters
    lookback_window: int = 1440
    upper_quantile: float = 0.67
    lower_quantile: float = 0.33

    # Signal type names
    signal_high: str = "regime_high"
    signal_low: str = "regime_low"

    def __post_init__(self) -> None:
        if not 0 < self.lower_quantile < self.upper_quantile < 1:
            raise ValueError(
                f"Quantiles must satisfy 0 < lower < upper < 1, "
                f"got lower={self.lower_quantile}, upper={self.upper_quantile}"
            )
        # Update allowed_signal_types based on configured signal names
        self.allowed_signal_types = {self.signal_high, self.signal_low}

    def detect(
        self,
        features: pl.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> Signals:
        """Detect regime using rolling percentile method.

        Args:
            features: Preprocessed DataFrame with target_feature column.
            context: Additional context (unused).

        Returns:
            Signals with regime_high/regime_low signal types.
        """
        results = []

        for pair_name, group in features.group_by(self.pair_col, maintain_order=True):
            arr = group[self.target_feature].to_numpy().astype(np.float64)
            n = len(arr)

            signal_types: list[str | None] = [None] * n
            probabilities: list[float | None] = [None] * n

            for t in range(n):
                if np.isnan(arr[t]):
                    continue

                # Get lookback window
                lb_start = max(0, t - self.lookback_window + 1)
                window = arr[lb_start : t + 1]
                valid = window[~np.isnan(window)]

                if len(valid) < 2:
                    continue

                # Compute percentile (fraction of values <= current)
                percentile = float(np.mean(valid <= arr[t]))

                if percentile > self.upper_quantile:
                    signal_types[t] = self.signal_high
                    probabilities[t] = percentile
                elif percentile < self.lower_quantile:
                    signal_types[t] = self.signal_low
                    probabilities[t] = 1.0 - percentile

            group = group.with_columns(
                [
                    pl.Series(name="signal_type", values=signal_types, dtype=pl.Utf8),
                    pl.Series(name="probability", values=probabilities, dtype=pl.Float64),
                ]
            )
            results.append(group)

        if not results:
            return Signals(pl.DataFrame())

        combined = pl.concat(results, how="vertical_relaxed")

        signals_df = combined.filter(pl.col("signal_type").is_not_null()).select(
            [
                self.pair_col,
                self.ts_col,
                "signal_type",
                pl.lit(1).alias("signal"),
                "probability",
            ]
        )

        return Signals(signals_df)
