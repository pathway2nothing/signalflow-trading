"""Z-score based market-wide detector.

Detects timestamps where the cross-pair aggregate return is a statistical
outlier relative to its own rolling distribution. More robust than naive
agreement-based detection on correlated markets because it adapts to the
current volatility regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import polars as pl
from loguru import logger

from signalflow.core import RawDataView, Signals
from signalflow.core.decorators import sf_component
from signalflow.core.enums import SfComponentType, SignalCategory
from signalflow.detector.base import SignalDetector


@dataclass
@sf_component(name="market_wide/zscore")
class MarketZScoreDetector(SignalDetector):
    """Detects market-wide signals via z-score of aggregate cross-pair return.

    More robust than agreement-based detection on correlated markets because
    it adapts to the current volatility regime.

    Algorithm:
        1. Compute log-return per pair per timestamp.
        2. Compute cross-pair mean return at each timestamp.
        3. Compute rolling mean and std of the aggregate return over
           ``rolling_window`` bars.
        4. z_score = (agg_return - rolling_mean) / rolling_std
        5. Signal if |z_score| > ``z_threshold``.

    Attributes:
        z_threshold: Absolute z-score threshold for detection.
        rolling_window: Window size for rolling statistics.
        min_pairs: Minimum number of active pairs at a timestamp.
        return_window: Bars for return computation.
        signal_type_name: Signal type name for detected signals.

    Example:
        ```python
        from signalflow.detector import MarketZScoreDetector
        from signalflow.target.utils import mask_targets_by_signals

        # Detect market-wide z-score outliers
        detector = MarketZScoreDetector(z_threshold=3.0)
        signals = detector.run(raw_data_view)

        # Mask labels overlapping with detected signals
        labeled_df = mask_targets_by_signals(
            df=labeled_df,
            signals=signals,
            mask_signal_types={"aggregate_outlier"},
            horizon_bars=60,
        )
        ```

    Note:
        Returns one signal per timestamp (not per pair) since market-wide
        signals affect all pairs simultaneously. The signal has a synthetic
        "ALL" pair.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.DETECTOR

    signal_category: SignalCategory = SignalCategory.MARKET_WIDE
    allowed_signal_types: set[str] | None = field(default_factory=lambda: {"aggregate_outlier"})

    z_threshold: float = 3.0
    rolling_window: int = 100
    min_pairs: int = 5
    return_window: int = 1
    price_col: str = "close"
    signal_type_name: str = "aggregate_outlier"

    def __post_init__(self) -> None:
        self.allowed_signal_types = {self.signal_type_name}

    def preprocess(
        self,
        raw_data_view: RawDataView,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Preprocess raw data: compute log returns.

        Returns raw OHLCV with _ret column added.
        """
        df = super().preprocess(raw_data_view, context)

        df = df.sort([self.pair_col, self.ts_col]).with_columns(
            (pl.col(self.price_col) / pl.col(self.price_col).shift(self.return_window))
            .log()
            .over(self.pair_col)
            .alias("_ret")
        )

        return df

    def detect(
        self,
        features: pl.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> Signals:
        """Detect market-wide signals via z-score.

        Args:
            features: Multi-pair OHLCV DataFrame with _ret column.
            context: Additional context (unused).

        Returns:
            Signals with aggregate_outlier signal type for detected timestamps.
        """
        min_samples = self.rolling_window

        # Cross-pair mean return per timestamp
        agg_return = (
            features.filter(pl.col("_ret").is_not_null() & pl.col("_ret").is_finite())
            .group_by(self.ts_col)
            .agg(
                pl.col("_ret").mean().alias("_agg_return"),
                pl.col("_ret").count().alias("_n_pairs"),
            )
            .filter(pl.col("_n_pairs") >= self.min_pairs)
            .sort(self.ts_col)
        )

        # Rolling mean/std and z-score
        result = (
            agg_return.with_columns(
                [
                    pl.col("_agg_return")
                    .rolling_mean(window_size=self.rolling_window, min_samples=min_samples)
                    .alias("_rolling_mean"),
                    pl.col("_agg_return")
                    .rolling_std(window_size=self.rolling_window, min_samples=min_samples)
                    .alias("_rolling_std"),
                ]
            )
            .with_columns(
                pl.when(pl.col("_rolling_std") > 1e-12)
                .then((pl.col("_agg_return") - pl.col("_rolling_mean")) / pl.col("_rolling_std"))
                .otherwise(pl.lit(0.0))
                .alias("_z_score")
            )
            .filter(pl.col("_z_score").abs() > self.z_threshold)
        )

        n_signals = result.height
        logger.info(f"MarketZScoreDetector: detected {n_signals} timestamps")

        if n_signals == 0:
            return Signals(
                pl.DataFrame(
                    schema={
                        self.pair_col: pl.Utf8,
                        self.ts_col: pl.Datetime,
                        "signal_type": pl.Utf8,
                        "signal": pl.Int64,
                        "probability": pl.Float64,
                    }
                )
            )

        # Create signals with synthetic "ALL" pair for market-wide signals
        # Probability based on normalized z-score
        signals_df = result.select(
            [
                pl.lit("ALL").alias(self.pair_col),
                self.ts_col,
                pl.lit(self.signal_type_name).alias("signal_type"),
                pl.lit(1).alias("signal"),
                (pl.col("_z_score").abs() / self.z_threshold).clip(0.0, 1.0).alias("probability"),
            ]
        )

        return Signals(signals_df)


# Backward compatibility alias with legacy default signal_type_name
@dataclass
class ZScoreEventDetector(MarketZScoreDetector):
    """Backward compatibility alias for MarketZScoreDetector.

    .. deprecated::
        Use MarketZScoreDetector instead. This alias preserves the legacy
        "global_event" signal_type_name for backward compatibility.
    """

    signal_type_name: str = "global_event"
    allowed_signal_types: set[str] | None = field(default_factory=lambda: {"global_event"})
