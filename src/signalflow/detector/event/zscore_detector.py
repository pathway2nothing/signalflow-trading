"""Z-score based global event detector.

Detects timestamps where the cross-pair aggregate return is a statistical
outlier relative to its own rolling distribution. More robust than naive
agreement-based detection on correlated markets because it adapts to the
current volatility regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from loguru import logger

from signalflow.core.decorators import sf_component
from signalflow.detector.event.base import EventDetectorBase


@dataclass
@sf_component(name="event_detector/zscore")
class ZScoreEventDetector(EventDetectorBase):
    """Detects global events via z-score of aggregate cross-pair return.

    Algorithm:
        1. Compute log-return per pair per timestamp.
        2. Compute cross-pair mean return at each timestamp.
        3. Compute rolling mean and std of the aggregate return over
           ``rolling_window`` bars.
        4. z_score = (agg_return - rolling_mean) / rolling_std
        5. Event if |z_score| > ``z_threshold``.

    Attributes:
        z_threshold: Absolute z-score threshold for event detection.
        rolling_window: Window size for rolling statistics.
        min_pairs: Minimum number of active pairs at a timestamp.
        return_window: Bars for return computation.
    """

    z_threshold: float = 3.0
    rolling_window: int = 100
    min_pairs: int = 5
    return_window: int = 1

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detect global event timestamps via z-score.

        Args:
            df: Multi-pair OHLCV DataFrame sorted by (pair, timestamp).

        Returns:
            DataFrame with columns: (timestamp, _is_global_event).
        """
        self._validate(df)

        # Step 1: log-returns per pair
        returns_df = df.sort([self.pair_col, self.ts_col]).with_columns(
            (pl.col(self.price_col) / pl.col(self.price_col).shift(self.return_window))
            .log()
            .over(self.pair_col)
            .alias("_ret")
        )

        # Step 2: cross-pair mean return per timestamp
        min_periods = self.rolling_window
        agg_return = (
            returns_df.filter(pl.col("_ret").is_not_null() & pl.col("_ret").is_finite())
            .group_by(self.ts_col)
            .agg(
                pl.col("_ret").mean().alias("_agg_return"),
                pl.col("_ret").count().alias("_n_pairs"),
            )
            .filter(pl.col("_n_pairs") >= self.min_pairs)
            .sort(self.ts_col)
        )

        # Steps 3-4: rolling mean/std and z-score
        result = (
            agg_return.with_columns(
                [
                    pl.col("_agg_return")
                    .rolling_mean(window_size=self.rolling_window, min_periods=min_periods)
                    .alias("_rolling_mean"),
                    pl.col("_agg_return")
                    .rolling_std(window_size=self.rolling_window, min_periods=min_periods)
                    .alias("_rolling_std"),
                ]
            )
            .with_columns(
                pl.when(pl.col("_rolling_std") > 1e-12)
                .then((pl.col("_agg_return") - pl.col("_rolling_mean")) / pl.col("_rolling_std"))
                .otherwise(pl.lit(0.0))
                .alias("_z_score")
            )
            .with_columns((pl.col("_z_score").abs() > self.z_threshold).alias("_is_global_event"))
            .select([self.ts_col, "_is_global_event"])
        )

        n_events = result.filter(pl.col("_is_global_event")).height
        logger.info(f"ZScoreEventDetector: detected {n_events} event timestamps")

        return result
