"""Agreement-based global event detector.

Detects timestamps where an unusually high fraction of trading pairs
move in the same direction simultaneously, signaling an exogenous
macro event (interest rate decision, regulatory news, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from loguru import logger

from signalflow.core.decorators import sf_component
from signalflow.detector.event.base import EventDetectorBase


@dataclass
@sf_component(name="event_detector/agreement")
class GlobalEventDetector(EventDetectorBase):
    """Detects timestamps where cross-pair return agreement is abnormally high.

    Algorithm:
        1. Compute log-return for each pair at each timestamp.
        2. At each timestamp, compute the fraction of pairs with
           same-sign return (majority sign).
        3. If fraction >= ``agreement_threshold``, mark as global event.

    Attributes:
        agreement_threshold: Fraction of pairs that must agree for event detection.
        min_pairs: Minimum number of active pairs at a timestamp.
        return_window: Bars for return computation.
    """

    agreement_threshold: float = 0.8
    min_pairs: int = 5
    return_window: int = 1

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detect global event timestamps by agreement threshold.

        Args:
            df: Multi-pair OHLCV DataFrame sorted by (pair, timestamp).

        Returns:
            DataFrame with columns: (timestamp, _is_global_event).
            One row per unique timestamp.
        """
        self._validate(df)

        returns_df = df.sort([self.pair_col, self.ts_col]).with_columns(
            (pl.col(self.price_col) / pl.col(self.price_col).shift(self.return_window))
            .log()
            .over(self.pair_col)
            .alias("_ret")
        )

        agreement = (
            returns_df.filter(pl.col("_ret").is_not_null() & pl.col("_ret").is_finite())
            .group_by(self.ts_col)
            .agg(
                pl.col("_ret").count().alias("_n_pairs"),
                (pl.col("_ret") > 0).sum().alias("_n_positive"),
                (pl.col("_ret") < 0).sum().alias("_n_negative"),
            )
            .filter(pl.col("_n_pairs") >= self.min_pairs)
            .with_columns(
                (
                    pl.max_horizontal("_n_positive", "_n_negative").cast(pl.Float64)
                    / pl.col("_n_pairs").cast(pl.Float64)
                ).alias("_agreement")
            )
            .with_columns((pl.col("_agreement") >= self.agreement_threshold).alias("_is_global_event"))
            .select([self.ts_col, "_is_global_event"])
        )

        n_events = agreement.filter(pl.col("_is_global_event")).height
        logger.info(f"GlobalEventDetector: detected {n_events} event timestamps")

        return agreement
