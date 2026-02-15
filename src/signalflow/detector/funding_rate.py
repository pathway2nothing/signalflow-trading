"""Funding rate transition detector.

Detects long entry opportunities when funding rate flips from sustained
positive (longs paying shorts) to negative (shorts paying longs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import polars as pl

from signalflow.core import (
    RawDataView,
    Signals,
    SfComponentType,
    SignalCategory,
    RawDataType,
    sf_component,
)
from signalflow.detector.base import SignalDetector


@dataclass
@sf_component(name="funding/rate_transition")
class FundingRateDetector(SignalDetector):
    """Detects long entries when funding rate transitions from positive to negative.

    Strategy logic:
        1. Extract non-null funding rate observations per pair
        2. Track the gap between consecutive non-positive readings
        3. When funding turns negative AND the previous non-positive reading
           was >= ``min_positive_hours`` ago (meaning all interim readings
           were positive), generate a RISE signal

    This pattern suggests overleveraged longs are exiting, potentially
    creating upward price pressure as shorts cover.

    Attributes:
        min_positive_hours: Minimum hours of sustained positive funding
            before a negative transition triggers a signal. Default: 24.
        funding_col: Column name for funding rate data.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.DETECTOR
    signal_category: SignalCategory = SignalCategory.PRICE_DIRECTION
    raw_data_type: RawDataType | str = RawDataType.PERPETUAL

    min_positive_hours: int = 24
    funding_col: str = "funding_rate"

    allowed_signal_types: set[str] | None = field(default_factory=lambda: {"rise"})

    def preprocess(
        self,
        raw_data_view: RawDataView,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Extract perpetual data with funding rates."""
        key = self.raw_data_type.value if hasattr(self.raw_data_type, "value") else str(self.raw_data_type)
        df = raw_data_view.to_polars(key)
        return df.sort([self.pair_col, self.ts_col])

    def detect(
        self,
        features: pl.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> Signals:
        """Detect funding rate transition signals.

        Algorithm:
            1. Filter to rows where ``funding_rate`` is not null
            2. For each non-positive reading, record its timestamp
            3. Shift by 1 and forward-fill to get the *previous*
               non-positive timestamp at each row
            4. At each negative reading, compute hours since that
               previous non-positive reading
            5. If hours >= ``min_positive_hours``, all interim readings
               were positive for long enough → signal
        """
        pair_col = self.pair_col
        ts_col = self.ts_col
        fr_col = self.funding_col

        # 1. Extract rows with actual funding rate observations
        funding = (
            features.filter(pl.col(fr_col).is_not_null()).select([pair_col, ts_col, fr_col]).sort([pair_col, ts_col])
        )

        if funding.is_empty():
            return Signals(
                pl.DataFrame(
                    schema={
                        pair_col: pl.Utf8,
                        ts_col: pl.Datetime("us"),
                        "signal_type": pl.Utf8,
                        "signal": pl.Int32,
                    }
                )
            )

        # 2. For non-positive readings, record their timestamp; null otherwise
        funding = funding.with_columns(
            pl.when(pl.col(fr_col) <= 0)
            .then(pl.col(ts_col))
            .otherwise(pl.lit(None, dtype=pl.Datetime("us")))
            .alias("_non_pos_ts"),
        )

        # 3. Shift by 1 to exclude the current row, then forward-fill
        #    within each pair to propagate the last *previous* non-positive ts
        funding = funding.with_columns(
            pl.col("_non_pos_ts").shift(1).forward_fill().over(pair_col).alias("_last_prev_non_pos_ts"),
        )

        # 4. Compute hours since previous non-positive reading
        funding = funding.with_columns(
            ((pl.col(ts_col) - pl.col("_last_prev_non_pos_ts")).dt.total_hours()).alias("_hours_gap"),
        )

        # 5. Signal: current is negative AND gap >= min_positive_hours
        signal_mask = (pl.col(fr_col) < 0) & (pl.col("_hours_gap") >= self.min_positive_hours)

        signals_df = funding.filter(signal_mask).select(
            [
                pair_col,
                ts_col,
                pl.lit("rise").alias("signal_type"),
                pl.lit(1).alias("signal"),
            ]
        )

        return Signals(signals_df)
