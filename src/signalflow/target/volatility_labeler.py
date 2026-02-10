"""Volatility regime labeler.

Labels bars based on forward realized volatility percentile within
a rolling lookback window.

Implementation uses pure Polars expressions for performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from signalflow.core import sf_component
from signalflow.core.enums import SignalCategory
from signalflow.target.base import Labeler


@dataclass
@sf_component(name="volatility_regime")
class VolatilityRegimeLabeler(Labeler):
    """Label bars by forward realized volatility regime.

    Algorithm:
        1. Compute log returns: ``ln(close[t] / close[t-1])``
        2. Forward realized volatility: ``std(log_returns[t+1 : t+horizon+1])``
           computed using reverse-shifted rolling std.
        3. Rolling percentile of realized vol over ``lookback_window``.
        4. If vol > ``upper_quantile`` percentile -> ``"vol_high"``
        5. If vol < ``lower_quantile`` percentile -> ``"vol_low"``
        6. Otherwise -> ``null`` (Polars null)

    Implementation:
        Uses pure Polars expressions instead of numpy loops for better
        performance and memory efficiency.

    Attributes:
        price_col: Price column name. Default: ``"close"``.
        horizon: Number of forward bars for realized vol. Default: ``60``.
        upper_quantile: Upper percentile threshold (0-1). Default: ``0.67``.
        lower_quantile: Lower percentile threshold (0-1). Default: ``0.33``.
        lookback_window: Rolling window for percentile calc. Default: ``1440``.

    Example:
        ```python
        from signalflow.target.volatility_labeler import VolatilityRegimeLabeler

        labeler = VolatilityRegimeLabeler(
            horizon=60,
            upper_quantile=0.67,
            lower_quantile=0.33,
            mask_to_signals=False,
        )
        result = labeler.compute(ohlcv_df)
        ```
    """

    signal_category: SignalCategory = SignalCategory.VOLATILITY

    price_col: str = "close"
    horizon: int = 60
    upper_quantile: float = 0.67
    lower_quantile: float = 0.33
    lookback_window: int = 1440

    meta_columns: tuple[str, ...] = ("realized_vol", "vol_percentile")

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if not (0.0 < self.lower_quantile < self.upper_quantile < 1.0):
            raise ValueError(
                "Require 0 < lower_quantile < upper_quantile < 1, "
                f"got lower_quantile={self.lower_quantile}, upper_quantile={self.upper_quantile}"
            )

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Compute volatility regime labels for a single pair.

        Args:
            group_df: Single pair data sorted by timestamp.
            data_context: Optional additional context.

        Returns:
            DataFrame with same row count, plus label and optional meta columns.
        """
        if group_df.height == 0:
            return group_df

        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        # Step 1: Log returns
        df = group_df.with_columns(
            (pl.col(self.price_col) / pl.col(self.price_col).shift(1)).log().alias("_log_ret")
        )

        # Step 2: Forward realized volatility
        # To compute std of log_returns[t+1 : t+horizon+1], we:
        # - Shift log_ret by -1 to start from next bar
        # - Apply rolling_std with window=horizon
        # - The result at position t+horizon-1 contains std of [t, t+horizon)
        # - Shift back by -(horizon-1) to align with position t
        df = df.with_columns(
            pl.col("_log_ret")
            .shift(-1)
            .rolling_std(window_size=self.horizon, min_samples=2)
            .shift(-(self.horizon - 1))
            .alias("_realized_vol")
        )

        # Step 3: Rolling percentile using rank-based approach
        # For each bar, compute what fraction of values in the lookback window
        # are <= current value. This is equivalent to percentile.
        #
        # Using rolling_map with a custom expression to compute percentile:
        # percentile = count(x <= current) / count(valid)
        df = df.with_columns(
            self._rolling_percentile_expr("_realized_vol", self.lookback_window).alias("_vol_percentile")
        )

        # Step 4-5: Assign labels based on percentile thresholds
        label_expr = (
            pl.when(pl.col("_vol_percentile").is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(pl.col("_vol_percentile") > self.upper_quantile)
            .then(pl.lit("vol_high"))
            .when(pl.col("_vol_percentile") < self.lower_quantile)
            .then(pl.lit("vol_low"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias(self.out_col)
        )

        df = df.with_columns(label_expr)

        if self.include_meta:
            df = df.with_columns(
                [
                    pl.col("_realized_vol").alias("realized_vol"),
                    pl.col("_vol_percentile").alias("vol_percentile"),
                ]
            )

        # Clean up temporary columns
        df = df.drop(["_log_ret", "_realized_vol", "_vol_percentile"])

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df

    def _rolling_percentile_expr(self, col_name: str, window: int) -> pl.Expr:
        """Compute rolling percentile using Polars expressions.

        For each row, computes the fraction of values in the lookback window
        that are less than or equal to the current value.

        Args:
            col_name: Column to compute percentile for.
            window: Lookback window size.

        Returns:
            Polars expression computing rolling percentile.
        """
        col = pl.col(col_name)

        # Create a struct with current value and row index
        # Then use rolling_map to compute percentile within each window
        return (
            pl.struct([col.alias("val"), pl.int_range(pl.len()).alias("idx")])
            .map_batches(
                lambda s: self._compute_percentile_series(s, window),
                return_dtype=pl.Float64,
            )
        )

    @staticmethod
    def _compute_percentile_series(s: pl.Series, window: int) -> pl.Series:
        """Compute rolling percentile for a series of structs.

        Args:
            s: Series of structs with 'val' and 'idx' fields.
            window: Lookback window size.

        Returns:
            Series of percentile values.
        """
        df = s.struct.unnest()
        vals = df["val"].to_numpy()
        n = len(vals)
        result = [None] * n

        import numpy as np

        for i in range(n):
            if np.isnan(vals[i]) if vals[i] is not None else True:
                continue

            start = max(0, i - window + 1)
            window_vals = vals[start : i + 1]

            # Filter out NaN/None values
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) < 2:
                continue

            # Percentile = fraction of values <= current
            result[i] = float(np.mean(valid <= vals[i]))

        return pl.Series(result, dtype=pl.Float64)
