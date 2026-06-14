"""
Volatility regime labeler.

Labels bars based on forward realized volatility percentile within
a rolling lookback window.

Implementation uses pure Polars expressions for performance.
"""


from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.enums import SignalCategory
from signalflow.target._soft_helpers import percentile_tercile_soft
from signalflow.target.base import register_target
from signalflow.target.labeler import Labeler


@dataclass
@register_target("volatility_regime")
class VolatilityRegimeLabeler(Labeler):
    """
    Label bars by forward realized volatility regime.

    Algorithm:
        1. Compute log returns: ``ln(close[t] / close[t-1])``
        2. Forward realized volatility: ``std(log_returns[t+1 : t+horizon+1])``
           computed using reverse-shifted rolling std.
        3. Rolling percentile of realized vol over ``lookback_window``.
        4. If vol > ``upper_quantile`` percentile -> ``"high_volatility"``
        5. If vol < ``lower_quantile`` percentile -> ``"low_volatility"``
        6. Otherwise -> ``null`` (Polars null)

    Implementation:
        Uses pure Polars expressions instead of numpy loops for better
        performance and memory efficiency.
    """

    signal_category: SignalCategory = SignalCategory.VOLATILITY

    soft_classes: ClassVar[tuple[str, ...]] = ("low_volatility", "mid_volatility", "high_volatility")
    softness_k: float = 20.0

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
        """Compute volatility regime labels for a single pair."""
        if group_df.height == 0:
            return group_df

        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")


        df = group_df.with_columns((pl.col(self.price_col) / pl.col(self.price_col).shift(1)).log().alias("_log_ret"))


        df = df.with_columns(
            pl.col("_log_ret")
            .shift(-1)
            .rolling_std(window_size=self.horizon, min_samples=2)
            .shift(-(self.horizon - 1))
            .alias("_realized_vol")
        )


        df = df.with_columns(
            self._rolling_percentile_expr("_realized_vol", self.lookback_window).alias("_vol_percentile")
        )


        label_expr = (
            pl.when(pl.col("_vol_percentile").is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(pl.col("_vol_percentile") > self.upper_quantile)
            .then(pl.lit("high_volatility"))
            .when(pl.col("_vol_percentile") < self.lower_quantile)
            .then(pl.lit("low_volatility"))
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


        df = df.drop(["_log_ret", "_realized_vol", "_vol_percentile"])

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Soft tercile probabilities from the rolling vol percentile."""
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        df = group_df.with_columns((pl.col(self.price_col) / pl.col(self.price_col).shift(1)).log().alias("_log_ret"))
        df = df.with_columns(
            pl.col("_log_ret")
            .shift(-1)
            .rolling_std(window_size=self.horizon, min_samples=2)
            .shift(-(self.horizon - 1))
            .alias("_realized_vol")
        )
        df = df.with_columns(
            self._rolling_percentile_expr("_realized_vol", self.lookback_window).alias("_vol_percentile")
        )

        p_low, p_mid, p_high = percentile_tercile_soft(
            pl.col("_vol_percentile"),
            lower_q=self.lower_quantile,
            upper_q=self.upper_quantile,
            k=self.softness_k,
        )
        df = df.with_columns(
            p_low.alias(f"{self.soft_col_prefix}low_volatility"),
            p_mid.alias(f"{self.soft_col_prefix}mid_volatility"),
            p_high.alias(f"{self.soft_col_prefix}high_volatility"),
        )
        df = df.drop(["_log_ret", "_realized_vol", "_vol_percentile"])
        return df

    def _rolling_percentile_expr(self, col_name: str, window: int) -> pl.Expr:
        """Compute rolling percentile using Polars expressions."""
        col = pl.col(col_name)


        return pl.struct([col.alias("val"), pl.int_range(pl.len()).alias("idx")]).map_batches(
            lambda s: self._compute_percentile_series(s, window),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def _compute_percentile_series(s: pl.Series, window: int) -> pl.Series:
        """Compute rolling percentile for a series of structs."""
        df = s.struct.unnest()
        vals = df["val"].to_numpy()
        n = len(vals)
        result: list[float | None] = [None] * n

        import numpy as np

        for i in range(n):
            if np.isnan(vals[i]) if vals[i] is not None else True:
                continue

            start = max(0, i - window + 1)
            window_vals = vals[start : i + 1]


            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) < 2:
                continue


            result[i] = float(np.mean(valid <= vals[i]))

        return pl.Series(result, dtype=pl.Float64)
