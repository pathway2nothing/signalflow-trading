"""Directional (long/short) mean-reversion labeler."""

from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.enums import SignalCategory
from signalflow.target._soft_helpers import sigmoid_expr
from signalflow.target.base import register_target
from signalflow.target.labeler import Labeler


@dataclass
@register_target("directional_mean_reversion")
class DirectionalMeanReversionLabeler(Labeler):
    """
    Three-class long/short/none mean-reversion label.

    Algorithm:
        1. Rolling µ, σ over ``z_window``; ``z_now = (close - µ) / σ`` and
           ``z_fwd = (close[t+horizon] - µ) / σ`` (same baseline).
        2. If ``z_now < -stretch_threshold`` (oversold) and ``z_fwd`` is at
           or above ``-revert_threshold`` -> ``"revert_long"`` (price moved
           back up toward / through the mean).
        3. If ``z_now > stretch_threshold`` (overbought) and ``z_fwd`` is at
           or below ``revert_threshold`` -> ``"revert_short"`` (price moved
           back down).
        4. Otherwise -> ``"no_revert"``. Bars with no future data (last
           ``horizon`` rows) are null.

    Use:
        Train one model that emits both long and short revert probabilities;
        downstream entry rules can route ``revert_long`` signals to long
        sleeves and ``revert_short`` to short sleeves. Pairs naturally with
        :class:`MeanReversionMagnitudeLabeler` (magnitude regressor) and
        :class:`MeanReversionEventLabeler` (direction-agnostic event flag).
    """

    signal_category: SignalCategory = SignalCategory.PRICE_STRUCTURE

    soft_classes: ClassVar[tuple[str, ...]] = (
        "revert_short",
        "no_revert",
        "revert_long",
    )
    positive_classes: ClassVar[tuple[str, ...]] = ("revert_long", "revert_short")
    softness_k: float = 5.0

    price_col: str = "close"
    horizon: int = 240
    z_window: int = 240
    stretch_threshold: float = 2.0
    revert_threshold: float = 0.5

    meta_columns: tuple[str, ...] = ("z_now", "z_fwd")

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.z_window <= 1:
            raise ValueError("z_window must be > 1")
        if self.stretch_threshold <= 0 or self.revert_threshold <= 0:
            raise ValueError("thresholds must be > 0")
        if self.revert_threshold >= self.stretch_threshold:
            raise ValueError("revert_threshold must be < stretch_threshold")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def _with_z(self, group_df: pl.DataFrame) -> pl.DataFrame:
        price = pl.col(self.price_col)
        df = group_df.with_columns(
            price.rolling_mean(self.z_window, min_samples=2).alias("_mu"),
            price.rolling_std(self.z_window, min_samples=2).alias("_sd"),
        )
        df = df.with_columns(
            pl.when(pl.col("_sd") > 0)
            .then((price - pl.col("_mu")) / pl.col("_sd"))
            .otherwise(pl.lit(None))
            .alias("_z_now"),
            pl.when(pl.col("_sd") > 0)
            .then((price.shift(-self.horizon) - pl.col("_mu")) / pl.col("_sd"))
            .otherwise(pl.lit(None))
            .alias("_z_fwd"),
        )
        return df

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        df = self._with_z(group_df)

        oversold = pl.col("_z_now") < pl.lit(-self.stretch_threshold)
        overbought = pl.col("_z_now") > pl.lit(self.stretch_threshold)

        long_reverted = oversold & (pl.col("_z_fwd") >= pl.lit(-self.revert_threshold))
        short_reverted = overbought & (pl.col("_z_fwd") <= pl.lit(self.revert_threshold))

        label_expr = (
            pl.when(pl.col("_z_now").is_null() | pl.col("_z_fwd").is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(long_reverted)
            .then(pl.lit("revert_long"))
            .when(short_reverted)
            .then(pl.lit("revert_short"))
            .otherwise(pl.lit("no_revert"))
            .alias(self.out_col)
        )
        df = df.with_columns(label_expr)

        if self.include_meta:
            df = df.with_columns(
                pl.col("_z_now").alias("z_now"),
                pl.col("_z_fwd").alias("z_fwd"),
            )

        df = df.drop(["_mu", "_sd", "_z_now", "_z_fwd"])

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Soft triple ``(p_revert_short, p_no_revert, p_revert_long)``."""
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        df = self._with_z(group_df)
        null_mask = pl.col("_z_now").is_null() | pl.col("_z_fwd").is_null()

        p_oversold = sigmoid_expr(-pl.col("_z_now") - pl.lit(self.stretch_threshold), self.softness_k)
        p_overbought = sigmoid_expr(pl.col("_z_now") - pl.lit(self.stretch_threshold), self.softness_k)
        p_long_given = sigmoid_expr(pl.col("_z_fwd") + pl.lit(self.revert_threshold), self.softness_k)
        p_short_given = sigmoid_expr(pl.lit(self.revert_threshold) - pl.col("_z_fwd"), self.softness_k)

        p_long_raw = p_oversold * p_long_given
        p_short_raw = p_overbought * p_short_given
        p_none_raw = (pl.lit(1.0) - p_long_raw - p_short_raw).clip(lower_bound=0.0)
        total = p_long_raw + p_short_raw + p_none_raw
        safe = pl.when(total > 0).then(total).otherwise(pl.lit(1.0))

        df = df.with_columns(
            pl.when(null_mask)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(p_short_raw / safe)
            .alias(f"{self.soft_col_prefix}revert_short"),
            pl.when(null_mask)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(p_none_raw / safe)
            .alias(f"{self.soft_col_prefix}no_revert"),
            pl.when(null_mask)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(p_long_raw / safe)
            .alias(f"{self.soft_col_prefix}revert_long"),
        )
        df = df.drop(["_mu", "_sd", "_z_now", "_z_fwd"])
        return df
