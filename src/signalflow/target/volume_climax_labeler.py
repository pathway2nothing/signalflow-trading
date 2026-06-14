"""
Volume climax labeler.

Labels bars based on the *maximum* forward volume within a horizon
relative to a trailing volume SMA - captures climax/exhaustion events
rather than the average forward volume.

Complements :class:`VolumeRegimeLabeler` (which uses forward *mean*):
this labeler is more sensitive to single high-volume bars within the
window and is per-pair idiosyncratic (per iter25 EDA: cross-pair κ ≈ 0.24
vs 0.97 for volume_regime).

Implementation uses pure Polars expressions for performance.
"""


from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.enums import SignalCategory
from signalflow.target._soft_helpers import signed_tercile_soft
from signalflow.target.base import register_target
from signalflow.target.labeler import Labeler


@dataclass
@register_target("volume_climax")
class VolumeClimaxLabeler(Labeler):
    """Label bars by forward max-volume vs trailing SMA ratio."""

    signal_category: SignalCategory = SignalCategory.VOLUME_LIQUIDITY

    soft_classes: ClassVar[tuple[str, ...]] = ("calm_vol", "normal_vol", "climax")

    volume_col: str = "volume"
    horizon: int = 240
    vol_sma_window: int = 1440
    climax_threshold: float = 5.0
    calm_threshold: float = 1.5

    meta_columns: tuple[str, ...] = ("volume_climax_ratio",)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.vol_sma_window <= 0:
            raise ValueError("vol_sma_window must be > 0")
        if self.calm_threshold >= self.climax_threshold:
            raise ValueError(
                f"calm_threshold ({self.calm_threshold}) must be < climax_threshold ({self.climax_threshold})"
            )

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        if group_df.height == 0:
            return group_df
        if self.volume_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.volume_col}'")

        vol = pl.col(self.volume_col)

        df = group_df.with_columns(
            vol.rolling_mean(window_size=self.vol_sma_window, min_samples=1).alias("_trailing_sma"),
        )

        df = df.with_columns(
            vol.shift(-1)
            .rolling_max(window_size=self.horizon, min_samples=1)
            .shift(-(self.horizon - 1))
            .alias("_forward_max"),
        )

        df = df.with_columns(
            pl.when(pl.col("_trailing_sma") > 0)
            .then(pl.col("_forward_max") / pl.col("_trailing_sma"))
            .otherwise(pl.lit(None))
            .alias("_climax_ratio"),
        )

        label_expr = (
            pl.when(pl.col("_climax_ratio").is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(pl.col("_climax_ratio") > self.climax_threshold)
            .then(pl.lit("climax"))
            .when(pl.col("_climax_ratio") < self.calm_threshold)
            .then(pl.lit("calm_vol"))
            .otherwise(pl.lit("normal_vol"))
            .alias(self.out_col)
        )

        df = df.with_columns(label_expr)

        if self.include_meta:
            df = df.with_columns(pl.col("_climax_ratio").alias("volume_climax_ratio"))

        df = df.drop(["_trailing_sma", "_forward_max", "_climax_ratio"])

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Soft triple ``(p_calm_vol, p_normal_vol, p_climax)`` from the forward max/SMA ratio."""
        if group_df.height == 0:
            return group_df
        if self.volume_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.volume_col}'")

        vol = pl.col(self.volume_col)
        df = group_df.with_columns(
            vol.rolling_mean(window_size=self.vol_sma_window, min_samples=1).alias("_trailing_sma"),
        )
        df = df.with_columns(
            vol.shift(-1)
            .rolling_max(window_size=self.horizon, min_samples=1)
            .shift(-(self.horizon - 1))
            .alias("_forward_max"),
        )
        centre = 0.5 * (self.calm_threshold + self.climax_threshold)
        df = df.with_columns(
            pl.when(pl.col("_trailing_sma") > 0)
            .then(pl.col("_forward_max") / pl.col("_trailing_sma") - centre)
            .otherwise(pl.lit(None))
            .alias("_centred_ratio"),
        )

        half_span = 0.5 * (self.climax_threshold - self.calm_threshold)
        p_calm, p_norm, p_climax = signed_tercile_soft(
            pl.col("_centred_ratio"),
            neg_threshold=half_span,
            pos_threshold=half_span,
            k=self.softness_k,
        )
        df = df.with_columns(
            p_calm.alias(f"{self.soft_col_prefix}calm_vol"),
            p_norm.alias(f"{self.soft_col_prefix}normal_vol"),
            p_climax.alias(f"{self.soft_col_prefix}climax"),
        )
        df = df.drop(["_trailing_sma", "_forward_max", "_centred_ratio"])
        return df
