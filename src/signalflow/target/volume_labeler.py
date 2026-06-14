"""
Volume regime labeler.

Labels bars based on forward volume ratio relative to a rolling
volume moving average.

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
@register_target("volume_regime")
class VolumeRegimeLabeler(Labeler):
    """
    Label bars by forward volume regime.

    Detects volume spikes and droughts by comparing forward average
    volume to a trailing volume SMA.

    Algorithm:
        1. Compute trailing volume SMA: ``rolling_mean(volume, vol_sma_window)``.
        2. Compute forward volume ratio:
           ``mean(volume[t+1 : t+horizon+1]) / trailing_sma[t]``
        3. If ratio > ``spike_threshold`` -> ``"abnormal_volume"``
        4. If ratio < ``drought_threshold`` -> ``"illiquidity"``
        5. Otherwise -> ``null``

    Implementation:
        Uses pure Polars expressions instead of numpy loops for better
        performance and memory efficiency.
    """

    signal_category: SignalCategory = SignalCategory.VOLUME_LIQUIDITY

    soft_classes: ClassVar[tuple[str, ...]] = ("illiquidity", "normal_volume", "abnormal_volume")

    volume_col: str = "volume"
    horizon: int = 60
    vol_sma_window: int = 1440
    spike_threshold: float = 2.0
    drought_threshold: float = 0.3

    meta_columns: tuple[str, ...] = ("volume_ratio",)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.vol_sma_window <= 0:
            raise ValueError("vol_sma_window must be > 0")
        if self.drought_threshold >= self.spike_threshold:
            raise ValueError(
                f"drought_threshold ({self.drought_threshold}) must be < spike_threshold ({self.spike_threshold})"
            )

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Compute volume regime labels for a single pair."""
        if group_df.height == 0:
            return group_df

        if self.volume_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.volume_col}'")

        vol = pl.col(self.volume_col)


        df = group_df.with_columns(
            vol.rolling_mean(window_size=self.vol_sma_window, min_samples=1).alias("_trailing_sma")
        )


        df = df.with_columns(
            vol.shift(-1)
            .rolling_mean(window_size=self.horizon, min_samples=1)
            .shift(-(self.horizon - 1))
            .alias("_forward_avg")
        )


        df = df.with_columns(
            pl.when(pl.col("_trailing_sma") > 0)
            .then(pl.col("_forward_avg") / pl.col("_trailing_sma"))
            .otherwise(pl.lit(None))
            .alias("_volume_ratio")
        )


        label_expr = (
            pl.when(pl.col("_volume_ratio").is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(pl.col("_volume_ratio") > self.spike_threshold)
            .then(pl.lit("abnormal_volume"))
            .when(pl.col("_volume_ratio") < self.drought_threshold)
            .then(pl.lit("illiquidity"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias(self.out_col)
        )

        df = df.with_columns(label_expr)

        if self.include_meta:
            df = df.with_columns(pl.col("_volume_ratio").alias("volume_ratio"))


        df = df.drop(["_trailing_sma", "_forward_avg", "_volume_ratio"])

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Soft triple ``(p_illiquidity, p_normal_volume, p_abnormal_volume)``."""
        if group_df.height == 0:
            return group_df
        if self.volume_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.volume_col}'")

        vol = pl.col(self.volume_col)
        df = group_df.with_columns(
            vol.rolling_mean(window_size=self.vol_sma_window, min_samples=1).alias("_trailing_sma")
        )
        df = df.with_columns(
            vol.shift(-1)
            .rolling_mean(window_size=self.horizon, min_samples=1)
            .shift(-(self.horizon - 1))
            .alias("_forward_avg")
        )
        df = df.with_columns(
            pl.when(pl.col("_trailing_sma") > 0)
            .then(pl.col("_forward_avg") / pl.col("_trailing_sma") - 1.0)
            .otherwise(pl.lit(None))
            .alias("_excess_ratio")
        )

        p_dry, p_norm, p_spike = signed_tercile_soft(
            pl.col("_excess_ratio"),
            neg_threshold=1.0 - self.drought_threshold,
            pos_threshold=self.spike_threshold - 1.0,
            k=self.softness_k,
        )
        df = df.with_columns(
            p_dry.alias(f"{self.soft_col_prefix}illiquidity"),
            p_norm.alias(f"{self.soft_col_prefix}normal_volume"),
            p_spike.alias(f"{self.soft_col_prefix}abnormal_volume"),
        )
        df = df.drop(["_trailing_sma", "_forward_avg", "_excess_ratio"])
        return df
