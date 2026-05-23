"""Volume climax labeler.

Labels bars based on the *maximum* forward volume within a horizon
relative to a trailing volume SMA — captures climax/exhaustion events
rather than the average forward volume.

Complements :class:`VolumeRegimeLabeler` (which uses forward *mean*):
this labeler is more sensitive to single high-volume bars within the
window and is per-pair idiosyncratic (per iter25 EDA: cross-pair κ ≈ 0.24
vs 0.97 for volume_regime).

Implementation uses pure Polars expressions for performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from signalflow.core import labeler
from signalflow.core.enums import SignalCategory
from signalflow.target.base import Labeler


@dataclass
@labeler("volume_climax")
class VolumeClimaxLabeler(Labeler):
    """Label bars by forward max-volume vs trailing SMA ratio.

    Algorithm:
        1. trailing_sma = rolling_mean(volume, vol_sma_window)
        2. forward_max  = rolling_max(volume[t+1 : t+horizon+1])
        3. ratio        = forward_max / trailing_sma
        4. If ratio > climax_threshold -> ``"climax"``
           If ratio < calm_threshold   -> ``"calm_vol"``
           Otherwise                    -> ``"normal_vol"``

    Attributes:
        volume_col: Volume column. Default: ``"volume"``.
        horizon: Number of forward bars. Default: ``240``.
        vol_sma_window: Trailing SMA window. Default: ``1440``.
        climax_threshold: Ratio above which the window is a climax.
            Default: ``5.0``. (Tuned for 1m crypto bars; forward window
            often contains a single spike-bar far above the SMA.)
        calm_threshold: Ratio below which the window is calm.
            Default: ``1.5``.

    Example:
        ```python
        from signalflow.target.volume_climax_labeler import VolumeClimaxLabeler

        labeler = VolumeClimaxLabeler(
            horizon=240,
            climax_threshold=3.0,
            calm_threshold=1.2,
            mask_to_signals=False,
        )
        result = labeler.compute(ohlcv_df)
        ```
    """

    signal_category: SignalCategory = SignalCategory.VOLUME_LIQUIDITY

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
