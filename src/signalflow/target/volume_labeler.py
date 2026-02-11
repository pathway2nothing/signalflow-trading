"""Volume regime labeler.

Labels bars based on forward volume ratio relative to a rolling
volume moving average.

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
@sf_component(name="volume_regime")
class VolumeRegimeLabeler(Labeler):
    """Label bars by forward volume regime.

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

    Attributes:
        volume_col: Volume column. Default: ``"volume"``.
        horizon: Number of forward bars. Default: ``60``.
        vol_sma_window: Trailing SMA window. Default: ``1440``.
        spike_threshold: Threshold for volume spike. Default: ``2.0``.
        drought_threshold: Threshold for volume drought. Default: ``0.3``.

    Example:
        ```python
        from signalflow.target.volume_labeler import VolumeRegimeLabeler

        labeler = VolumeRegimeLabeler(
            horizon=60,
            spike_threshold=2.0,
            drought_threshold=0.3,
            mask_to_signals=False,
        )
        result = labeler.compute(ohlcv_df)
        ```
    """

    signal_category: SignalCategory = SignalCategory.VOLUME_LIQUIDITY

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
        """Compute volume regime labels for a single pair.

        Args:
            group_df: Single pair data sorted by timestamp.
            data_context: Optional additional context.

        Returns:
            DataFrame with same row count, plus label and optional meta columns.
        """
        if group_df.height == 0:
            return group_df

        if self.volume_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.volume_col}'")

        vol = pl.col(self.volume_col)

        # Step 1: Trailing volume SMA using Polars rolling_mean
        # min_samples=1 allows SMA from the first bar (like original behavior)
        df = group_df.with_columns(
            vol.rolling_mean(window_size=self.vol_sma_window, min_samples=1).alias("_trailing_sma")
        )

        # Step 2: Forward average volume
        # To compute mean(volume[t+1 : t+horizon+1]), we:
        # - Shift volume by -1 to start from next bar
        # - Apply rolling_mean with window=horizon
        # - The result at position t+horizon-1 contains mean of [t, t+horizon)
        # - Shift back by -(horizon-1) to align with position t
        df = df.with_columns(
            vol.shift(-1)
            .rolling_mean(window_size=self.horizon, min_samples=1)
            .shift(-(self.horizon - 1))
            .alias("_forward_avg")
        )

        # Step 3: Volume ratio = forward_avg / trailing_sma
        df = df.with_columns(
            pl.when(pl.col("_trailing_sma") > 0)
            .then(pl.col("_forward_avg") / pl.col("_trailing_sma"))
            .otherwise(pl.lit(None))
            .alias("_volume_ratio")
        )

        # Step 4-5: Assign labels based on thresholds
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

        # Clean up temporary columns
        df = df.drop(["_trailing_sma", "_forward_avg", "_volume_ratio"])

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df
