"""Volume regime labeler.

Labels bars based on forward volume ratio relative to a rolling
volume moving average.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
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
        3. If ratio > ``spike_threshold`` -> ``"volume_spike"``
        4. If ratio < ``drought_threshold`` -> ``"volume_drought"``
        5. Otherwise -> ``null``

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

        vol_arr = group_df[self.volume_col].to_numpy().astype(np.float64)
        n = len(vol_arr)

        # Step 1: Trailing volume SMA
        trailing_sma = np.full(n, np.nan, dtype=np.float64)
        cumsum = np.nancumsum(vol_arr)
        for t in range(n):
            lb_start = max(0, t - self.vol_sma_window + 1)
            count = t - lb_start + 1
            if lb_start == 0:
                trailing_sma[t] = cumsum[t] / count
            else:
                trailing_sma[t] = (cumsum[t] - cumsum[lb_start - 1]) / count

        # Step 2: Forward average volume
        forward_avg = np.full(n, np.nan, dtype=np.float64)
        for t in range(n):
            start = t + 1
            end = t + 1 + self.horizon
            if end > n:
                break
            forward_avg[t] = np.mean(vol_arr[start:end])

        # Step 3: Volume ratio
        volume_ratio = np.full(n, np.nan, dtype=np.float64)
        valid_mask = (~np.isnan(forward_avg)) & (~np.isnan(trailing_sma)) & (trailing_sma > 0)
        volume_ratio[valid_mask] = forward_avg[valid_mask] / trailing_sma[valid_mask]

        # Step 4-5: Labels
        labels = [None] * n
        for t in range(n):
            if np.isnan(volume_ratio[t]):
                continue
            if volume_ratio[t] > self.spike_threshold:
                labels[t] = "volume_spike"
            elif volume_ratio[t] < self.drought_threshold:
                labels[t] = "volume_drought"

        df = group_df.with_columns(pl.Series(name=self.out_col, values=labels, dtype=pl.Utf8))

        if self.include_meta:
            df = df.with_columns(pl.Series(name="volume_ratio", values=volume_ratio.tolist(), dtype=pl.Float64))

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df
