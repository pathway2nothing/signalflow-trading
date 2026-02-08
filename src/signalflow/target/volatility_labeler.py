"""Volatility regime labeler.

Labels bars based on forward realized volatility percentile within
a rolling lookback window.
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
@sf_component(name="volatility_regime")
class VolatilityRegimeLabeler(Labeler):
    """Label bars by forward realized volatility regime.

    Algorithm:
        1. Compute log returns: ``ln(close[t] / close[t-1])``
        2. Forward realized volatility: ``std(log_returns[t+1 : t+horizon+1])``
           using a shift(-i) pattern to gather future returns.
        3. Rolling percentile of realized vol over ``lookback_window``.
        4. If vol > ``upper_quantile`` percentile -> ``"vol_high"``
        5. If vol < ``lower_quantile`` percentile -> ``"vol_low"``
        6. Otherwise -> ``null`` (Polars null)

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

        # Step 1: log returns
        df = group_df.with_columns((pl.col(self.price_col) / pl.col(self.price_col).shift(1)).log().alias("_log_ret"))

        # Step 2: Forward realized volatility via numpy for the shift(-i) pattern
        log_ret_arr = df["_log_ret"].to_numpy().astype(np.float64)
        n = len(log_ret_arr)
        realized_vol = np.full(n, np.nan, dtype=np.float64)

        for t in range(n):
            start = t + 1
            end = t + 1 + self.horizon
            if end > n:
                break
            window = log_ret_arr[start:end]
            valid = window[~np.isnan(window)]
            if len(valid) >= 2:
                realized_vol[t] = np.std(valid, ddof=1)

        # Step 3: Rolling percentile using numpy
        vol_percentile = np.full(n, np.nan, dtype=np.float64)
        for t in range(n):
            if np.isnan(realized_vol[t]):
                continue
            lb_start = max(0, t - self.lookback_window + 1)
            window = realized_vol[lb_start : t + 1]
            valid = window[~np.isnan(window)]
            if len(valid) < 2:
                continue
            vol_percentile[t] = np.mean(valid <= realized_vol[t])

        # Step 4-5: Assign labels
        labels = np.empty(n, dtype=object)
        for t in range(n):
            if np.isnan(vol_percentile[t]):
                labels[t] = None
            elif vol_percentile[t] > self.upper_quantile:
                labels[t] = "vol_high"
            elif vol_percentile[t] < self.lower_quantile:
                labels[t] = "vol_low"
            else:
                labels[t] = None

        df = df.with_columns(pl.Series(name=self.out_col, values=labels.tolist(), dtype=pl.Utf8))

        if self.include_meta:
            df = df.with_columns(
                [
                    pl.Series(name="realized_vol", values=realized_vol.tolist(), dtype=pl.Float64),
                    pl.Series(name="vol_percentile", values=vol_percentile.tolist(), dtype=pl.Float64),
                ]
            )

        df = df.drop("_log_ret")

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df
