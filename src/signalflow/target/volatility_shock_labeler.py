"""Volatility shock labeler.

Labels bars based on the z-score change of forward realized volatility
relative to a trailing volatility regime — captures regime *changes* rather
than absolute levels. Complements :class:`VolatilityRegimeLabeler`.

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
@labeler("volatility_shock")
class VolatilityShockLabeler(Labeler):
    """Label bars by forward-vs-past volatility z-score shock.

    Detects abrupt regime transitions in realized volatility. While
    :class:`VolatilityRegimeLabeler` answers "what is the forward vol level
    (high/low) relative to history?", this labeler answers "how *different*
    is the forward vol from the recent regime?".

    Algorithm:
        1. log_returns = ln(close[t] / close[t-1])
        2. forward_vol  = std(log_returns[t+1 : t+horizon+1])
        3. past_vol     = rolling_std(log_returns, past_vol_window)  -- trailing baseline
        4. past_vol_std = rolling_std(rolling_std(log_returns, vol_window_short), past_vol_window)
                          -- variability of vol itself (denominator for z-score)
        5. z = (forward_vol - past_vol) / past_vol_std
        6. If z >  shock_threshold      -> ``"vol_shock_up"``
           If z < -shock_threshold      -> ``"vol_shock_down"``
           Otherwise                     -> ``"vol_normal"``

    Attributes:
        price_col: Price column name. Default: ``"close"``.
        horizon: Forward window for realized vol. Default: ``120``.
        past_vol_window: Trailing window for baseline vol & vol-of-vol.
            Default: ``1440``.
        vol_window_short: Short window for vol-of-vol inner std. Default: ``60``.
        shock_threshold: |z| above which the bar is labelled a shock.
            Default: ``1.0``.

    Example:
        ```python
        from signalflow.target.volatility_shock_labeler import VolatilityShockLabeler

        labeler = VolatilityShockLabeler(
            horizon=120,
            past_vol_window=1440,
            shock_threshold=1.0,
            mask_to_signals=False,
        )
        result = labeler.compute(ohlcv_df)
        ```
    """

    signal_category: SignalCategory = SignalCategory.VOLATILITY

    price_col: str = "close"
    horizon: int = 120
    past_vol_window: int = 1440
    vol_window_short: int = 60
    shock_threshold: float = 1.0

    meta_columns: tuple[str, ...] = ("vol_z",)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.past_vol_window <= self.vol_window_short:
            raise ValueError("past_vol_window must be > vol_window_short")
        if self.shock_threshold <= 0:
            raise ValueError("shock_threshold must be > 0")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        log_ret = (pl.col(self.price_col) / pl.col(self.price_col).shift(1)).log()

        df = group_df.with_columns(log_ret.alias("_log_ret"))

        df = df.with_columns(
            pl.col("_log_ret").rolling_std(self.past_vol_window, min_samples=2).alias("_past_vol"),
            pl.col("_log_ret").rolling_std(self.vol_window_short, min_samples=2).alias("_inner_vol"),
            pl.col("_log_ret")
            .shift(-1)
            .rolling_std(window_size=self.horizon, min_samples=2)
            .shift(-(self.horizon - 1))
            .alias("_fwd_vol"),
        )

        df = df.with_columns(
            pl.col("_inner_vol").rolling_std(self.past_vol_window, min_samples=2).alias("_past_vol_std"),
        )

        df = df.with_columns(
            pl.when(pl.col("_past_vol_std") > 0)
            .then((pl.col("_fwd_vol") - pl.col("_past_vol")) / pl.col("_past_vol_std"))
            .otherwise(pl.lit(None))
            .alias("_vol_z")
        )

        label_expr = (
            pl.when(pl.col("_vol_z").is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(pl.col("_vol_z") > self.shock_threshold)
            .then(pl.lit("vol_shock_up"))
            .when(pl.col("_vol_z") < -self.shock_threshold)
            .then(pl.lit("vol_shock_down"))
            .otherwise(pl.lit("vol_normal"))
            .alias(self.out_col)
        )

        df = df.with_columns(label_expr)

        if self.include_meta:
            df = df.with_columns(pl.col("_vol_z").alias("vol_z"))

        df = df.drop(["_log_ret", "_past_vol", "_inner_vol", "_fwd_vol", "_past_vol_std", "_vol_z"])

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df
