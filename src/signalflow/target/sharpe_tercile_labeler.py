"""Sharpe-tercile labeler.

Labels bars by the tercile of a risk-adjusted forward return
(``forward_log_return / forward_volatility``), measured against a trailing
rolling baseline so the bins adapt per-pair without look-ahead.

Complements :class:`FixedHorizonLabeler` and :class:`TripleBarrierLabeler`:
those measure *raw* direction; this one penalises chop and rewards
clean directional moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow.core import labeler
from signalflow.core.enums import SignalCategory
from signalflow.target._soft_helpers import percentile_tercile_soft
from signalflow.target.base import Labeler


@dataclass
@labeler("sharpe_tercile")
class SharpeTercileLabeler(Labeler):
    """Label bars by tercile of forward risk-adjusted log-return.

    Algorithm:
        1. fwd_log_ret = ln(close[t+horizon] / close[t])
        2. fwd_vol     = std(log_returns[t+1 : t+horizon+1])
        3. sharpe      = fwd_log_ret / fwd_vol
        4. Within trailing ``lookback_window`` of past sharpe values,
           assign tercile:
              percentile > upper_quantile -> ``"sharpe_pos"``
              percentile < lower_quantile -> ``"sharpe_neg"``
              otherwise                    -> ``"sharpe_mid"``

    Attributes:
        price_col: Price column. Default: ``"close"``.
        horizon: Forward window for return and vol. Default: ``240``.
        lookback_window: Trailing window for tercile fitting. Default: ``10080``.
        upper_quantile: Default ``0.67``.
        lower_quantile: Default ``0.33``.

    Example:
        ```python
        from signalflow.target.sharpe_tercile_labeler import SharpeTercileLabeler

        labeler = SharpeTercileLabeler(
            horizon=240,
            mask_to_signals=False,
        )
        result = labeler.compute(ohlcv_df)
        ```
    """

    signal_category: SignalCategory = SignalCategory.PRICE_DIRECTION

    soft_classes: ClassVar[tuple[str, ...]] = ("sharpe_neg", "sharpe_mid", "sharpe_pos")
    softness_k: float = 20.0

    price_col: str = "close"
    horizon: int = 240
    lookback_window: int = 10080
    upper_quantile: float = 0.67
    lower_quantile: float = 0.33

    meta_columns: tuple[str, ...] = ("sharpe", "sharpe_percentile")

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if not (0.0 < self.lower_quantile < self.upper_quantile < 1.0):
            raise ValueError(
                "Require 0 < lower_quantile < upper_quantile < 1, "
                f"got {self.lower_quantile}, {self.upper_quantile}"
            )
        if self.lookback_window < 100:
            raise ValueError("lookback_window must be >= 100 for stable terciles")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        price = pl.col(self.price_col)
        log_ret = (price / price.shift(1)).log()

        df = group_df.with_columns(log_ret.alias("_log_ret"))

        df = df.with_columns(
            (price.shift(-self.horizon) / price).log().alias("_fwd_ret"),
            pl.col("_log_ret")
            .shift(-1)
            .rolling_std(window_size=self.horizon, min_samples=2)
            .shift(-(self.horizon - 1))
            .alias("_fwd_vol"),
        )

        df = df.with_columns(
            pl.when(pl.col("_fwd_vol") > 0)
            .then(pl.col("_fwd_ret") / pl.col("_fwd_vol"))
            .otherwise(pl.lit(None))
            .alias("_sharpe"),
        )

        sharpe_arr = df.get_column("_sharpe").to_numpy()
        pct = self._rolling_percentile(sharpe_arr, self.lookback_window)

        labels: list[str | None] = [None] * len(sharpe_arr)
        for i in range(len(sharpe_arr)):
            p = pct[i]
            if np.isnan(p):
                continue
            if p > self.upper_quantile:
                labels[i] = "sharpe_pos"
            elif p < self.lower_quantile:
                labels[i] = "sharpe_neg"
            else:
                labels[i] = "sharpe_mid"

        df = df.with_columns(pl.Series(self.out_col, labels, dtype=pl.Utf8))

        if self.include_meta:
            df = df.with_columns(
                pl.col("_sharpe").alias("sharpe"),
                pl.Series("sharpe_percentile", pct, dtype=pl.Float64),
            )

        df = df.drop(["_log_ret", "_fwd_ret", "_fwd_vol", "_sharpe"])

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Soft tercile probabilities ``(p_sharpe_neg, p_sharpe_mid, p_sharpe_pos)``.

        Uses the same rolling Sharpe percentile that drives the hard tercile
        and maps it through :func:`percentile_tercile_soft` so the middle
        bucket is represented explicitly.
        """
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        price = pl.col(self.price_col)
        log_ret = (price / price.shift(1)).log()
        df = group_df.with_columns(log_ret.alias("_log_ret"))
        df = df.with_columns(
            (price.shift(-self.horizon) / price).log().alias("_fwd_ret"),
            pl.col("_log_ret")
            .shift(-1)
            .rolling_std(window_size=self.horizon, min_samples=2)
            .shift(-(self.horizon - 1))
            .alias("_fwd_vol"),
        )
        df = df.with_columns(
            pl.when(pl.col("_fwd_vol") > 0)
            .then(pl.col("_fwd_ret") / pl.col("_fwd_vol"))
            .otherwise(pl.lit(None))
            .alias("_sharpe"),
        )

        sharpe_arr = df.get_column("_sharpe").to_numpy()
        pct = self._rolling_percentile(sharpe_arr, self.lookback_window)
        df = df.with_columns(pl.Series("_pct", pct, dtype=pl.Float64))

        p_neg, p_mid, p_pos = percentile_tercile_soft(
            pl.col("_pct"),
            lower_q=self.lower_quantile,
            upper_q=self.upper_quantile,
            k=self.softness_k,
        )
        df = df.with_columns(
            p_neg.alias(f"{self.soft_col_prefix}sharpe_neg"),
            p_mid.alias(f"{self.soft_col_prefix}sharpe_mid"),
            p_pos.alias(f"{self.soft_col_prefix}sharpe_pos"),
        )
        df = df.drop(["_log_ret", "_fwd_ret", "_fwd_vol", "_sharpe", "_pct"])
        return df

    @staticmethod
    def _rolling_percentile(values: np.ndarray, window: int) -> np.ndarray:
        n = len(values)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            v = values[i]
            if v is None or np.isnan(v):
                continue
            start = max(0, i - window + 1)
            win = values[start : i + 1]
            valid = win[~np.isnan(win)]
            if len(valid) < 20:
                continue
            out[i] = float(np.mean(valid <= v))
        return out
