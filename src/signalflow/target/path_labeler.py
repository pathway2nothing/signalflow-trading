"""
Path-property labelers.

Three labelers that describe the *shape* of the forward price path rather
than its direction or magnitude. Per iter25 EDA, each occupies its own
hierarchical cluster (K=5) - they carry information uncorrelated with
direction / volatility / volume labels.

- :class:`HurstRegimeLabeler` - Hurst exponent (R/S) of forward returns:
  trending vs random vs mean-reverting.
- :class:`MeanReversionEventLabeler` - does an overstretched price (|z|>2σ)
  revert to its mean within the horizon?
- :class:`TrendBreakLabeler` - does the sign of the OLS slope flip between
  past and forward windows?

All three use Numba-accelerated inner loops where the operation is not
expressible as a Polars rolling expression.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow.enums import SignalCategory
from signalflow.target._numba import njit, prange
from signalflow.target._soft_helpers import (
    gaussian_membership_soft,
    sigmoid_expr,
)
from signalflow.target.base import register_target
from signalflow.target.labeler import Labeler


@njit(cache=True)
def _hurst_rs_window(series: np.ndarray) -> float:
    """Rescaled-range Hurst exponent for a single window."""
    n = len(series)
    if n < 32:
        return np.nan

    sizes = np.array([n // 8, n // 4, n // 2, n], dtype=np.int64)
    logs = np.zeros(4, dtype=np.float64)
    logr = np.zeros(4, dtype=np.float64)
    n_valid = 0

    for k in range(4):
        sz = sizes[k]
        if sz < 8:
            continue
        chunks = n // sz
        rs_sum = 0.0
        rs_cnt = 0
        for c in range(chunks):
            x = series[c * sz : (c + 1) * sz]
            mean = 0.0
            for v in x:
                mean += v
            mean /= sz
            cumdev = 0.0
            cmax = -1e18
            cmin = 1e18
            ssq = 0.0
            for v in x:
                d = v - mean
                cumdev += d
                if cumdev > cmax:
                    cmax = cumdev
                if cumdev < cmin:
                    cmin = cumdev
                ssq += d * d
            std = np.sqrt(ssq / sz)
            if std > 0:
                rs_sum += (cmax - cmin) / std
                rs_cnt += 1
        if rs_cnt > 0:
            logs[n_valid] = np.log(float(sz))
            logr[n_valid] = np.log(rs_sum / rs_cnt)
            n_valid += 1

    if n_valid < 2:
        return np.nan

    sx = sy = sxx = sxy = 0.0
    for i in range(n_valid):
        sx += logs[i]
        sy += logr[i]
        sxx += logs[i] * logs[i]
        sxy += logs[i] * logr[i]
    denom = n_valid * sxx - sx * sx
    if denom == 0:
        return np.nan
    return (n_valid * sxy - sx * sy) / denom


@njit(parallel=True, cache=True)
def _hurst_forward(log_ret: np.ndarray, horizon: int, step: int) -> np.ndarray:
    """Hurst exponent of forward `horizon` log-returns, computed every `step` bars."""
    n = len(log_ret)
    out = np.full(n, np.nan, dtype=np.float64)
    n_samples = (n - horizon + step - 1) // step
    for k in prange(n_samples):
        i = k * step
        if i + horizon > n:
            continue
        window = log_ret[i + 1 : i + 1 + horizon]
        out[i] = _hurst_rs_window(window)
    return out


@dataclass
@register_target("hurst_regime")
class HurstRegimeLabeler(Labeler):
    """Label bars by forward Hurst-exponent regime.

    The binary target treats the mean-reverting regime as the positive class; random-walk
    and trending regimes are the negative class.
    """

    signal_category: SignalCategory = SignalCategory.PRICE_STRUCTURE

    soft_classes: ClassVar[tuple[str, ...]] = ("mean_reverting", "random_walk", "trending")
    positive_classes: ClassVar[tuple[str, ...]] = ("mean_reverting",)
    softness_k: float = 20.0

    price_col: str = "close"
    horizon: int = 480
    stride: int = 30
    trending_threshold: float = 0.55
    reverting_threshold: float = 0.45

    meta_columns: tuple[str, ...] = ("hurst",)

    def __post_init__(self) -> None:
        if self.horizon < 32:
            raise ValueError("horizon must be >= 32 for R/S estimator")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")
        if not (0.0 < self.reverting_threshold < self.trending_threshold < 1.0):
            raise ValueError(
                "Require 0 < reverting_threshold < trending_threshold < 1, "
                f"got {self.reverting_threshold}, {self.trending_threshold}"
            )

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        prices = group_df.get_column(self.price_col).to_numpy().astype(np.float64)
        n = len(prices)

        log_ret = np.zeros(n, dtype=np.float64)
        log_ret[1:] = np.log(prices[1:] / prices[:-1])

        hurst = _hurst_forward(log_ret, self.horizon, self.stride)

        last = np.nan
        for i in range(n):
            if np.isnan(hurst[i]):
                hurst[i] = last
            else:
                last = hurst[i]

        labels: list[str | None] = [None] * n
        for i in range(n):
            h = hurst[i]
            if np.isnan(h):
                continue
            if h >= self.trending_threshold:
                labels[i] = "trending"
            elif h <= self.reverting_threshold:
                labels[i] = "mean_reverting"
            else:
                labels[i] = "random_walk"

        df = group_df.with_columns(pl.Series(self.out_col, labels, dtype=pl.Utf8))
        if self.include_meta:
            df = df.with_columns(pl.Series("hurst", hurst, dtype=pl.Float64))

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Gaussian-membership soft probabilities over the three Hurst centres."""
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        prices = group_df.get_column(self.price_col).to_numpy().astype(np.float64)
        n = len(prices)
        log_ret = np.zeros(n, dtype=np.float64)
        log_ret[1:] = np.log(prices[1:] / prices[:-1])

        hurst = _hurst_forward(log_ret, self.horizon, self.stride)
        last = np.nan
        for i in range(n):
            if np.isnan(hurst[i]):
                hurst[i] = last
            else:
                last = hurst[i]

        df = group_df.with_columns(pl.Series("_hurst", hurst, dtype=pl.Float64))
        centres = (self.reverting_threshold, 0.5, self.trending_threshold)
        p_rev, p_rand, p_trend = gaussian_membership_soft(pl.col("_hurst"), centres, k=self.softness_k)
        df = df.with_columns(
            p_rev.alias(f"{self.soft_col_prefix}mean_reverting"),
            p_rand.alias(f"{self.soft_col_prefix}random_walk"),
            p_trend.alias(f"{self.soft_col_prefix}trending"),
        )
        df = df.drop("_hurst")
        return df


@dataclass
@register_target("mean_reversion_event")
class MeanReversionEventLabeler(Labeler):
    """
    Label bars by whether an overstretched price reverts within horizon.

    Algorithm:
        1. Rolling mean µ and std σ of price over ``z_window``.
        2. z_now = (close[t] − µ[t]) / σ[t]
        3. z_fwd = (close[t+horizon] − µ[t]) / σ[t]   (same baseline)
        4. If |z_now| > ``stretch_threshold``:
              if |z_fwd| < ``revert_threshold`` -> ``"mean_reverted"``
              else                                -> ``"trend_continuation"``
           Otherwise                              -> ``"no_reversion"``

    Use:
        Complements :class:`StructureLabeler` and
        :class:`ZigzagStructureLabeler`, which detect extrema themselves;
        this labeler asks the follow-on question - does the extreme persist
        or revert?
    """

    signal_category: SignalCategory = SignalCategory.PRICE_STRUCTURE

    soft_classes: ClassVar[tuple[str, ...]] = ("mean_reverted", "trend_continuation", "no_reversion")
    positive_classes: ClassVar[tuple[str, ...]] = ("mean_reverted",)

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

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

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

        overstretched = pl.col("_z_now").abs() > self.stretch_threshold
        reverted = overstretched & (pl.col("_z_fwd").abs() < self.revert_threshold)

        label_expr = (
            pl.when(pl.col("_z_now").is_null() | pl.col("_z_fwd").is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(reverted)
            .then(pl.lit("mean_reverted"))
            .when(overstretched)
            .then(pl.lit("trend_continuation"))
            .otherwise(pl.lit("no_reversion"))
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
        """Soft triple ``(p_mean_reverted, p_trend_continuation, p_no_reversion)``."""
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

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
        null_mask = pl.col("_z_now").is_null() | pl.col("_z_fwd").is_null()
        p_stretched = sigmoid_expr(pl.col("_z_now").abs() - pl.lit(self.stretch_threshold), self.softness_k)
        p_revert_given = sigmoid_expr(pl.lit(self.revert_threshold) - pl.col("_z_fwd").abs(), self.softness_k)

        df = df.with_columns(
            pl.when(null_mask)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(p_stretched * p_revert_given)
            .alias(f"{self.soft_col_prefix}mean_reverted"),
            pl.when(null_mask)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(p_stretched * (pl.lit(1.0) - p_revert_given))
            .alias(f"{self.soft_col_prefix}trend_continuation"),
            pl.when(null_mask)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.lit(1.0) - p_stretched)
            .alias(f"{self.soft_col_prefix}no_reversion"),
        )
        df = df.drop(["_mu", "_sd", "_z_now", "_z_fwd"])
        return df


@njit(parallel=True, cache=True)
def _slope_windows(log_close: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS slope of log_close over past and forward windows for each bar."""
    n = len(log_close)
    past = np.full(n, np.nan, dtype=np.float64)
    fwd = np.full(n, np.nan, dtype=np.float64)

    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_c = x - x_mean
    denom = (x_c * x_c).sum()

    for i in prange(window, n - window):
        y_past = log_close[i - window + 1 : i + 1]
        y_fwd = log_close[i + 1 : i + 1 + window]
        ym_p = 0.0
        ym_f = 0.0
        for k in range(window):
            ym_p += y_past[k]
            ym_f += y_fwd[k]
        ym_p /= window
        ym_f /= window
        num_p = 0.0
        num_f = 0.0
        for k in range(window):
            num_p += (y_past[k] - ym_p) * x_c[k]
            num_f += (y_fwd[k] - ym_f) * x_c[k]
        past[i] = num_p / denom
        fwd[i] = num_f / denom
    return past, fwd


@dataclass
@register_target("trend_break")
class TrendBreakLabeler(Labeler):
    """
    Label bars by whether forward OLS slope flips sign vs past slope.

    Algorithm:
        1. Fit OLS slope of log(close) over past ``window`` bars -> b_past
        2. Fit OLS slope of log(close) over forward ``window`` bars -> b_fwd
        3. Require both |b_past| and |b_fwd| > ``slope_eps`` (otherwise null/noise).
        4. If sign(b_past) != sign(b_fwd) -> ``"break"``
           Else                            -> ``"continue"``

    Use:
        Complements :class:`TrendScanningLabeler` (which detects the
        *presence* of a trend) by classifying its *continuation* vs *break*.
    """

    signal_category: SignalCategory = SignalCategory.TREND_MOMENTUM

    soft_classes: ClassVar[tuple[str, ...]] = ("no_break", "continue", "break")
    positive_classes: ClassVar[tuple[str, ...]] = ("break",)

    price_col: str = "close"
    window: int = 240
    slope_eps: float = 1e-7

    meta_columns: tuple[str, ...] = ("past_slope", "fwd_slope")

    def __post_init__(self) -> None:
        if self.window <= 1:
            raise ValueError("window must be > 1")
        if self.slope_eps < 0:
            raise ValueError("slope_eps must be >= 0")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        prices = group_df.get_column(self.price_col).to_numpy().astype(np.float64)
        log_close = np.log(prices)

        if len(log_close) < 2 * self.window:
            df = group_df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(self.out_col))
            if self.include_meta:
                df = df.with_columns(
                    pl.lit(None).alias("past_slope"),
                    pl.lit(None).alias("fwd_slope"),
                )
            return df

        past, fwd = _slope_windows(log_close, self.window)
        n = len(prices)
        labels: list[str | None] = [None] * n
        for i in range(n):
            if np.isnan(past[i]) or np.isnan(fwd[i]):
                continue
            if abs(past[i]) < self.slope_eps or abs(fwd[i]) < self.slope_eps:
                labels[i] = "no_break"
                continue
            sp = 1 if past[i] > 0 else -1
            sf = 1 if fwd[i] > 0 else -1
            labels[i] = "break" if sp != sf else "continue"

        df = group_df.with_columns(pl.Series(self.out_col, labels, dtype=pl.Utf8))
        if self.include_meta:
            df = df.with_columns(
                pl.Series("past_slope", past, dtype=pl.Float64),
                pl.Series("fwd_slope", fwd, dtype=pl.Float64),
            )

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df
