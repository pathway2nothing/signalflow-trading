"""Drawdown / runup labeler.

Labels bars by the worst forward drawdown, the best forward runup, or the
ratio of forward return to drawdown (Calmar) — captures path-risk that
direction labels miss entirely.

Uses rolling-quantile binning so that thresholds adapt to per-pair scale
(no look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from signalflow.core import labeler
from signalflow.core.enums import SignalCategory
from signalflow.target.base import Labeler

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn

        return decorator if args and callable(args[0]) is False else args[0]

    prange = range  # type: ignore[assignment]


@njit(parallel=True, cache=True)
def _forward_dd_ru_ret(prices: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (max_dd, max_ru, log_return) over forward horizon for each bar.

    Returns:
        max_dd[i] = -min over [t+1..t+H] of (price/cummax - 1)   (positive severity)
        max_ru[i] =  max over [t+1..t+H] of (price/cummin - 1)
        log_ret[i] = log(price[t+H] / price[t])
    """
    n = len(prices)
    max_dd = np.full(n, np.nan, dtype=np.float64)
    max_ru = np.full(n, np.nan, dtype=np.float64)
    log_ret = np.full(n, np.nan, dtype=np.float64)

    for i in prange(n - horizon):
        cummax = prices[i + 1]
        cummin = prices[i + 1]
        worst_dd = 0.0
        best_ru = 0.0
        for k in range(1, horizon + 1):
            p = prices[i + k]
            if p > cummax:
                cummax = p
            if p < cummin:
                cummin = p
            dd = p / cummax - 1.0
            ru = p / cummin - 1.0
            if dd < worst_dd:
                worst_dd = dd
            if ru > best_ru:
                best_ru = ru
        max_dd[i] = -worst_dd
        max_ru[i] = best_ru
        log_ret[i] = np.log(prices[i + horizon] / prices[i])

    return max_dd, max_ru, log_ret


@dataclass
@labeler("drawdown")
class DrawdownLabeler(Labeler):
    """Label bars by forward path-risk metric, terciled within a rolling baseline.

    Supports three modes:
        - ``"drawdown"``: max forward drawdown severity.
            Classes: ``dd_mild`` / ``dd_normal`` / ``dd_severe``.
        - ``"runup"``: max forward runup magnitude.
            Classes: ``ru_mild`` / ``ru_normal`` / ``ru_strong``.
        - ``"calmar"``: forward log-return divided by max drawdown.
            Classes: ``calmar_low`` / ``calmar_mid`` / ``calmar_high``.

    For each bar the chosen metric is computed over the forward window,
    then assigned a tercile based on its rank inside a trailing
    ``lookback_window`` of the same metric (per-pair rolling, no look-ahead).

    Attributes:
        price_col: Price column. Default: ``"close"``.
        horizon: Forward window for the metric. Default: ``480``.
        mode: Which metric to label. One of ``"drawdown"``, ``"runup"``,
            ``"calmar"``. Default: ``"drawdown"``.
        lookback_window: Trailing window for tercile fitting. Default: ``10080``
            (1 week of 1-min bars).
        upper_quantile: Default ``0.67``.
        lower_quantile: Default ``0.33``.

    Example:
        ```python
        from signalflow.target.drawdown_labeler import DrawdownLabeler

        labeler = DrawdownLabeler(
            horizon=480,
            mode="drawdown",
            mask_to_signals=False,
        )
        result = labeler.compute(ohlcv_df)
        ```
    """

    signal_category: SignalCategory = SignalCategory.VOLATILITY

    price_col: str = "close"
    horizon: int = 480
    mode: str = "drawdown"
    lookback_window: int = 10080
    upper_quantile: float = 0.67
    lower_quantile: float = 0.33

    meta_columns: tuple[str, ...] = ("metric_value", "metric_percentile")

    _VALID_MODES = ("drawdown", "runup", "calmar")

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {self._VALID_MODES}, got {self.mode!r}")
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

        prices = group_df.get_column(self.price_col).to_numpy().astype(np.float64)
        max_dd, max_ru, log_ret = _forward_dd_ru_ret(prices, self.horizon)

        if self.mode == "drawdown":
            metric = max_dd
            names = ("dd_mild", "dd_normal", "dd_severe")
        elif self.mode == "runup":
            metric = max_ru
            names = ("ru_mild", "ru_normal", "ru_strong")
        else:  # calmar
            metric = np.where(max_dd > 1e-12, log_ret / max_dd, np.nan)
            names = ("calmar_low", "calmar_mid", "calmar_high")

        # Rolling tercile percentile (trailing window, no look-ahead)
        pct = self._rolling_percentile(metric, self.lookback_window)

        labels: list[str | None] = [None] * len(metric)
        for i in range(len(metric)):
            p = pct[i]
            if np.isnan(p):
                continue
            if p > self.upper_quantile:
                labels[i] = names[2]
            elif p < self.lower_quantile:
                labels[i] = names[0]
            else:
                labels[i] = names[1]

        df = group_df.with_columns(pl.Series(self.out_col, labels, dtype=pl.Utf8))
        if self.include_meta:
            df = df.with_columns(
                pl.Series("metric_value", metric, dtype=pl.Float64),
                pl.Series("metric_percentile", pct, dtype=pl.Float64),
            )

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df

    @staticmethod
    def _rolling_percentile(values: np.ndarray, window: int) -> np.ndarray:
        """Trailing rolling percentile: fraction of past ``window`` values <= current."""
        n = len(values)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            if np.isnan(values[i]):
                continue
            start = max(0, i - window + 1)
            win = values[start : i + 1]
            valid = win[~np.isnan(win)]
            if len(valid) < 20:
                continue
            out[i] = float(np.mean(valid <= values[i]))
        return out
