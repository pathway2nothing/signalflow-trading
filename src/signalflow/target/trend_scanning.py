"""Trend scanning labeler.

Implements De Prado's trend scanning method (ML for Asset Managers, Ch5)
to detect statistically significant trends in price data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from signalflow.core import sf_component
from signalflow.core.enums import SignalCategory
from signalflow.target.base import Labeler

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _trend_scan_numpy(
    prices: np.ndarray,
    min_lf: int,
    max_lf: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure numpy fallback for trend scanning.

    For each bar t, fits OLS price[t+i] = alpha + beta*i for i=0..L-1
    across multiple window sizes L, and selects the window with the
    highest absolute t-statistic on beta.

    Args:
        prices: 1-D array of prices.
        min_lf: Minimum look-forward window.
        max_lf: Maximum look-forward window.
        step: Step between window sizes.

    Returns:
        t_stats: Best t-statistic for each bar.
        best_windows: Window size corresponding to best t-stat.
    """
    n = len(prices)
    t_stats = np.full(n, np.nan, dtype=np.float64)
    best_windows = np.full(n, np.nan, dtype=np.float64)

    windows = list(range(min_lf, max_lf + 1, step))

    for t in range(n):
        best_abs = -1.0
        best_t_stat = np.nan
        best_w = np.nan

        for L in windows:
            end = t + L
            if end > n:
                continue

            y = prices[t:end].copy()
            L_len = len(y)
            if L_len < 3:
                continue

            x = np.arange(L_len, dtype=np.float64)
            x_mean = x.mean()
            y_mean = y.mean()

            ss_xx = np.sum((x - x_mean) ** 2)
            if ss_xx == 0.0:
                continue

            ss_xy = np.sum((x - x_mean) * (y - y_mean))
            beta = ss_xy / ss_xx
            alpha = y_mean - beta * x_mean

            residuals = y - (alpha + beta * x)
            sse = np.sum(residuals**2)
            dof = L_len - 2
            if dof <= 0:
                continue

            mse = sse / dof
            if mse <= 0.0:
                continue

            se_beta = np.sqrt(mse / ss_xx)
            if se_beta == 0.0:
                continue

            t_val = beta / se_beta
            abs_t = abs(t_val)

            if abs_t > best_abs:
                best_abs = abs_t
                best_t_stat = t_val
                best_w = float(L)

        t_stats[t] = best_t_stat
        best_windows[t] = best_w

    return t_stats, best_windows


if _HAS_NUMBA:

    @njit(cache=True)
    def _trend_scan_numba(
        prices: np.ndarray,
        min_lf: int,
        max_lf: int,
        step: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated trend scanning.

        For each bar t, fits OLS price[t+i] = alpha + beta*i for i=0..L-1
        across multiple window sizes L, and selects the window with the
        highest absolute t-statistic on beta.

        Args:
            prices: 1-D array of prices.
            min_lf: Minimum look-forward window.
            max_lf: Maximum look-forward window.
            step: Step between window sizes.

        Returns:
            t_stats: Best t-statistic for each bar.
            best_windows: Window size corresponding to best t-stat.
        """
        n = len(prices)
        t_stats = np.full(n, np.nan, dtype=np.float64)
        best_windows = np.full(n, np.nan, dtype=np.float64)

        for t in range(n):
            best_abs = -1.0
            best_t_stat = np.nan
            best_w = np.nan

            L = min_lf
            while L <= max_lf:
                end = t + L
                if end > n:
                    L += step
                    continue

                L_len = L
                if L_len < 3:
                    L += step
                    continue

                # OLS: y = alpha + beta * x, x = 0, 1, ..., L_len-1
                # Using closed-form formulas
                x_mean = (L_len - 1.0) / 2.0
                y_sum = 0.0
                for i in range(L_len):
                    y_sum += prices[t + i]
                y_mean = y_sum / L_len

                # ss_xx = sum((x - x_mean)^2) = L*(L^2-1)/12
                ss_xx = L_len * (L_len * L_len - 1.0) / 12.0
                if ss_xx == 0.0:
                    L += step
                    continue

                ss_xy = 0.0
                for i in range(L_len):
                    ss_xy += (i - x_mean) * (prices[t + i] - y_mean)

                beta = ss_xy / ss_xx
                alpha = y_mean - beta * x_mean

                sse = 0.0
                for i in range(L_len):
                    res = prices[t + i] - (alpha + beta * i)
                    sse += res * res

                dof = L_len - 2
                if dof <= 0:
                    L += step
                    continue

                mse = sse / dof
                if mse <= 0.0:
                    L += step
                    continue

                se_beta = np.sqrt(mse / ss_xx)
                if se_beta == 0.0:
                    L += step
                    continue

                t_val = beta / se_beta
                abs_t = abs(t_val)

                if abs_t > best_abs:
                    best_abs = abs_t
                    best_t_stat = t_val
                    best_w = float(L)

                L += step

            t_stats[t] = best_t_stat
            best_windows[t] = best_w

        return t_stats, best_windows


def _trend_scan(
    prices: np.ndarray,
    min_lf: int,
    max_lf: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to numba or numpy implementation."""
    if _HAS_NUMBA:
        return _trend_scan_numba(prices, min_lf, max_lf, step)
    return _trend_scan_numpy(prices, min_lf, max_lf, step)


@dataclass
@sf_component(name="trend_scanning")
class TrendScanningLabeler(Labeler):
    """Label bars using De Prado's trend scanning method.

    For each bar, fits OLS regressions over multiple forward windows and
    selects the window with the strongest t-statistic. The sign and
    magnitude of the t-statistic determine the label.

    Reference:
        De Prado, M. L. (2020). Machine Learning for Asset Managers, Ch. 5.

    Algorithm:
        1. For each bar t, for each window L in
           range(min_lookforward, max_lookforward+1, step):

           - Fit OLS: Price[t+i] = alpha + beta * i, for i=0..L-1
           - Compute t-statistic: t = beta / SE(beta)

        2. Select L* = argmax_L |t_stat(t, L)|
        3. Label:

           - ``"rise"`` if t_stat > critical_value
           - ``"fall"`` if t_stat < -critical_value
           - ``null`` otherwise

    Attributes:
        price_col: Price column. Default: ``"close"``.
        min_lookforward: Minimum forward window. Default: ``5``.
        max_lookforward: Maximum forward window. Default: ``60``.
        step: Step between window sizes. Default: ``5``.
        critical_value: T-stat threshold for significance. Default: ``1.96``.

    Example:
        ```python
        from signalflow.target.trend_scanning import TrendScanningLabeler

        labeler = TrendScanningLabeler(
            min_lookforward=5,
            max_lookforward=60,
            critical_value=1.96,
            mask_to_signals=False,
        )
        result = labeler.compute(ohlcv_df)
        ```
    """

    signal_category: SignalCategory = SignalCategory.TREND_MOMENTUM

    price_col: str = "close"
    min_lookforward: int = 5
    max_lookforward: int = 60
    step: int = 5
    critical_value: float = 1.96

    meta_columns: tuple[str, ...] = ("t_stat", "best_window")

    def __post_init__(self) -> None:
        if self.min_lookforward < 3:
            raise ValueError("min_lookforward must be >= 3 (need at least 3 points for OLS)")
        if self.max_lookforward < self.min_lookforward:
            raise ValueError("max_lookforward must be >= min_lookforward")
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if self.critical_value <= 0:
            raise ValueError("critical_value must be > 0")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Compute trend scanning labels for a single pair.

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

        prices = group_df[self.price_col].to_numpy().astype(np.float64)

        t_stats, best_windows = _trend_scan(
            prices,
            self.min_lookforward,
            self.max_lookforward,
            self.step,
        )

        # Assign labels based on t-statistic vs critical value
        n = len(prices)
        labels = [None] * n
        for t in range(n):
            if np.isnan(t_stats[t]):
                continue
            if t_stats[t] > self.critical_value:
                labels[t] = "rise"
            elif t_stats[t] < -self.critical_value:
                labels[t] = "fall"

        df = group_df.with_columns(pl.Series(name=self.out_col, values=labels, dtype=pl.Utf8))

        if self.include_meta:
            df = df.with_columns(
                [
                    pl.Series(name="t_stat", values=t_stats.tolist(), dtype=pl.Float64),
                    pl.Series(
                        name="best_window",
                        values=best_windows.tolist(),
                        dtype=pl.Float64,
                    ),
                ]
            )

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df
