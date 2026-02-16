"""Bootstrap confidence interval estimation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from signalflow.analytic.stats._numba_kernels import (
    bootstrap_sharpe_ratio,
    compute_acceleration,
    compute_calmar_ratio,
    compute_profit_factor,
    compute_sortino_ratio,
    compute_win_rate,
)
from signalflow.analytic.stats.base import StatisticalValidator, extract_pnls, extract_returns
from signalflow.analytic.stats.results import BootstrapResult, ConfidenceInterval

if TYPE_CHECKING:
    from signalflow.api.result import BacktestResult


# Metric computation functions
METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}


def _sharpe_ratio(returns: np.ndarray, pnls: np.ndarray) -> float:
    """Compute Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    std = np.std(returns)
    if std < 1e-10:
        return 0.0
    return float(np.mean(returns) / std)


def _sortino_ratio(returns: np.ndarray, pnls: np.ndarray) -> float:
    """Compute Sortino ratio."""
    return float(compute_sortino_ratio(returns))


def _calmar_ratio(returns: np.ndarray, pnls: np.ndarray) -> float:
    """Compute Calmar ratio."""
    return float(compute_calmar_ratio(returns))


def _profit_factor(returns: np.ndarray, pnls: np.ndarray) -> float:
    """Compute profit factor."""
    return float(compute_profit_factor(pnls))


def _win_rate(returns: np.ndarray, pnls: np.ndarray) -> float:
    """Compute win rate."""
    return float(compute_win_rate(pnls))


METRIC_FUNCTIONS = {
    "sharpe_ratio": _sharpe_ratio,
    "sortino_ratio": _sortino_ratio,
    "calmar_ratio": _calmar_ratio,
    "profit_factor": _profit_factor,
    "win_rate": _win_rate,
}


def _block_bootstrap_indices(
    n: int,
    block_size: int,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Generate block bootstrap indices for time-series.

    Args:
        n: Length of original series
        block_size: Size of each block
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Array of shape (n_bootstrap, n) with bootstrap indices
    """
    rng = np.random.default_rng(seed)
    n_blocks = (n + block_size - 1) // block_size
    indices = np.empty((n_bootstrap, n), dtype=np.int32)

    for b in range(n_bootstrap):
        # Sample block starting points
        starts = rng.integers(0, max(1, n - block_size + 1), n_blocks)
        idx = []
        for start in starts:
            idx.extend(range(start, min(start + block_size, n)))
        indices[b, :] = idx[:n]

    return indices


@dataclass
class BootstrapValidator(StatisticalValidator):
    """Bootstrap confidence interval estimation with BCa and block support.

    Supports:
    - BCa (bias-corrected accelerated) bootstrap for general metrics
    - Percentile bootstrap for simple intervals
    - Block bootstrap for time-series data with autocorrelation

    Attributes:
        n_bootstrap: Number of bootstrap resamples (default: 5,000)
        method: Bootstrap method ("bca", "percentile", "block")
        block_size: Block size for block bootstrap (auto if None)
        confidence_level: Confidence level (default: 0.95)
        random_seed: Random seed for reproducibility
        metrics: Metrics to compute intervals for

    Example:
        >>> from signalflow.analytic.stats import BootstrapValidator
        >>> bootstrap = BootstrapValidator(
        ...     n_bootstrap=5000,
        ...     method="bca",
        ...     metrics=("sharpe_ratio", "sortino_ratio", "profit_factor")
        ... )
        >>> result = bootstrap.validate(backtest_result)
        >>> print(result.intervals["sharpe_ratio"])
    """

    n_bootstrap: int = 5_000
    method: Literal["bca", "percentile", "block"] = "bca"
    block_size: int | None = None
    confidence_level: float = 0.95
    random_seed: int | None = None
    metrics: tuple[str, ...] = (
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "profit_factor",
        "win_rate",
    )

    def validate(self, result: BacktestResult) -> BootstrapResult:
        """Run bootstrap analysis on backtest result.

        Args:
            result: BacktestResult to analyze

        Returns:
            BootstrapResult with confidence intervals for each metric
        """
        returns = extract_returns(result)
        pnls = extract_pnls(result.trades)

        seed = self.random_seed if self.random_seed is not None else 42

        intervals: dict[str, ConfidenceInterval] = {}
        distributions: dict[str, np.ndarray] = {}

        for metric in self.metrics:
            if metric not in METRIC_FUNCTIONS:
                continue

            # Compute bootstrap distribution
            dist = self._bootstrap_metric(metric, returns, pnls, seed)
            distributions[metric] = dist

            # Compute point estimate
            point_estimate = METRIC_FUNCTIONS[metric](returns, pnls)

            # Compute confidence interval
            if self.method == "bca":
                ci = self._bca_interval(dist, point_estimate, returns, pnls, metric)
            elif self.method == "block":
                ci = self._block_bootstrap_interval(metric, returns, pnls, seed)
            else:  # percentile
                ci = self._percentile_interval(dist, point_estimate, metric)

            intervals[metric] = ci

        return BootstrapResult(
            n_bootstrap=self.n_bootstrap,
            method=self.method,
            intervals=intervals,
            distributions=distributions,
            block_size=self.block_size,
        )

    def _bootstrap_metric(
        self,
        metric: str,
        returns: np.ndarray,
        pnls: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        """Generate bootstrap distribution for metric."""
        if metric == "sharpe_ratio" and len(returns) > 0:
            return bootstrap_sharpe_ratio(returns, self.n_bootstrap, seed)
        else:
            return self._generic_bootstrap(metric, returns, pnls, seed)

    def _generic_bootstrap(
        self,
        metric: str,
        returns: np.ndarray,
        pnls: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        """Generic bootstrap resampling."""
        rng = np.random.default_rng(seed)
        n = max(len(returns), len(pnls))
        if n == 0:
            return np.zeros(self.n_bootstrap, dtype=np.float64)

        results = np.empty(self.n_bootstrap, dtype=np.float64)
        metric_func = METRIC_FUNCTIONS.get(metric)

        if metric_func is None:
            return np.zeros(self.n_bootstrap, dtype=np.float64)

        for b in range(self.n_bootstrap):
            # Resample with replacement
            indices = rng.integers(0, n, n)

            sample_returns = returns[indices] if len(returns) == n else returns
            sample_pnls = pnls[indices] if len(pnls) == n else pnls

            results[b] = metric_func(sample_returns, sample_pnls)

        return results

    def _bca_interval(
        self,
        bootstrap_dist: np.ndarray,
        point_estimate: float,
        returns: np.ndarray,
        pnls: np.ndarray,
        metric: str,
    ) -> ConfidenceInterval:
        """Compute BCa (bias-corrected accelerated) confidence interval.

        BCa adjusts for bias and skewness in the bootstrap distribution.
        """
        try:
            from scipy import stats as scipy_stats
        except ImportError:
            # Fallback to percentile if scipy not available
            return self._percentile_interval(bootstrap_dist, point_estimate, metric)

        alpha = 1 - self.confidence_level

        # Bias correction: proportion of bootstrap values below point estimate
        p0 = np.mean(bootstrap_dist < point_estimate)
        if p0 == 0:
            p0 = 1 / (self.n_bootstrap + 1)
        elif p0 == 1:
            p0 = self.n_bootstrap / (self.n_bootstrap + 1)
        z0 = scipy_stats.norm.ppf(p0)

        # Acceleration (jackknife)
        n = max(len(returns), len(pnls))
        if n < 3:
            a = 0.0
        else:
            theta_i = np.empty(n, dtype=np.float64)
            metric_func = METRIC_FUNCTIONS.get(metric)

            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False

                jack_returns = returns[mask] if len(returns) == n else returns
                jack_pnls = pnls[mask] if len(pnls) == n else pnls

                if metric_func:
                    theta_i[i] = metric_func(jack_returns, jack_pnls)
                else:
                    theta_i[i] = 0.0

            a = float(compute_acceleration(theta_i))

        # Adjusted percentiles
        z_alpha = scipy_stats.norm.ppf(alpha / 2)
        z_1_alpha = scipy_stats.norm.ppf(1 - alpha / 2)

        # BCa formula for adjusted percentiles
        denom1 = 1 - a * (z0 + z_alpha)
        denom2 = 1 - a * (z0 + z_1_alpha)

        if abs(denom1) < 1e-10 or abs(denom2) < 1e-10:
            # Fallback to percentile
            return self._percentile_interval(bootstrap_dist, point_estimate, metric)

        alpha1 = scipy_stats.norm.cdf(z0 + (z0 + z_alpha) / denom1)
        alpha2 = scipy_stats.norm.cdf(z0 + (z0 + z_1_alpha) / denom2)

        # Clip to valid percentile range
        alpha1 = np.clip(alpha1, 0.001, 0.999)
        alpha2 = np.clip(alpha2, 0.001, 0.999)

        lower = float(np.percentile(bootstrap_dist, alpha1 * 100))
        upper = float(np.percentile(bootstrap_dist, alpha2 * 100))

        return ConfidenceInterval(
            metric_name=metric,
            point_estimate=point_estimate,
            lower=lower,
            upper=upper,
            confidence_level=self.confidence_level,
            method="bca",
        )

    def _percentile_interval(
        self,
        bootstrap_dist: np.ndarray,
        point_estimate: float,
        metric: str,
    ) -> ConfidenceInterval:
        """Compute simple percentile confidence interval."""
        alpha = 1 - self.confidence_level
        lower = float(np.percentile(bootstrap_dist, alpha / 2 * 100))
        upper = float(np.percentile(bootstrap_dist, (1 - alpha / 2) * 100))

        return ConfidenceInterval(
            metric_name=metric,
            point_estimate=point_estimate,
            lower=lower,
            upper=upper,
            confidence_level=self.confidence_level,
            method="percentile",
        )

    def _block_bootstrap_interval(
        self,
        metric: str,
        returns: np.ndarray,
        pnls: np.ndarray,
        seed: int,
    ) -> ConfidenceInterval:
        """Compute block bootstrap confidence interval for time-series."""
        n = max(len(returns), len(pnls))
        if n == 0:
            return ConfidenceInterval(
                metric_name=metric,
                point_estimate=0.0,
                lower=0.0,
                upper=0.0,
                confidence_level=self.confidence_level,
                method="block",
            )

        # Auto-calculate block size if not provided
        block_size = self.block_size or max(1, int(np.sqrt(n)))

        indices = _block_bootstrap_indices(n, block_size, self.n_bootstrap, seed)

        results = np.empty(self.n_bootstrap, dtype=np.float64)
        metric_func = METRIC_FUNCTIONS.get(metric)

        if metric_func is None:
            return self._percentile_interval(np.zeros(self.n_bootstrap), 0.0, metric)

        for b in range(self.n_bootstrap):
            sample_returns = returns[indices[b]] if len(returns) == n else returns
            sample_pnls = pnls[indices[b]] if len(pnls) == n else pnls
            results[b] = metric_func(sample_returns, sample_pnls)

        point_estimate = metric_func(returns, pnls)

        return self._percentile_interval(results, point_estimate, metric)


def bootstrap(
    result: BacktestResult,
    n_bootstrap: int = 5_000,
    method: Literal["bca", "percentile", "block"] = "bca",
    confidence_level: float = 0.95,
    metrics: tuple[str, ...] | None = None,
    block_size: int | None = None,
    random_seed: int | None = None,
) -> BootstrapResult:
    """Compute bootstrap confidence intervals for backtest metrics.

    Convenience function that creates and runs a BootstrapValidator.

    Args:
        result: BacktestResult to analyze
        n_bootstrap: Number of bootstrap resamples
        method: Bootstrap method ("bca", "percentile", "block")
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        metrics: Metrics to bootstrap (default: sharpe, sortino, calmar, profit_factor, win_rate)
        block_size: Block size for block bootstrap (auto if None)
        random_seed: Random seed for reproducibility

    Returns:
        BootstrapResult with confidence intervals for each metric

    Example:
        >>> from signalflow.analytic.stats import bootstrap
        >>> bs = bootstrap(result, method="bca")
        >>> print(bs.intervals["sharpe_ratio"])
    """
    if metrics is None:
        metrics = (
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "profit_factor",
            "win_rate",
        )

    validator = BootstrapValidator(
        n_bootstrap=n_bootstrap,
        method=method,
        block_size=block_size,
        confidence_level=confidence_level,
        random_seed=random_seed,
        metrics=metrics,
    )
    return validator.validate(result)
