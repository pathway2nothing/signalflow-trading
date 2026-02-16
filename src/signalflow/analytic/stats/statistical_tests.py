"""Statistical significance tests for trading performance.

Implements Probabilistic Sharpe Ratio (PSR) and Minimum Track Record
Length (MinTRL) based on Bailey & Lopez de Prado (2012).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from signalflow.analytic.stats.base import StatisticalValidator, extract_returns
from signalflow.analytic.stats.results import StatisticalTestResult

if TYPE_CHECKING:
    from signalflow.api.result import BacktestResult


@dataclass
class StatisticalTestsValidator(StatisticalValidator):
    """Statistical significance tests for trading performance.

    Implements:
    - Probabilistic Sharpe Ratio (PSR): P(SR > benchmark | observed data)
    - Minimum Track Record Length (MinTRL): trades needed for significance

    Based on Bailey & Lopez de Prado (2012): "The Sharpe Ratio Efficient Frontier"

    Attributes:
        sr_benchmark: Benchmark Sharpe ratio to compare against (default: 0)
        confidence_level: Required confidence level (default: 0.95)
        annualization_factor: Factor to annualize Sharpe ratio (default: sqrt(252))

    Example:
        >>> from signalflow.analytic.stats import StatisticalTestsValidator
        >>> tests = StatisticalTestsValidator(
        ...     sr_benchmark=0.5,  # Compare against SR of 0.5
        ...     confidence_level=0.95
        ... )
        >>> result = tests.validate(backtest_result)
        >>> print(f"PSR: {result.psr:.2%}")
        >>> print(f"Min trades needed: {result.min_track_record_length}")
    """

    sr_benchmark: float = 0.0
    confidence_level: float = 0.95
    annualization_factor: float = np.sqrt(252)

    def validate(self, result: BacktestResult) -> StatisticalTestResult:
        """Run statistical significance tests.

        Args:
            result: BacktestResult to analyze

        Returns:
            StatisticalTestResult with PSR and MinTRL values
        """
        returns = extract_returns(result)
        n = len(returns)

        if n < 2:
            return StatisticalTestResult(
                psr=None,
                psr_benchmark=self.sr_benchmark,
                psr_is_significant=False,
                min_track_record_length=None,
                current_track_record=n,
                track_record_sufficient=False,
            )

        # Compute observed Sharpe ratio
        sr_observed = self._compute_sharpe(returns)

        # Probabilistic Sharpe Ratio
        psr = self._probabilistic_sharpe_ratio(returns, sr_observed)

        # Minimum Track Record Length
        min_trl = self._minimum_track_record_length(returns, sr_observed)

        # Check significance
        psr_significant = psr > self.confidence_level if psr is not None else False
        track_record_sufficient = n >= min_trl if min_trl is not None else False

        return StatisticalTestResult(
            psr=psr,
            psr_benchmark=self.sr_benchmark,
            psr_is_significant=psr_significant,
            min_track_record_length=min_trl,
            current_track_record=n,
            track_record_sufficient=track_record_sufficient,
        )

    def _probabilistic_sharpe_ratio(
        self,
        returns: np.ndarray,
        sr_observed: float,
    ) -> float:
        """Compute Probabilistic Sharpe Ratio.

        PSR = P(SR > SR_benchmark | observed returns)

        Formula (Bailey & Lopez de Prado 2012):
        PSR = Φ[(SR - SR*) × √(n-1) / √(1 - γ₃×SR + (γ₄-1)/4 × SR²)]

        where:
        - Φ is the standard normal CDF
        - γ₃ = skewness
        - γ₄ = kurtosis
        - SR* = benchmark SR

        Args:
            returns: Array of period returns
            sr_observed: Observed Sharpe ratio

        Returns:
            PSR value (probability between 0 and 1)
        """
        try:
            from scipy import stats as scipy_stats
        except ImportError:
            # Simple fallback without scipy
            return self._simple_psr(returns, sr_observed)

        n = len(returns)
        if n < 3:
            return 0.5  # Uninformative

        # Compute higher moments
        gamma3 = scipy_stats.skew(returns)  # Skewness
        gamma4 = scipy_stats.kurtosis(returns, fisher=False)  # Kurtosis (not excess)

        # Standard error of Sharpe ratio
        # SE(SR) = sqrt((1 - γ₃×SR + (γ₄-1)/4 × SR²) / (n-1))
        variance_term = 1 - gamma3 * sr_observed + (gamma4 - 1) / 4 * sr_observed**2

        if variance_term < 0:
            variance_term = 1.0  # Fallback for numerical stability

        sr_std = np.sqrt(variance_term / (n - 1))

        if sr_std < 1e-10:
            return 1.0 if sr_observed > self.sr_benchmark else 0.0

        # Z-score
        z = (sr_observed - self.sr_benchmark) / sr_std

        # PSR = P(SR > benchmark) = Φ(z)
        psr = float(scipy_stats.norm.cdf(z))

        return psr

    def _simple_psr(
        self,
        returns: np.ndarray,
        sr_observed: float,
    ) -> float:
        """Simple PSR calculation without scipy (assumes normality)."""
        n = len(returns)
        if n < 2:
            return 0.5

        # Under normality assumption, SE(SR) ≈ 1/√(n)
        sr_std = 1.0 / np.sqrt(n)

        z = (sr_observed - self.sr_benchmark) / sr_std

        # Approximate normal CDF
        # Using logistic approximation: Φ(x) ≈ 1/(1 + exp(-1.7*x))
        psr = 1.0 / (1.0 + np.exp(-1.7 * z))

        return float(psr)

    def _minimum_track_record_length(
        self,
        returns: np.ndarray,
        sr_observed: float,
    ) -> int | None:
        """Compute Minimum Track Record Length.

        MinTRL = number of observations needed for SR to be statistically
        significant at the given confidence level.

        Formula:
        MinTRL = 1 + (1 - γ₃×SR + (γ₄-1)/4 × SR²) × (z_α / (SR - SR*))²

        Args:
            returns: Array of period returns
            sr_observed: Observed Sharpe ratio

        Returns:
            MinTRL (number of observations), or None if cannot be significant
        """
        if sr_observed <= self.sr_benchmark:
            return None  # Cannot be significant if SR <= benchmark

        try:
            from scipy import stats as scipy_stats

            z_alpha = scipy_stats.norm.ppf(self.confidence_level)
        except ImportError:
            # Approximate z for common confidence levels
            z_map = {0.90: 1.28, 0.95: 1.645, 0.99: 2.33}
            z_alpha = z_map.get(self.confidence_level, 1.645)

        n = len(returns)
        if n < 3:
            return None

        try:
            from scipy import stats as scipy_stats

            gamma3 = scipy_stats.skew(returns)
            gamma4 = scipy_stats.kurtosis(returns, fisher=False)
        except ImportError:
            # Assume normality (γ₃=0, γ₄=3)
            gamma3 = 0.0
            gamma4 = 3.0

        # MinTRL formula
        variance_term = 1 - gamma3 * sr_observed + (gamma4 - 1) / 4 * sr_observed**2

        if variance_term < 0:
            variance_term = 1.0

        denominator = (sr_observed - self.sr_benchmark) ** 2

        if denominator < 1e-10:
            return None

        min_trl = 1 + variance_term * (z_alpha**2) / denominator

        return int(np.ceil(min_trl))

    def _compute_sharpe(self, returns: np.ndarray) -> float:
        """Compute Sharpe ratio from returns.

        Args:
            returns: Array of period returns

        Returns:
            Sharpe ratio (not annualized)
        """
        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret < 1e-10:
            return 0.0

        return float(mean_ret / std_ret)


def statistical_tests(
    result: BacktestResult,
    sr_benchmark: float = 0.0,
    confidence_level: float = 0.95,
) -> StatisticalTestResult:
    """Run statistical significance tests on backtest results.

    Convenience function that creates and runs a StatisticalTestsValidator.

    Args:
        result: BacktestResult to analyze
        sr_benchmark: Benchmark Sharpe ratio to compare against
        confidence_level: Required confidence level

    Returns:
        StatisticalTestResult with PSR and MinTRL values

    Example:
        >>> from signalflow.analytic.stats import statistical_tests
        >>> tests = statistical_tests(result, sr_benchmark=0.5)
        >>> print(f"PSR: {tests.psr:.1%}")
        >>> print(f"Significant: {tests.psr_is_significant}")
    """
    validator = StatisticalTestsValidator(
        sr_benchmark=sr_benchmark,
        confidence_level=confidence_level,
    )
    return validator.validate(result)
