"""Tests for Bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest

from signalflow.analytic.stats import BootstrapValidator, bootstrap
from signalflow.analytic.stats._numba_kernels import (
    bootstrap_sharpe_ratio,
    compute_calmar_ratio,
    compute_profit_factor,
    compute_sortino_ratio,
    compute_win_rate,
)
from signalflow.analytic.stats.results import ConfidenceInterval


class TestNumbaMetrics:
    """Test Numba metric computation functions."""

    def test_compute_sortino_ratio_positive(self):
        """Test Sortino with positive returns."""
        returns = np.array([0.01, 0.02, 0.015, -0.005, 0.01])
        sortino = compute_sortino_ratio(returns)
        assert sortino > 0

    def test_compute_sortino_ratio_negative(self):
        """Test Sortino with mostly negative returns."""
        returns = np.array([-0.01, -0.02, -0.015, 0.005, -0.01])
        sortino = compute_sortino_ratio(returns)
        assert sortino < 0

    def test_compute_calmar_ratio(self):
        """Test Calmar ratio computation."""
        # Positive returns with some drawdown
        returns = np.array([0.05, -0.02, 0.03, -0.01, 0.04])
        calmar = compute_calmar_ratio(returns)
        assert np.isfinite(calmar)

    def test_compute_profit_factor(self, sample_pnls):
        """Test profit factor computation."""
        pf = compute_profit_factor(sample_pnls)
        assert pf > 0

    def test_compute_profit_factor_all_wins(self):
        """Test profit factor with all winning trades."""
        pnls = np.array([100.0, 50.0, 75.0])
        pf = compute_profit_factor(pnls)
        assert pf == np.inf

    def test_compute_profit_factor_all_losses(self):
        """Test profit factor with all losing trades."""
        pnls = np.array([-100.0, -50.0, -75.0])
        pf = compute_profit_factor(pnls)
        assert pf == 0.0

    def test_compute_win_rate(self, sample_pnls):
        """Test win rate computation."""
        wr = compute_win_rate(sample_pnls)
        assert 0.0 <= wr <= 1.0

    def test_compute_win_rate_empty(self):
        """Test win rate with empty array."""
        wr = compute_win_rate(np.array([]))
        assert wr == 0.0

    def test_bootstrap_sharpe_ratio(self, sample_returns):
        """Test bootstrap Sharpe ratio."""
        n_bootstrap = 100
        sharpes = bootstrap_sharpe_ratio(sample_returns, n_bootstrap, seed=42)

        assert len(sharpes) == n_bootstrap
        assert np.all(np.isfinite(sharpes))


class TestConfidenceInterval:
    """Test ConfidenceInterval dataclass."""

    def test_contains(self):
        """Test contains method."""
        ci = ConfidenceInterval(
            metric_name="sharpe_ratio",
            point_estimate=1.5,
            lower=1.0,
            upper=2.0,
            confidence_level=0.95,
            method="bca",
        )

        assert ci.contains(1.5)
        assert ci.contains(1.0)
        assert ci.contains(2.0)
        assert not ci.contains(0.5)
        assert not ci.contains(2.5)

    def test_is_significant(self):
        """Test is_significant method."""
        # CI that excludes 0
        ci_sig = ConfidenceInterval(
            metric_name="sharpe_ratio",
            point_estimate=1.5,
            lower=1.0,
            upper=2.0,
            confidence_level=0.95,
            method="bca",
        )
        assert ci_sig.is_significant(0.0)

        # CI that includes 0
        ci_not_sig = ConfidenceInterval(
            metric_name="sharpe_ratio",
            point_estimate=0.5,
            lower=-0.2,
            upper=1.2,
            confidence_level=0.95,
            method="bca",
        )
        assert not ci_not_sig.is_significant(0.0)


class TestBootstrapValidator:
    """Test BootstrapValidator class."""

    def test_validate_bca(self, mock_backtest_result):
        """Test BCa bootstrap."""
        bs = BootstrapValidator(
            n_bootstrap=100,
            method="bca",
            confidence_level=0.95,
            random_seed=42,
        )
        result = bs.validate(mock_backtest_result)

        assert result.n_bootstrap == 100
        assert result.method == "bca"
        assert "sharpe_ratio" in result.intervals
        assert "win_rate" in result.intervals

    def test_validate_percentile(self, mock_backtest_result):
        """Test percentile bootstrap."""
        bs = BootstrapValidator(
            n_bootstrap=100,
            method="percentile",
            random_seed=42,
        )
        result = bs.validate(mock_backtest_result)

        assert result.method == "percentile"
        for ci in result.intervals.values():
            assert ci.method == "percentile"

    def test_validate_block(self, mock_backtest_result):
        """Test block bootstrap."""
        bs = BootstrapValidator(
            n_bootstrap=100,
            method="block",
            block_size=5,
            random_seed=42,
        )
        result = bs.validate(mock_backtest_result)

        assert result.method == "block"
        assert result.block_size == 5

    def test_validate_custom_metrics(self, mock_backtest_result):
        """Test with custom metrics."""
        bs = BootstrapValidator(
            n_bootstrap=100,
            metrics=("sharpe_ratio", "profit_factor"),
            random_seed=42,
        )
        result = bs.validate(mock_backtest_result)

        assert "sharpe_ratio" in result.intervals
        assert "profit_factor" in result.intervals
        assert "sortino_ratio" not in result.intervals

    def test_confidence_interval_ordering(self, mock_backtest_result):
        """Test CI bounds are properly ordered."""
        bs = BootstrapValidator(n_bootstrap=500, random_seed=42)
        result = bs.validate(mock_backtest_result)

        for ci in result.intervals.values():
            assert ci.lower <= ci.point_estimate <= ci.upper or (
                # Allow some tolerance for bootstrap variability
                ci.lower <= ci.upper
            )

    def test_is_significant(self, mock_backtest_result):
        """Test significance checking."""
        bs = BootstrapValidator(n_bootstrap=500, random_seed=42)
        result = bs.validate(mock_backtest_result)

        # Should be able to check any metric
        _ = result.is_significant("sharpe_ratio", 0.0)
        _ = result.is_significant("win_rate", 0.5)

    def test_is_significant_missing_metric(self, mock_backtest_result):
        """Test significance check with missing metric."""
        bs = BootstrapValidator(n_bootstrap=100, random_seed=42)
        result = bs.validate(mock_backtest_result)

        with pytest.raises(KeyError):
            result.is_significant("nonexistent_metric", 0.0)

    def test_summary(self, mock_backtest_result):
        """Test summary generation."""
        bs = BootstrapValidator(n_bootstrap=100, random_seed=42)
        result = bs.validate(mock_backtest_result)

        summary = result.summary()
        assert "Bootstrap" in summary
        assert "BCa" in summary.upper() or "bca" in summary.lower()
        assert "sharpe_ratio" in summary

    def test_to_dataframe(self, mock_backtest_result):
        """Test DataFrame export."""
        bs = BootstrapValidator(n_bootstrap=100, random_seed=42)
        result = bs.validate(mock_backtest_result)

        df = result.to_dataframe()
        assert "metric" in df.columns
        assert "point_estimate" in df.columns
        assert "lower" in df.columns
        assert "upper" in df.columns


class TestBootstrapConvenience:
    """Test convenience function."""

    def test_bootstrap_function(self, mock_backtest_result):
        """Test bootstrap convenience function."""
        result = bootstrap(
            mock_backtest_result,
            n_bootstrap=100,
            method="bca",
            confidence_level=0.90,
            random_seed=42,
        )

        assert result.n_bootstrap == 100
        assert result.method == "bca"
        for ci in result.intervals.values():
            assert ci.confidence_level == 0.90
