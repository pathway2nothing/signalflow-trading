"""Tests for signalflow.analytic.stats.results module."""

import numpy as np
import pytest

from signalflow.analytic.stats.results import (
    BootstrapResult,
    ConfidenceInterval,
    MonteCarloResult,
    StatisticalTestResult,
)


class TestConfidenceInterval:
    """Tests for ConfidenceInterval dataclass."""

    def test_contains_in_range(self):
        ci = ConfidenceInterval(
            metric_name="sharpe",
            point_estimate=1.5,
            lower=1.0,
            upper=2.0,
            confidence_level=0.95,
            method="bca",
        )
        assert ci.contains(1.5) is True
        assert ci.contains(1.0) is True
        assert ci.contains(2.0) is True

    def test_contains_out_of_range(self):
        ci = ConfidenceInterval(
            metric_name="sharpe",
            point_estimate=1.5,
            lower=1.0,
            upper=2.0,
            confidence_level=0.95,
            method="bca",
        )
        assert ci.contains(0.5) is False
        assert ci.contains(2.5) is False

    def test_is_significant_excludes_benchmark(self):
        ci = ConfidenceInterval(
            metric_name="sharpe",
            point_estimate=1.5,
            lower=1.0,
            upper=2.0,
            confidence_level=0.95,
            method="bca",
        )
        assert ci.is_significant(0.0) is True  # 0 is outside [1.0, 2.0]

    def test_is_significant_includes_benchmark(self):
        ci = ConfidenceInterval(
            metric_name="sharpe",
            point_estimate=1.5,
            lower=-0.5,
            upper=2.0,
            confidence_level=0.95,
            method="bca",
        )
        assert ci.is_significant(0.0) is False  # 0 is inside [-0.5, 2.0]


class TestMonteCarloResult:
    """Tests for MonteCarloResult dataclass."""

    @pytest.fixture
    def monte_carlo_result(self):
        np.random.seed(42)
        n = 1000
        return MonteCarloResult(
            n_simulations=n,
            final_equity_dist=np.random.normal(15000, 1000, n),
            max_drawdown_dist=np.random.beta(2, 8, n) * 0.5,
            max_consecutive_losses_dist=np.random.poisson(3, n),
            longest_drawdown_duration_dist=np.random.exponential(20, n),
            equity_percentiles={0.05: 13000, 0.50: 15000, 0.95: 17000},
            drawdown_percentiles={0.05: 0.05, 0.50: 0.15, 0.95: 0.35},
            risk_of_ruin=0.02,
            ruin_threshold=0.5,
            expected_max_drawdown=0.15,
            expected_worst_equity=13000,
            original_final_equity=15500,
            original_max_drawdown=0.12,
        )

    def test_to_dataframe(self, monte_carlo_result):
        df = monte_carlo_result.to_dataframe()
        assert df.height == 1000
        assert "final_equity" in df.columns
        assert "max_drawdown" in df.columns
        assert "max_consecutive_losses" in df.columns
        assert "longest_drawdown_duration" in df.columns

    def test_summary(self, monte_carlo_result):
        summary = monte_carlo_result.summary()
        assert "Monte Carlo" in summary
        assert "1,000" in summary  # n_simulations formatted
        assert "Final Equity" in summary
        assert "Max Drawdown" in summary
        assert "Risk of Ruin" in summary


class TestBootstrapResult:
    """Tests for BootstrapResult dataclass."""

    @pytest.fixture
    def bootstrap_result(self):
        return BootstrapResult(
            n_bootstrap=1000,
            method="bca",
            intervals={
                "sharpe": ConfidenceInterval(
                    metric_name="sharpe",
                    point_estimate=1.5,
                    lower=1.0,
                    upper=2.0,
                    confidence_level=0.95,
                    method="bca",
                ),
                "sortino": ConfidenceInterval(
                    metric_name="sortino",
                    point_estimate=2.0,
                    lower=1.5,
                    upper=2.5,
                    confidence_level=0.95,
                    method="bca",
                ),
            },
            distributions={
                "sharpe": np.random.normal(1.5, 0.3, 1000),
                "sortino": np.random.normal(2.0, 0.3, 1000),
            },
            block_size=None,
        )

    @pytest.fixture
    def bootstrap_result_with_block(self):
        return BootstrapResult(
            n_bootstrap=1000,
            method="block",
            intervals={
                "sharpe": ConfidenceInterval(
                    metric_name="sharpe",
                    point_estimate=1.5,
                    lower=-0.5,
                    upper=2.0,
                    confidence_level=0.95,
                    method="block",
                ),
            },
            distributions={
                "sharpe": np.random.normal(1.5, 0.3, 1000),
            },
            block_size=20,
        )

    def test_to_dataframe(self, bootstrap_result):
        df = bootstrap_result.to_dataframe()
        assert df.height == 2
        assert "metric" in df.columns
        assert "point_estimate" in df.columns
        assert "lower" in df.columns
        assert "upper" in df.columns

    def test_is_significant(self, bootstrap_result):
        assert bootstrap_result.is_significant("sharpe", 0.0) is True

    def test_is_significant_not_found(self, bootstrap_result):
        with pytest.raises(KeyError, match="not found"):
            bootstrap_result.is_significant("nonexistent", 0.0)

    def test_summary(self, bootstrap_result):
        summary = bootstrap_result.summary()
        assert "Bootstrap" in summary
        assert "BCA" in summary
        assert "sharpe" in summary
        assert "sortino" in summary
        assert "*" in summary  # significance marker

    def test_summary_with_block_size(self, bootstrap_result_with_block):
        summary = bootstrap_result_with_block.summary()
        assert "Block size: 20" in summary


class TestStatisticalTestResult:
    """Tests for StatisticalTestResult dataclass."""

    def test_summary_with_psr(self):
        result = StatisticalTestResult(
            psr=0.97,
            psr_benchmark=0.0,
            psr_is_significant=True,
            min_track_record_length=50,
            current_track_record=100,
            track_record_sufficient=True,
        )
        summary = result.summary()
        assert "Statistical" in summary
        assert "PSR" in summary or "Probabilistic" in summary

    def test_summary_without_psr(self):
        result = StatisticalTestResult(
            psr=None,
            psr_benchmark=0.0,
            psr_is_significant=False,
            min_track_record_length=None,
            current_track_record=1,
            track_record_sufficient=False,
        )
        summary = result.summary()
        assert "Statistical" in summary

    def test_default_values(self):
        result = StatisticalTestResult()
        assert result.psr is None
        assert result.psr_benchmark == 0.0
        assert result.psr_is_significant is False
        assert result.current_track_record == 0
