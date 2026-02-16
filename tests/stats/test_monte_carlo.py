"""Tests for Monte Carlo simulation."""

from __future__ import annotations

import numpy as np

from signalflow.analytic.stats import MonteCarloSimulator, monte_carlo
from signalflow.analytic.stats._numba_kernels import (
    _compute_equity_metrics,
    _fisher_yates_shuffle,
    simulate_equity_curves,
)


class TestNumbaKernels:
    """Test Numba-accelerated kernels."""

    def test_fisher_yates_shuffle(self):
        """Test shuffle produces valid permutation."""
        arr = np.arange(10, dtype=np.float64)
        original = arr.copy()

        _fisher_yates_shuffle(arr, seed=42)

        # Should be a permutation (same elements, different order)
        assert set(arr) == set(original)
        # Should be shuffled (very unlikely to be same order)
        assert not np.array_equal(arr, original)

    def test_fisher_yates_shuffle_deterministic(self):
        """Test shuffle is deterministic with same seed."""
        arr1 = np.arange(10, dtype=np.float64)
        arr2 = np.arange(10, dtype=np.float64)

        _fisher_yates_shuffle(arr1, seed=42)
        _fisher_yates_shuffle(arr2, seed=42)

        assert np.array_equal(arr1, arr2)

    def test_compute_equity_metrics_winning(self):
        """Test metrics computation with all winning trades."""
        pnls = np.array([100.0, 50.0, 75.0, 25.0])
        initial = 1000.0

        equity, max_dd, consec_losses, dd_duration = _compute_equity_metrics(pnls, initial)

        assert equity == 1250.0  # 1000 + 250
        assert max_dd == 0.0  # No drawdown with all wins
        assert consec_losses == 0
        assert dd_duration == 0

    def test_compute_equity_metrics_losing(self):
        """Test metrics computation with all losing trades."""
        pnls = np.array([-100.0, -50.0, -75.0, -25.0])
        initial = 1000.0

        equity, max_dd, consec_losses, dd_duration = _compute_equity_metrics(pnls, initial)

        assert equity == 750.0  # 1000 - 250
        assert max_dd > 0.0  # Should have drawdown
        assert consec_losses == 4  # All losses
        assert dd_duration == 4

    def test_compute_equity_metrics_mixed(self):
        """Test metrics computation with mixed trades."""
        pnls = np.array([100.0, -50.0, -30.0, 80.0, -20.0])
        initial = 1000.0

        equity, max_dd, consec_losses, dd_duration = _compute_equity_metrics(pnls, initial)

        assert equity == 1080.0  # 1000 + 80
        assert max_dd > 0.0
        assert consec_losses == 2  # -50, -30 streak

    def test_simulate_equity_curves(self, sample_pnls):
        """Test full simulation."""
        n_simulations = 100

        equities, drawdowns, consec_losses, dd_durations = simulate_equity_curves(
            pnls=sample_pnls,
            initial_capital=10_000.0,
            n_simulations=n_simulations,
            seed=42,
        )

        assert len(equities) == n_simulations
        assert len(drawdowns) == n_simulations
        assert len(consec_losses) == n_simulations
        assert len(dd_durations) == n_simulations

        # All simulations should produce valid results
        assert np.all(np.isfinite(equities))
        assert np.all(drawdowns >= 0)
        assert np.all(drawdowns <= 1)  # Drawdown as fraction
        assert np.all(consec_losses >= 0)
        assert np.all(dd_durations >= 0)


class TestMonteCarloSimulator:
    """Test MonteCarloSimulator class."""

    def test_validate_basic(self, mock_backtest_result):
        """Test basic validation."""
        mc = MonteCarloSimulator(n_simulations=100, random_seed=42)
        result = mc.validate(mock_backtest_result)

        assert result.n_simulations == 100
        assert len(result.final_equity_dist) == 100
        assert len(result.max_drawdown_dist) == 100

    def test_validate_risk_of_ruin(self, mock_backtest_result):
        """Test risk of ruin calculation."""
        mc = MonteCarloSimulator(
            n_simulations=1000,
            ruin_threshold=0.10,  # 10% threshold
            random_seed=42,
        )
        result = mc.validate(mock_backtest_result)

        assert 0.0 <= result.risk_of_ruin <= 1.0
        assert result.ruin_threshold == 0.10

    def test_validate_percentiles(self, mock_backtest_result):
        """Test percentile calculations."""
        mc = MonteCarloSimulator(
            n_simulations=1000,
            confidence_levels=(0.05, 0.25, 0.50, 0.75, 0.95),
            random_seed=42,
        )
        result = mc.validate(mock_backtest_result)

        # Check all percentiles present
        assert 0.05 in result.equity_percentiles
        assert 0.50 in result.equity_percentiles
        assert 0.95 in result.equity_percentiles

        # Percentiles should be ordered
        assert result.equity_percentiles[0.05] <= result.equity_percentiles[0.50]
        assert result.equity_percentiles[0.50] <= result.equity_percentiles[0.95]

    def test_validate_reproducible(self, mock_backtest_result):
        """Test results are reproducible with same seed."""
        mc1 = MonteCarloSimulator(n_simulations=100, random_seed=42)
        mc2 = MonteCarloSimulator(n_simulations=100, random_seed=42)

        result1 = mc1.validate(mock_backtest_result)
        result2 = mc2.validate(mock_backtest_result)

        np.testing.assert_array_equal(result1.final_equity_dist, result2.final_equity_dist)

    def test_validate_different_seeds(self, mock_backtest_result):
        """Test different seeds produce different results."""
        mc1 = MonteCarloSimulator(n_simulations=100, random_seed=42)
        mc2 = MonteCarloSimulator(n_simulations=100, random_seed=123)

        result1 = mc1.validate(mock_backtest_result)
        result2 = mc2.validate(mock_backtest_result)

        # Results should be different (very unlikely to be same)
        assert not np.array_equal(result1.final_equity_dist, result2.final_equity_dist)

    def test_summary(self, mock_backtest_result):
        """Test summary generation."""
        mc = MonteCarloSimulator(n_simulations=100, random_seed=42)
        result = mc.validate(mock_backtest_result)

        summary = result.summary()
        assert "Monte Carlo" in summary
        assert "Risk of Ruin" in summary
        assert "Simulations: 100" in summary

    def test_to_dataframe(self, mock_backtest_result):
        """Test DataFrame export."""
        mc = MonteCarloSimulator(n_simulations=100, random_seed=42)
        result = mc.validate(mock_backtest_result)

        df = result.to_dataframe()
        assert df.height == 100
        assert "final_equity" in df.columns
        assert "max_drawdown" in df.columns


class TestMonteCarloConvenience:
    """Test convenience function."""

    def test_monte_carlo_function(self, mock_backtest_result):
        """Test monte_carlo convenience function."""
        result = monte_carlo(
            mock_backtest_result,
            n_simulations=100,
            ruin_threshold=0.15,
            random_seed=42,
        )

        assert result.n_simulations == 100
        assert result.ruin_threshold == 0.15
