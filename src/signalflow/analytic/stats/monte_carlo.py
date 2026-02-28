"""Monte Carlo simulation for backtest validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from signalflow.analytic.stats._numba_kernels import simulate_equity_curves
from signalflow.analytic.stats.base import StatisticalValidator, extract_pnls
from signalflow.analytic.stats.results import MonteCarloResult

if TYPE_CHECKING:
    from signalflow.api.result import BacktestResult


@dataclass
class MonteCarloSimulator(StatisticalValidator):
    """Monte Carlo simulation via trade shuffling.

    Randomizes trade execution order to estimate distribution of outcomes
    under different trade sequences. This helps assess strategy robustness
    and estimate risk metrics like probability of ruin.

    Attributes:
        n_simulations: Number of simulations to run (default: 10,000)
        random_seed: Random seed for reproducibility (None for random)
        confidence_levels: Percentile levels to compute (default: 5%, 50%, 95%)
        ruin_threshold: Max drawdown threshold for risk of ruin (default: 20%)

    Example:
        >>> from signalflow.analytic.stats import MonteCarloSimulator
        >>> mc = MonteCarloSimulator(n_simulations=10_000, ruin_threshold=0.30)
        >>> mc_result = mc.validate(backtest_result)
        >>> print(mc_result.summary())
        >>> mc_result.plot()

    Note:
        This simulation shuffles trade order but keeps trade PnLs unchanged.
        It answers: "What if these same trades occurred in a different order?"
    """

    n_simulations: int = 10_000
    random_seed: int | None = None
    confidence_levels: tuple[float, ...] = (0.05, 0.50, 0.95)
    ruin_threshold: float = 0.20

    def validate(self, result: BacktestResult) -> MonteCarloResult:
        """Run Monte Carlo simulation on backtest trades.

        Args:
            result: BacktestResult containing trades to simulate

        Returns:
            MonteCarloResult with simulation distributions and risk metrics

        Raises:
            ValueError: If no trades available for simulation
        """
        # Extract trade PnLs
        pnls = extract_pnls(result.trades)

        if len(pnls) == 0:
            raise ValueError("No trades available for Monte Carlo simulation")

        initial_capital = result.initial_capital
        if initial_capital <= 0:
            initial_capital = 10_000.0  # Default fallback

        seed = self.random_seed if self.random_seed is not None else 42

        # Run simulation using Numba-accelerated kernel
        (
            final_equities,
            max_drawdowns,
            max_consec_losses,
            longest_dd_durations,
        ) = simulate_equity_curves(
            pnls=pnls,
            initial_capital=initial_capital,
            n_simulations=self.n_simulations,
            seed=seed,
        )

        # Compute percentiles
        equity_percentiles = {p: float(np.percentile(final_equities, p * 100)) for p in self.confidence_levels}
        drawdown_percentiles = {p: float(np.percentile(max_drawdowns, p * 100)) for p in self.confidence_levels}

        # Risk of ruin: probability of hitting ruin threshold
        risk_of_ruin = float(np.mean(max_drawdowns > self.ruin_threshold))

        # Get original metrics from backtest
        original_final_equity = result.final_capital
        original_max_drawdown = result.metrics.get("max_drawdown", 0.0)

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            final_equity_dist=final_equities,
            max_drawdown_dist=max_drawdowns,
            max_consecutive_losses_dist=max_consec_losses,
            longest_drawdown_duration_dist=longest_dd_durations,
            equity_percentiles=equity_percentiles,
            drawdown_percentiles=drawdown_percentiles,
            risk_of_ruin=risk_of_ruin,
            ruin_threshold=self.ruin_threshold,
            expected_max_drawdown=float(np.mean(max_drawdowns)),
            expected_worst_equity=float(np.percentile(final_equities, 5)),
            original_final_equity=original_final_equity,
            original_max_drawdown=original_max_drawdown,
        )


def monte_carlo(
    result: BacktestResult,
    n_simulations: int = 10_000,
    ruin_threshold: float = 0.20,
    random_seed: int | None = None,
    confidence_levels: tuple[float, ...] = (0.05, 0.50, 0.95),
) -> MonteCarloResult:
    """Run Monte Carlo simulation on backtest result.

    Convenience function that creates and runs a MonteCarloSimulator.

    Args:
        result: BacktestResult to validate
        n_simulations: Number of simulations to run
        ruin_threshold: Max drawdown threshold for risk of ruin
        random_seed: Random seed for reproducibility
        confidence_levels: Percentile levels to compute

    Returns:
        MonteCarloResult with simulation distributions and risk metrics

    Example:
        >>> from signalflow.analytic.stats import monte_carlo
        >>> mc = monte_carlo(result, n_simulations=5000)
        >>> print(f"Risk of Ruin: {mc.risk_of_ruin:.1%}")
    """
    simulator = MonteCarloSimulator(
        n_simulations=n_simulations,
        random_seed=random_seed,
        confidence_levels=confidence_levels,
        ruin_threshold=ruin_threshold,
    )
    return simulator.validate(result)
