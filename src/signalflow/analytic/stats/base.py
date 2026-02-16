"""Base classes and protocols for statistical validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from signalflow.analytic.stats.results import (
        BootstrapResult,
        MonteCarloResult,
        StatisticalTestResult,
    )
    from signalflow.api.result import BacktestResult


class TradeProtocol(Protocol):
    """Protocol for trade objects with PnL."""

    @property
    def pnl(self) -> float | None:
        """Trade profit/loss."""
        ...


@dataclass
class SimulationConfig:
    """Configuration for statistical simulations.

    Attributes:
        n_simulations: Number of simulations to run
        random_seed: Random seed for reproducibility (None for random)
        confidence_levels: Tuple of confidence levels for percentiles
        n_jobs: Number of parallel jobs (future use)
    """

    n_simulations: int = 10_000
    random_seed: int | None = None
    confidence_levels: tuple[float, ...] = (0.05, 0.50, 0.95)
    n_jobs: int = 1


class StatisticalValidator(ABC):
    """Base class for statistical validation methods."""

    @abstractmethod
    def validate(self, result: BacktestResult) -> MonteCarloResult | BootstrapResult | StatisticalTestResult:
        """Run validation and return results.

        Args:
            result: BacktestResult to validate

        Returns:
            Validation result object
        """
        ...


def extract_pnls(trades: Sequence[Any]) -> np.ndarray:
    """Extract trade PnLs from various trade formats.

    Supports:
    - Objects with .pnl attribute
    - Objects with .realized_pnl attribute
    - Dictionaries with "pnl" or "realized_pnl" key

    Args:
        trades: Sequence of trade objects

    Returns:
        NumPy array of PnL values
    """
    pnls = []
    for trade in trades:
        pnl: float | None = None

        # Try different access patterns
        if hasattr(trade, "pnl"):
            pnl = getattr(trade, "pnl", None)
        elif hasattr(trade, "realized_pnl"):
            pnl = getattr(trade, "realized_pnl", None)
        elif hasattr(trade, "total_pnl"):
            pnl = getattr(trade, "total_pnl", None)
        elif isinstance(trade, dict):
            pnl = trade.get("pnl") or trade.get("realized_pnl") or trade.get("total_pnl")

        pnls.append(float(pnl) if pnl is not None else 0.0)

    return np.array(pnls, dtype=np.float64)


def extract_returns(
    result: BacktestResult,
) -> np.ndarray:
    """Extract periodic returns from BacktestResult.

    Tries to extract from metrics_df first (bar-by-bar returns),
    falls back to trade-based returns.

    Args:
        result: BacktestResult with metrics_df or trades

    Returns:
        NumPy array of period returns
    """
    # Try metrics_df first (bar-by-bar equity returns)
    if result.metrics_df is not None and "total_return" in result.metrics_df.columns:
        total_returns = result.metrics_df.get_column("total_return").to_numpy()
        # Convert cumulative to period returns
        if len(total_returns) > 1:
            # Period return = (1 + r_t) / (1 + r_{t-1}) - 1
            cumulative = 1 + total_returns
            period_returns = np.diff(cumulative) / cumulative[:-1]
            return period_returns

    # Fallback: compute from trade PnLs
    pnls = extract_pnls(result.trades)
    if len(pnls) == 0:
        return np.array([], dtype=np.float64)

    initial = result.initial_capital
    if initial <= 0:
        initial = 1.0  # Avoid division by zero

    # Simple returns based on trade PnL
    returns = pnls / initial
    return returns
