"""Backtest execution modes."""

from enum import Enum


class BacktestMode(Enum):
    """Backtest execution modes.

    Attributes:
        SEQUENTIAL: Standard bar-by-bar processing (BacktestRunner)
        OPTIMIZED: Cached lookups for faster access (OptimizedBacktestRunner)
        ISOLATED: Parallel processing with isolated balance per pair
        UNLIMITED: No balance constraints for maximum speed
    """

    SEQUENTIAL = "sequential"
    OPTIMIZED = "optimized"
    ISOLATED = "isolated"
    UNLIMITED = "unlimited"
