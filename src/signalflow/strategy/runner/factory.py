"""Factory function for creating backtest runners."""

from __future__ import annotations

from typing import Literal, Any

from signalflow.strategy.runner.base import StrategyRunner
from signalflow.strategy.runner.backtest_runner import BacktestRunner
from signalflow.strategy.runner.optimized_backtest_runner import OptimizedBacktestRunner
from signalflow.strategy.runner.parallel.modes import BacktestMode
from signalflow.strategy.runner.parallel.isolated_runner import IsolatedBalanceRunner
from signalflow.strategy.runner.parallel.unlimited_runner import UnlimitedBalanceRunner


def create_backtest_runner(
    mode: BacktestMode | Literal["sequential", "optimized", "isolated", "unlimited"] = BacktestMode.SEQUENTIAL,
    **kwargs: Any,
) -> StrategyRunner:
    """Create appropriate backtest runner based on mode.

    Args:
        mode: Backtest execution mode
            - sequential: Standard bar-by-bar processing (BacktestRunner)
            - optimized: Cached lookups for faster access (OptimizedBacktestRunner)
            - isolated: Parallel processing with isolated balance per pair
            - unlimited: No balance constraints for maximum speed
        **kwargs: Configuration passed to the runner

    Returns:
        Configured StrategyRunner instance

    Examples:
        >>> # Standard sequential backtest
        >>> runner = create_backtest_runner(
        ...     mode="sequential",
        ...     broker=broker,
        ...     entry_rules=[entry_rule],
        ...     exit_rules=[exit_rule],
        ...     initial_capital=10000,
        ... )
        >>> result = runner.run(raw_data, signals)

        >>> # Parallel isolated balance mode
        >>> runner = create_backtest_runner(
        ...     mode="isolated",
        ...     initial_capital=10000,
        ...     max_workers=4,
        ...     entry_rules=[entry_rule],
        ...     exit_rules=[exit_rule],
        ... )
        >>> results = runner.run(raw_data, signals)
        >>> print(results.pair_results)  # dict[pair, PairResult]

        >>> # Unlimited balance for signal validation
        >>> runner = create_backtest_runner(
        ...     mode="unlimited",
        ...     position_size=1.0,
        ...     take_profit_pct=0.02,
        ...     stop_loss_pct=0.01,
        ...     entry_rules=[entry_rule],
        ... )
        >>> results = runner.run(raw_data, signals)
        >>> print(f"Win rate: {results.win_rate:.2%}")
    """
    if isinstance(mode, str):
        mode = BacktestMode(mode)

    match mode:
        case BacktestMode.SEQUENTIAL:
            return BacktestRunner(**kwargs)
        case BacktestMode.OPTIMIZED:
            return OptimizedBacktestRunner(**kwargs)
        case BacktestMode.ISOLATED:
            return IsolatedBalanceRunner(**kwargs)
        case BacktestMode.UNLIMITED:
            return UnlimitedBalanceRunner(**kwargs)
        case _:
            raise ValueError(f"Unknown backtest mode: {mode}")
