"""
High-level API for SignalFlow.

This module provides simplified interfaces for common operations:
- `Backtest`: Fluent builder for backtest configuration
- `BacktestResult`: Rich result container with analytics
- `load`: Quick data loading from exchanges or files
- `backtest`: One-liner backtest execution

Example:
    >>> import signalflow as sf
    >>>
    >>> result = (
    ...     sf.Backtest("my_strategy")
    ...     .data(exchange="binance", pairs=["BTCUSDT"], start="2024-01-01")
    ...     .detector("example/sma_cross", fast_period=20, slow_period=50)
    ...     .exit(tp=0.03, sl=0.015)
    ...     .run()
    ... )
    >>> print(result.summary())
"""

from signalflow.api.result import BacktestResult
from signalflow.api.builder import Backtest, BacktestBuilder
from signalflow.api.shortcuts import load, backtest
from signalflow.api.exceptions import (
    SignalFlowError,
    ConfigurationError,
    DataError,
    ComponentNotFoundError,
    DetectorNotFoundError,
    MissingDataError,
    MissingDetectorError,
    InvalidParameterError,
)

__all__ = [
    # Builder API
    "Backtest",
    "BacktestBuilder",
    "BacktestResult",
    # Shortcuts
    "load",
    "backtest",
    # Exceptions
    "SignalFlowError",
    "ConfigurationError",
    "DataError",
    "ComponentNotFoundError",
    "DetectorNotFoundError",
    "MissingDataError",
    "MissingDetectorError",
    "InvalidParameterError",
]
