"""Backtest data exporters for ML training.

This module provides exporters for converting backtest results
into training data for external ML models.

Example:
    >>> from signalflow.strategy.exporter import BacktestExporter
    >>>
    >>> exporter = BacktestExporter()
    >>>
    >>> # During or after backtest
    >>> for ts, signals, metrics in backtest_data:
    ...     exporter.export_bar(ts, signals, metrics, state)
    >>>
    >>> # Write to disk
    >>> exporter.finalize(Path("./training_data"))
    >>>
    >>> # Load for training
    >>> import polars as pl
    >>> bars = pl.read_parquet("./training_data/bars.parquet")
    >>> trades = pl.read_parquet("./training_data/trades.parquet")
"""

from signalflow.strategy.exporter.base import ExporterProtocol
from signalflow.strategy.exporter.parquet_exporter import BacktestExporter

__all__ = [
    "BacktestExporter",
    "ExporterProtocol",
]
