"""Base protocol for backtest data exporters."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from signalflow.core import Signals, StrategyState


class ExporterProtocol(Protocol):
    """Protocol for backtest data exporters.

    Exporters record data during backtest execution and write
    it to files for external ML model training.
    """

    def export_bar(
        self,
        timestamp: datetime,
        signals: Signals,
        metrics: dict[str, float],
        state: StrategyState,
    ) -> None:
        """Record data for a single bar.

        Args:
            timestamp: Current bar timestamp.
            signals: Current bar signals.
            metrics: Current strategy metrics.
            state: Current strategy state.
        """
        ...

    def export_trade(
        self,
        trade_data: dict[str, Any],
    ) -> None:
        """Record a completed trade.

        Args:
            trade_data: Dictionary with trade details.
        """
        ...

    def finalize(self, output_path: Path) -> None:
        """Write all recorded data to files.

        Args:
            output_path: Directory to write output files.
        """
        ...
