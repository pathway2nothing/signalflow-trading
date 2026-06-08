"""Backtest broker implementation."""

from __future__ import annotations

from dataclasses import dataclass

from signalflow.core import executor
from signalflow.strategy.broker.base import Broker


@dataclass
@executor("backtest")
class BacktestBroker(Broker):
    """Broker for backtesting — order execution, position management, persistence.

    Position/cash accounting is inherited from :class:`Broker` (delegating to
    ``core.eventlog.apply_fill`` with the default cash-tracking policy).

    Execution flow:
        1. Mark prices on positions
        2. Submit orders -> get fills
        3. Process fills (fold into portfolio)
        4. Persist state
    """
