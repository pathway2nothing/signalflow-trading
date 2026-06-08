"""Unlimited backtest broker - ignores balance constraints."""

from __future__ import annotations

from dataclasses import dataclass, field

from signalflow.core import CashPolicy, executor
from signalflow.strategy.broker.base import Broker


@dataclass
@executor("unlimited_backtest")
class UnlimitedBacktestBroker(Broker):
    """Broker that ignores balance constraints.

    Identical to :class:`BacktestBroker` except cash is never debited/credited
    (``cash_policy.track_cash = False``). Positions and trades are still tracked
    for analytics.

    Use case: Fast signal validation without capital constraints.
    """

    cash_policy: CashPolicy = field(default_factory=lambda: CashPolicy(track_cash=False))
