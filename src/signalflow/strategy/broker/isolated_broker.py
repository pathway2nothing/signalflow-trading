"""Isolated backtest broker for single-pair processing."""

from __future__ import annotations

from dataclasses import dataclass

from signalflow.core import executor
from signalflow.core.containers.order import Order, OrderFill
from signalflow.core.containers.position import Position
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.containers.trade import Trade
from signalflow.strategy.broker.base import Broker


@dataclass
@executor("isolated_backtest")
class IsolatedBacktestBroker(Broker):
    """Broker variant for isolated balance mode.

    Operates on a single pair only. Shares all position/cash accounting with
    :class:`Broker`; the only specialisation is a guard that rejects fills for
    any other pair.

    Attributes:
        pair: Trading pair this broker handles (locked to single pair).
    """

    pair: str = ""

    def process_fills(self, fills: list[OrderFill], orders: list[Order], state: StrategyState) -> list[Trade]:
        """Process fills after validating they all belong to ``self.pair``."""
        if self.pair:
            for fill in fills:
                if fill.pair != self.pair:
                    raise ValueError(f"Invalid pair: {fill.pair} != {self.pair}")
        return super().process_fills(fills, orders, state)

    def get_open_position(self, state: StrategyState) -> Position | None:
        """Get the open position for this pair, if any."""
        return self.get_open_position_for_pair(state, self.pair)
