"""Isolated backtest broker for single-pair processing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar
import uuid

from signalflow.core.enums import SfComponentType, PositionType
from signalflow.core.decorators import sf_component
from signalflow.core.containers.position import Position
from signalflow.core.containers.trade import Trade
from signalflow.core.containers.order import Order, OrderFill
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.strategy.broker.base import Broker
from signalflow.strategy.broker.executor.base import OrderExecutor
from signalflow.data.strategy_store.base import StrategyStore


@dataclass
@sf_component(name="isolated_backtest", override=True)
class IsolatedBacktestBroker(Broker):
    """Broker variant for isolated balance mode.

    Operates on a single pair only, with no cross-pair position checks.
    Optimized for single-threaded per-pair execution in parallel backtest.

    Attributes:
        pair: Trading pair this broker handles (locked to single pair)
        executor: Order execution implementation
        store: State persistence implementation
        fee_rate: Trading fee rate (e.g., 0.001 = 0.1%)
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_BROKER

    pair: str = ""

    def create_position(self, order: Order, fill: OrderFill) -> Position:
        """Create a new position from an order fill."""
        position_type = PositionType.LONG if order.side == "BUY" else PositionType.SHORT

        position = Position(
            id=str(uuid.uuid4()),
            is_closed=False,
            pair=fill.pair,
            position_type=position_type,
            signal_strength=order.signal_strength,
            entry_time=fill.ts,
            last_time=fill.ts,
            entry_price=fill.price,
            last_price=fill.price,
            qty=fill.qty,
            fees_paid=fill.fee,
            realized_pnl=0.0,
            meta={"order_id": order.id, "fill_id": fill.id, **order.meta},
        )
        return position

    def apply_fill_to_position(self, position: Position, fill: OrderFill) -> None:
        """Apply a fill to an existing position."""
        trade = Trade(
            id=fill.id,
            position_id=position.id,
            pair=fill.pair,
            side=fill.side,
            ts=fill.ts,
            price=fill.price,
            qty=fill.qty,
            fee=fill.fee,
            meta=fill.meta,
        )
        position.apply_trade(trade)

    def process_fills(self, fills: list[OrderFill], orders: list[Order], state: StrategyState) -> list[Trade]:
        """Process fills and update cash balance.

        Same as BacktestBroker but validates all fills are for self.pair.
        """
        trades: list[Trade] = []
        order_map = {o.id: o for o in orders}

        for fill in fills:
            if self.pair and fill.pair != self.pair:
                raise ValueError(f"Invalid pair: {fill.pair} != {self.pair}")

            order = order_map.get(fill.order_id)
            if order is None:
                continue

            notional = fill.price * fill.qty

            if fill.position_id and fill.position_id in state.portfolio.positions:
                position = state.portfolio.positions[fill.position_id]
                self.apply_fill_to_position(position, fill)

                if fill.side == "SELL":
                    state.portfolio.cash += notional - fill.fee
                elif fill.side == "BUY":
                    state.portfolio.cash -= notional + fill.fee

                trade = Trade(
                    id=fill.id,
                    position_id=position.id,
                    pair=fill.pair,
                    side=fill.side,
                    ts=fill.ts,
                    price=fill.price,
                    qty=fill.qty,
                    fee=fill.fee,
                    meta={"type": "exit", **fill.meta},
                )
                trades.append(trade)

            else:
                position = self.create_position(order, fill)
                state.portfolio.positions[position.id] = position

                if fill.side == "BUY":
                    state.portfolio.cash -= notional + fill.fee
                elif fill.side == "SELL":
                    state.portfolio.cash += notional - fill.fee

                trade = Trade(
                    id=fill.id,
                    position_id=position.id,
                    pair=fill.pair,
                    side=fill.side,
                    ts=fill.ts,
                    price=fill.price,
                    qty=fill.qty,
                    fee=fill.fee,
                    meta={"type": "entry", **fill.meta},
                )
                trades.append(trade)

        return trades

    def mark_positions(self, state: StrategyState, prices: dict[str, float], ts: datetime) -> None:
        """Mark all open positions to current prices."""
        for position in state.portfolio.open_positions():
            price = prices.get(position.pair)
            if price is not None and price > 0:
                position.mark(ts=ts, price=price)

    def get_open_position(self, state: StrategyState) -> Position | None:
        """Get the open position for this pair, if any."""
        for pos in state.portfolio.open_positions():
            if pos.pair == self.pair:
                return pos
        return None
