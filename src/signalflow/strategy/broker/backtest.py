"""Backtest broker implementation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from signalflow.core.containers.order import Order, OrderFill
from signalflow.core.containers.position import Position
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.containers.trade import Trade
from signalflow.core.decorators import sf_component
from signalflow.core.enums import PositionType, SfComponentType
from signalflow.strategy.broker.base import Broker


@dataclass
@sf_component(name="backtest", override=True)
class BacktestBroker(Broker):
    """
    Broker for backtesting - handles order execution, position management, and state persistence.

    Execution flow:
        1. Mark prices on positions
        2. Submit orders -> get fills
        3. Apply fills to positions
        4. Persist state
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_BROKER

    def create_position(self, order: Order, fill: OrderFill) -> Position:
        """Create a new position from an order fill."""
        # Determine position type from order side
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
        """
        Process fills and UPDATE CASH BALANCE.

        FIX: Properly update portfolio.cash when opening/closing positions
        """
        trades: list[Trade] = []
        order_map = {o.id: o for o in orders}

        for fill in fills:
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

    def get_open_position_for_pair(self, state: StrategyState, pair: str) -> Position | None:
        """Get open position for a specific pair, if any."""
        for pos in state.portfolio.open_positions():
            if pos.pair == pair:
                return pos
        return None

    def get_open_positions_by_pair(self, state: StrategyState) -> dict[str, Position]:
        """Get dict of pair -> open position."""
        return {pos.pair: pos for pos in state.portfolio.open_positions()}
