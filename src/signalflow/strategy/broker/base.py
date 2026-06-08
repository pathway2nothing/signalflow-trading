"""
SignalFlow Broker Base.

Broker is the bridge between Strategy (business logic) and execution.
It handles:
    - Order execution (backtest or live)
    - State persistence
    - Fill synchronization

Key principle: Portfolio changes ONLY through fills.
Strategy generates intents (orders), Broker executes them.
"""

from __future__ import annotations

import uuid
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

from signalflow.core import CashPolicy, Order, OrderFill, Position, StrategyState, Trade, apply_fill
from signalflow.core.enums import SfComponentType
from signalflow.data.strategy_store.base import StrategyStore
from signalflow.strategy.broker.executor.base import OrderExecutor


@dataclass
class Broker(ABC):
    """
    Base Broker class.

    Combines execution and storage. Single source of truth through fills:
    - Strategy generates orders (intents)
    - Broker executes them and returns fills
    - Fills are the ONLY way portfolio changes

    This design ensures:
    - Clean separation between intent and execution
    - Easy switch from backtest to live (just swap executor)
    - Proper state recovery on restart

    Attributes:
        executor: Order execution implementation
        store: State persistence implementation
        fee_rate: Trading fee rate (e.g., 0.001 = 0.1%)
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_BROKER
    executor: OrderExecutor
    store: StrategyStore
    fee_rate: float = 0.001
    cash_policy: CashPolicy = field(default_factory=CashPolicy)

    _pending_fills: list[OrderFill] = field(default_factory=list)

    def submit_orders(
        self,
        orders: list[Order],
        prices: dict[str, float],
        ts: datetime,
    ) -> list[OrderFill]:
        """
        Submit orders for execution.

        Args:
            orders: Orders to execute
            prices: Current prices per pair
            ts: Current timestamp

        Returns:
            List of fills from execution
        """
        if not orders:
            return []

        fills = self.executor.execute(orders, prices, ts)

        return fills

    def _get_strategy_id(self, orders: list[Order]) -> str | None:
        """Extract strategy_id from orders metadata."""
        for order in orders:
            if "strategy_id" in order.meta:
                return str(order.meta["strategy_id"])
        return None

    def sync_fills(self) -> list[OrderFill]:
        """
        Synchronize fills from external source.

        For backtest: returns empty (fills are immediate)
        For live: returns fills that arrived since last sync

        Returns:
            List of new fills
        """
        fills = list(self._pending_fills)
        self._pending_fills.clear()
        return fills

    def add_pending_fill(self, fill: OrderFill) -> None:
        """Add fill to pending queue (for live executor callbacks)."""
        self._pending_fills.append(fill)

    def persist_state(self, state: StrategyState) -> None:
        """
        Persist current strategy state.

        Called at end of each tick to save:
        - Portfolio state
        - Positions
        - Metrics
        """
        self.store.save_state(state)
        if state.last_ts and state.metrics:
            self.store.append_metrics(
                state.strategy_id,
                state.last_ts,
                state.metrics,
            )

    def restore_state(self, strategy_id: str) -> StrategyState:
        """
        Restore strategy state from storage.

        Called on startup to recover from last known state.

        Args:
            strategy_id: Strategy to restore

        Returns:
            Restored state (or fresh state if not found)
        """
        state = self.store.load_state(strategy_id)

        if state is None:
            return StrategyState(strategy_id=strategy_id)

        return state

    @staticmethod
    def _fill_to_trade(fill: OrderFill, order: Order, *, is_exit: bool) -> Trade:
        """Translate an execution fill into the canonical Trade event.

        Entry trades carry the data needed to fully reconstruct the position on
        replay (signal_strength, order/fill ids, order meta); exit trades only
        reference the affected position. See ``core.eventlog`` for the fold.
        """
        if is_exit:
            return Trade(
                id=fill.id,
                position_id=fill.position_id,
                pair=fill.pair,
                side=fill.side,
                ts=fill.ts,
                price=fill.price,
                qty=fill.qty,
                fee=fill.fee,
                meta={"type": "exit", **fill.meta},
            )
        return Trade(
            id=fill.id,
            position_id=str(uuid.uuid4()),
            pair=fill.pair,
            side=fill.side,
            ts=fill.ts,
            price=fill.price,
            qty=fill.qty,
            fee=fill.fee,
            meta={
                "type": "entry",
                "signal_strength": order.signal_strength,
                "order_id": order.id,
                "fill_id": fill.id,
                **order.meta,
                **fill.meta,
            },
        )

    def process_fills(self, fills: list[OrderFill], orders: list[Order], state: StrategyState) -> list[Trade]:
        """Fold execution fills into the portfolio and return the trade events.

        Position math and cash accounting live in ``core.eventlog.apply_fill``
        (parameterised by ``self.cash_policy``), so every broker shares one
        implementation — no duplicated fill->position logic.
        """
        trades: list[Trade] = []
        order_map = {o.id: o for o in orders}

        for fill in fills:
            order = order_map.get(fill.order_id)
            if order is None:
                continue

            is_exit = bool(fill.position_id) and fill.position_id in state.portfolio.positions
            trade = self._fill_to_trade(fill, order, is_exit=is_exit)
            apply_fill(state.portfolio, trade, policy=self.cash_policy)
            trades.append(trade)

        return trades

    def mark_positions(self, state: StrategyState, prices: dict[str, float], ts: datetime) -> None:
        """Mark all open positions to current prices (mark-to-market)."""
        for position in state.portfolio.open_positions():
            price = prices.get(position.pair)
            if price is not None and price > 0:
                position.mark(ts=ts, price=price)

    def get_open_position_for_pair(self, state: StrategyState, pair: str) -> Position | None:
        """Get the open position for a specific pair, if any."""
        for pos in state.portfolio.open_positions():
            if pos.pair == pair:
                return pos
        return None

    def get_open_positions_by_pair(self, state: StrategyState) -> dict[str, Position]:
        """Get dict of pair -> open position."""
        return {pos.pair: pos for pos in state.portfolio.open_positions()}
