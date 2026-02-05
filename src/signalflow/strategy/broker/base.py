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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from signalflow.strategy.broker.executor.base import OrderExecutor
from signalflow.data.strategy_store.base import StrategyStore
from signalflow.core import Position, Order, OrderFill, StrategyState
from signalflow.core.enums import SfComponentType
from typing import ClassVar


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

        # Persist fills
        if fills:
            strategy_id = self._get_strategy_id(orders)
            if strategy_id:
                self.store.save_fills(strategy_id, fills)

        return fills

    def _get_strategy_id(self, orders: list[Order]) -> str | None:
        """Extract strategy_id from orders metadata."""
        for order in orders:
            if "strategy_id" in order.meta:
                return order.meta["strategy_id"]
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
        self.store.save_positions(state.strategy_id, list(state.positions.values()))
        if state.last_ts and state.all_metrics:
            self.store.save_metrics(
                state.strategy_id,
                state.last_ts,
                state.all_metrics,
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

        positions = self.store.load_positions(strategy_id, open_only=False)
        state.portfolio.positions = {p.id: p for p in positions}

        return state

    @abstractmethod
    def create_position(
        self,
        order: Order,
        fill: OrderFill,
    ) -> Position:
        """
        Create new position from order and fill.

        Args:
            order: Original open order
            fill: Execution fill

        Returns:
            New Position instance
        """
        ...

    @abstractmethod
    def apply_fill_to_position(
        self,
        position: Position,
        fill: OrderFill,
    ) -> None:
        """
        Apply fill to existing position.

        Mutates position in-place.

        Args:
            position: Position to update
            fill: Fill to apply
        """
        ...
