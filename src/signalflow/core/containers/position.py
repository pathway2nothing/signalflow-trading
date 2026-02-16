from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from signalflow.core.containers.trade import Trade, TradeSide
from signalflow.core.enums import PositionType


@dataclass(slots=True)
class Position:
    """Trading position aggregate.

    Mutable by design - tracks the lifecycle of a trading position through
    multiple trades (entry, partial exits, full exit).

    Position state changes through two operations:
        - mark(): Update to current market price (mark-to-market)
        - apply_trade(): Apply executed trade to position

    Attributes:
        id (str): Unique position identifier.
        is_closed (bool): Whether position is closed.
        pair (str): Trading pair (e.g. "BTCUSDT").
        position_type (PositionType): LONG or SHORT.
        signal_strength (float): Strength of initial signal (0-1).
        entry_time (datetime | None): Position entry timestamp.
        last_time (datetime | None): Last update timestamp.
        entry_price (float): Average entry price.
        last_price (float): Current/last marked price.
        qty (float): Current quantity held.
        fees_paid (float): Total fees paid.
        realized_pnl (float): Realized profit/loss.
        meta (dict[str, Any]): Additional metadata.

    Example:
        ```python
        from signalflow.core import Position, PositionType, Trade
        from datetime import datetime

        # Create position
        position = Position(
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=45000.0,
            qty=0.5,
            signal_strength=0.85,
            entry_time=datetime.now()
        )

        # Mark to market
        position.mark(ts=datetime.now(), price=46000.0)

        # Apply exit trade
        exit_trade = Trade(
            pair="BTCUSDT",
            side="SELL",
            ts=datetime.now(),
            price=46500.0,
            qty=0.5,
            fee=23.25
        )
        position.apply_trade(exit_trade)

        # Check results
        print(f"Total PnL: ${position.total_pnl:.2f}")
        print(f"Closed: {position.is_closed}")
        ```

    Note:
        Position changes ONLY through apply_trade() and mark().
        Direct attribute modification should be avoided.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_closed: bool = False

    pair: str = ""
    position_type: PositionType = PositionType.LONG
    signal_strength: float = 1.0

    entry_time: datetime | None = None
    last_time: datetime | None = None

    entry_price: float = 0.0
    last_price: float = 0.0

    qty: float = 0.0
    fees_paid: float = 0.0
    realized_pnl: float = 0.0

    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def side_sign(self) -> float:
        """Position direction multiplier.

        Returns:
            float: 1.0 for LONG, -1.0 for SHORT.

        Example:
            ```python
            long_pos = Position(position_type=PositionType.LONG)
            assert long_pos.side_sign == 1.0

            short_pos = Position(position_type=PositionType.SHORT)
            assert short_pos.side_sign == -1.0
            ```
        """
        return 1.0 if self.position_type == PositionType.LONG else -1.0

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss at current price.

        Calculated as: side_sign * (last_price - entry_price) * qty

        Returns:
            float: Unrealized PnL in currency units.

        Example:
            ```python
            position = Position(
                position_type=PositionType.LONG,
                entry_price=45000.0,
                last_price=46000.0,
                qty=0.5
            )
            # (46000 - 45000) * 0.5 = 500
            assert position.unrealized_pnl == 500.0
            ```
        """
        return self.side_sign * (self.last_price - self.entry_price) * self.qty

    @property
    def total_pnl(self) -> float:
        """Total profit/loss including fees.

        Calculated as: realized_pnl + unrealized_pnl - fees_paid

        Returns:
            float: Total PnL in currency units.

        Example:
            ```python
            position = Position(
                realized_pnl=100.0,
                fees_paid=50.0
            )
            position.mark(ts=datetime.now(), price=46000.0)
            total = position.total_pnl  # realized + unrealized - fees
            ```
        """
        return self.realized_pnl + self.unrealized_pnl - self.fees_paid

    def mark(self, *, ts: datetime, price: float) -> None:
        """Update position to current market price (mark-to-market).

        Updates last_time and last_price without affecting position size.
        Used for tracking unrealized PnL over time.

        Args:
            ts (datetime): Update timestamp.
            price (float): Current market price.

        Example:
            ```python
            # Mark position at each bar
            for bar in bars:
                position.mark(ts=bar.timestamp, price=bar.close)
                print(f"Unrealized PnL: ${position.unrealized_pnl:.2f}")
            ```
        """
        self.last_time = ts
        self.last_price = float(price)

    def apply_trade(self, trade: Trade) -> None:
        """Apply trade fill to position.

        Updates position state based on trade:
            - Increases position if trade direction matches position type
            - Decreases position (partial/full close) if direction opposes
            - Updates fees, realized PnL, and closing status

        Trade processing:
            - BUY increases LONG, SELL decreases LONG
            - SELL increases SHORT, BUY decreases SHORT
            - Entry price updated as weighted average on increases
            - Realized PnL computed on decreases

        Args:
            trade (Trade): Trade to apply. Must have qty > 0.

        Example:
            ```python
            # Open position
            entry_trade = Trade(
                pair="BTCUSDT",
                side="BUY",
                price=45000.0,
                qty=1.0,
                fee=45.0
            )
            position.apply_trade(entry_trade)

            # Partial close
            exit_trade = Trade(
                pair="BTCUSDT",
                side="SELL",
                price=46000.0,
                qty=0.5,
                fee=23.0
            )
            position.apply_trade(exit_trade)

            # Check state
            assert position.qty == 0.5
            assert position.realized_pnl == 500.0  # (46000-45000)*0.5
            assert not position.is_closed
            ```

        Note:
            Assumes trade.qty > 0. Position automatically marked as closed
            when qty reaches 0.
        """
        self.last_time = trade.ts
        self.last_price = float(trade.price)
        self.fees_paid += float(trade.fee)

        is_increase = self._is_increase(trade.side)

        if is_increase:
            self._increase(trade)
        else:
            self._decrease(trade)

    def _is_increase(self, side: TradeSide) -> bool:
        """Check if trade increases position size.

        Args:
            side (TradeSide): Trade side ("BUY" or "SELL").

        Returns:
            bool: True if trade increases position.
        """
        return (self.position_type == PositionType.LONG and side == "BUY") or (
            self.position_type == PositionType.SHORT and side == "SELL"
        )

    def _increase(self, trade: Trade) -> None:
        """Increase position size with new trade.

        Updates entry_price as weighted average of existing and new position.

        Args:
            trade (Trade): Trade to add to position.
        """
        new_qty = self.qty + trade.qty
        if new_qty <= 0:
            return

        if self.qty == 0:
            self.entry_price = trade.price
            self.entry_time = trade.ts
        else:
            self.entry_price = (self.entry_price * self.qty + trade.price * trade.qty) / new_qty

        self.qty = new_qty

    def _decrease(self, trade: Trade) -> None:
        """Decrease position size (partial or full close).

        Computes realized PnL for closed portion.
        Marks position as closed if qty reaches 0.

        Args:
            trade (Trade): Trade to close position.
        """
        close_qty = min(self.qty, trade.qty)
        if close_qty <= 0:
            return

        pnl = self.side_sign * (trade.price - self.entry_price) * close_qty
        self.realized_pnl += pnl
        self.qty -= close_qty

        if self.qty == 0:
            self.is_closed = True
