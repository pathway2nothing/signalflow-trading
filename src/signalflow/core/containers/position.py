from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Literal

import polars as pl

from signalflow.core.enums import PositionType


@dataclass(slots=True)
class Position:
    """
    Trading position aggregate.
    Mutable by design.
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
        return 1.0 if self.position_type == PositionType.LONG else -1.0

    @property
    def unrealized_pnl(self) -> float:
        return self.side_sign * (self.last_price - self.entry_price) * self.qty

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl - self.fees_paid

    def mark(self, *, ts: datetime, price: float) -> None:
        self.last_time = ts
        self.last_price = float(price)

    def apply_trade(self, trade: Trade) -> None:
        """
        Apply trade fill to position.
        Assumptions:
        - trade.qty > 0
        - BUY increases LONG, SELL decreases LONG
        - SELL increases SHORT, BUY decreases SHORT
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
        return (
            (self.position_type == PositionType.LONG and side == "BUY")
            or (self.position_type == PositionType.SHORT and side == "SELL")
        )

    def _increase(self, trade: Trade) -> None:
        new_qty = self.qty + trade.qty
        if new_qty <= 0:
            return

        if self.qty == 0:
            self.entry_price = trade.price
            self.entry_time = trade.ts
        else:
            self.entry_price = (
                self.entry_price * self.qty + trade.price * trade.qty
            ) / new_qty

        self.qty = new_qty

    def _decrease(self, trade: Trade) -> None:
        close_qty = min(self.qty, trade.qty)
        if close_qty <= 0:
            return

        pnl = self.side_sign * (trade.price - self.entry_price) * close_qty
        self.realized_pnl += pnl
        self.qty -= close_qty

        if self.qty == 0:
            self.is_closed = True