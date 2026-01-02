from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Literal

import polars as pl

from signalflow.core.containers.position import Position
from signalflow.core.containers.trade import Trade

@dataclass(slots=True)
class Portfolio:
    """
    Portfolio snapshot (pure domain).
    """

    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if not p.is_closed]

    def equity(self, *, prices: dict[str, float]) -> float:
        """
        Equity = cash + signed marked value of positions.
        Executor must keep accounting consistent.
        """
        eq = self.cash
        for p in self.positions.values():
            px = prices.get(p.pair, p.last_price)
            eq += p.side_sign * px * p.qty
        return eq

    @staticmethod
    def positions_to_pl(positions: Iterable[Position]) -> pl.DataFrame:
        if not positions:
            return pl.DataFrame()
        return pl.DataFrame([
            {
                "id": p.id,
                "is_closed": p.is_closed,
                "pair": p.pair,
                "position_type": p.position_type.value,
                "signal_strength": p.signal_strength,
                "entry_time": p.entry_time,
                "last_time": p.last_time,
                "entry_price": p.entry_price,
                "last_price": p.last_price,
                "qty": p.qty,
                "fees_paid": p.fees_paid,
                "realized_pnl": p.realized_pnl,
                "meta": p.meta,
            }
            for p in positions
        ])

    @staticmethod
    def trades_to_pl(trades: Iterable[Trade]) -> pl.DataFrame:
        if not trades:
            return pl.DataFrame()
        return pl.DataFrame([
            {
                "id": t.id,
                "position_id": t.position_id,
                "pair": t.pair,
                "side": t.side,
                "ts": t.ts,
                "price": t.price,
                "qty": t.qty,
                "fee": t.fee,
                "meta": t.meta,
            }
            for t in trades
        ])