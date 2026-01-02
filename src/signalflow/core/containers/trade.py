from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Literal

import polars as pl

TradeSide = Literal["BUY", "SELL"]

@dataclass(frozen=True, slots=True)
class Trade:
    """
    Executed trade / fill.
    Immutable domain event.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    position_id: str | None = None

    pair: str = ""
    side: TradeSide = "BUY"
    ts: datetime | None = None

    price: float = 0.0
    qty: float = 0.0
    fee: float = 0.0

    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return float(self.price) * float(self.qty)

