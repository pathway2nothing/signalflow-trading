"""Order and OrderFill containers for strategy execution."""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

OrderSide = Literal['BUY', 'SELL']
OrderType = Literal['MARKET', 'LIMIT']
OrderStatus = Literal['NEW', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'REJECTED']


@dataclass(slots=True)
class Order:
    """Represents a trading order."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pair: str = ''
    side: OrderSide = 'BUY'
    order_type: OrderType = 'MARKET'
    qty: float = 0.0
    price: float | None = None 
    created_at: datetime | None = None
    status: OrderStatus = 'NEW'
    position_id: str | None = None  
    signal_strength: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.side == 'BUY'

    @property
    def is_sell(self) -> bool:
        return self.side == 'SELL'

    @property
    def is_market(self) -> bool:
        return self.order_type == 'MARKET'


@dataclass(frozen=True, slots=True)
class OrderFill:
    """Represents a filled order."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ''
    pair: str = ''
    side: OrderSide = 'BUY'
    ts: datetime | None = None
    price: float = 0.0
    qty: float = 0.0
    fee: float = 0.0
    position_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.price * self.qty