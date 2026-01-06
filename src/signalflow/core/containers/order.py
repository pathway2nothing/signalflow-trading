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
    """Trading order intent (mutable).

    Represents the intent to trade. Orders are instructions that may or may not
    be filled by the executor.

    Order lifecycle:
        NEW → PARTIALLY_FILLED → FILLED
                              → CANCELLED
                              → REJECTED

    Attributes:
        id (str): Unique order identifier.
        pair (str): Trading pair (e.g. "BTCUSDT").
        side (OrderSide): Order side - "BUY" or "SELL".
        order_type (OrderType): Order type - "MARKET" or "LIMIT".
        qty (float): Order quantity (always positive).
        price (float | None): Limit price (None for MARKET orders).
        created_at (datetime | None): Order creation timestamp.
        status (OrderStatus): Current order status.
        position_id (str | None): ID of position this order affects.
        signal_strength (float): Strength of signal triggering order (0-1).
        meta (dict[str, Any]): Additional metadata (e.g., signal info, exit reason).

    Example:
        ```python
        from signalflow.core import Order

        # Market buy order
        market_order = Order(
            pair="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            qty=0.5,
            position_id="pos_123",
            signal_strength=0.85,
            meta={"type": "entry", "signal": "sma_cross"}
        )

        # Limit sell order
        limit_order = Order(
            pair="BTCUSDT",
            side="SELL",
            order_type="LIMIT",
            qty=0.5,
            price=46000.0,
            position_id="pos_123",
            meta={"type": "exit", "reason": "take_profit"}
        )

        # Check order properties
        assert market_order.is_buy
        assert market_order.is_market
        assert limit_order.is_sell
        ```

    Note:
        Orders are mutable to allow status updates.
        Not all orders result in fills (e.g., insufficient liquidity, rejected).
    """
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
        """Check if order is a buy order.

        Returns:
            bool: True if side is "BUY".

        Example:
            ```python
            order = Order(side="BUY")
            assert order.is_buy
            ```
        """
        return self.side == 'BUY'

    @property
    def is_sell(self) -> bool:
        """Check if order is a sell order.

        Returns:
            bool: True if side is "SELL".

        Example:
            ```python
            order = Order(side="SELL")
            assert order.is_sell
            ```
        """
        return self.side == 'SELL'

    @property
    def is_market(self) -> bool:
        """Check if order is a market order.

        Returns:
            bool: True if order_type is "MARKET".

        Example:
            ```python
            order = Order(order_type="MARKET")
            assert order.is_market
            ```
        """
        return self.order_type == 'MARKET'


@dataclass(frozen=True, slots=True)
class OrderFill:
    """Order execution result (immutable).

    Represents the actual execution of an order. Maps directly to trade(s).
    OrderFill is the bridge between order intent and executed trades.

    Attributes:
        id (str): Unique fill identifier.
        order_id (str): ID of the order that was filled.
        pair (str): Trading pair.
        side (OrderSide): Fill side - "BUY" or "SELL".
        ts (datetime | None): Fill timestamp.
        price (float): Execution price.
        qty (float): Filled quantity (may differ from order qty for partial fills).
        fee (float): Transaction fee for this fill.
        position_id (str | None): ID of the position affected.
        meta (dict[str, Any]): Additional fill metadata.

    Example:
        ```python
        from signalflow.core import OrderFill
        from datetime import datetime

        # Complete fill
        fill = OrderFill(
            order_id="order_123",
            pair="BTCUSDT",
            side="BUY",
            ts=datetime.now(),
            price=45000.0,
            qty=0.5,
            fee=22.5,
            position_id="pos_123"
        )

        # Check fill details
        print(f"Notional: ${fill.notional:.2f}")
        print(f"Fee: ${fill.fee:.2f}")

        # Partial fill
        partial_fill = OrderFill(
            order_id="order_456",
            pair="BTCUSDT",
            side="BUY",
            ts=datetime.now(),
            price=45100.0,
            qty=0.3,  # Order was for 0.5
            fee=13.53,
            position_id="pos_123",
            meta={"status": "partial", "remaining": 0.2}
        )
        ```

    Note:
        OrderFill.qty may be less than Order.qty (partial fills).
        Always check fills to update accounting correctly.
    """
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
        """Calculate notional value of the fill.

        Notional = price * qty

        Returns:
            float: Notional value in currency units.

        Example:
            ```python
            fill = OrderFill(price=45000.0, qty=0.5)
            assert fill.notional == 22500.0  # 45000 * 0.5
            ```
        """
        return self.price * self.qty