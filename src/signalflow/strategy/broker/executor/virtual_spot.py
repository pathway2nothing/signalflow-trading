"""Virtual executor for backtesting - simulates order fills at current prices."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from signalflow.core.decorators import sf_component
from signalflow.core.enums import SfComponentType


@dataclass
@sf_component(name="virtual/spot", override=True)
class VirtualSpotExecutor:
    """
    Simulates order execution for backtesting.

    Fills orders instantly at the provided price with configurable slippage.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_EXECUTOR

    fee_rate: float = 0.001
    slippage_pct: float = 0.0

    def execute(self, orders: list, prices: dict[str, float], ts: datetime) -> list:
        """
        Execute orders at current prices.

        Args:
            orders: List of Order objects to execute
            prices: Dict mapping pair -> current price
            ts: Current timestamp

        Returns:
            List of OrderFill objects
        """
        from signalflow.core.containers.order import Order, OrderFill

        fills: list[OrderFill] = []

        for order in orders:
            if not isinstance(order, Order):
                continue

            price = prices.get(order.pair)
            if price is None or price <= 0:
                continue

            if order.side == "BUY":
                fill_price = price * (1 + self.slippage_pct)
            else:
                fill_price = price * (1 - self.slippage_pct)

            notional = fill_price * order.qty
            fee = notional * self.fee_rate

            fill = OrderFill(
                id=str(uuid.uuid4()),
                order_id=order.id,
                pair=order.pair,
                side=order.side,
                ts=ts,
                price=fill_price,
                qty=order.qty,
                fee=fee,
                position_id=order.position_id,
                meta={
                    "order_meta": order.meta,
                    "signal_strength": order.signal_strength,
                },
            )
            fills.append(fill)

            order.status = "FILLED"

        return fills
