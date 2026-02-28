from datetime import datetime
from typing import ClassVar, Protocol

from signalflow.core import Order, OrderFill, SfComponentType


class OrderExecutor(Protocol):
    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_EXECUTOR
    """
    Protocol for order execution.

    Implementations:
        - VirtualExecutor: Simulates fills at current prices
        - LiveExecutor: Submits orders to exchange
        - BinanceExecutor(LiveExecutor): Submits orders to Binance
    """

    def execute(
        self,
        orders: list[Order],
        prices: dict[str, float],
        ts: datetime,
    ) -> list[OrderFill]:
        """
        Execute orders and return fills.

        Args:
            orders: List of orders to execute
            prices: Current prices per pair
            ts: Current timestamp

        Returns:
            List of fills (may be empty if orders rejected)
        """
        ...
