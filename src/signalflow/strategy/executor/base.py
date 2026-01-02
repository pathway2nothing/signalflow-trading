from signalflow.strategy.types import NewPositionOrder, ClosePositionOrder, Position, Trade
from typing import Protocol


class OrderExecutor(Protocol):
    """Protocol for order execution."""
    
    def execute_entry(
        self,
        order: NewPositionOrder,
        fee_rate: float,
    ) -> tuple[Position, Trade]:
        """Execute entry order, return new position and trade."""
        ...
    
    def execute_exit(
        self,
        position: Position,
        order: ClosePositionOrder,
        fee_rate: float,
    ) -> Trade:
        """Execute exit order, return trade."""
        ...
