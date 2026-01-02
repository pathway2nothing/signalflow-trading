"""Strategy type definitions."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from signalflow.core.enums import PositionType

Symbol = str
StrategyId = str


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Context passed to all strategy components during a step.
    
    Attributes:
        strategy_id: Unique identifier for the strategy
        ts: Current timestamp
        prices: Current prices for all symbols {symbol: price}
        metrics: Computed metrics for current step (available in entry/exit rules)
        runtime: Arbitrary runtime state
    """
    strategy_id: StrategyId
    ts: datetime
    prices: dict[Symbol, float]
    metrics: dict[str, float] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NewPositionOrder:
    """Order to open a new position.
    
    Attributes:
        pair: Trading pair symbol
        position_type: LONG (SHORT not supported in v0.x)
        ts: Order timestamp
        qty: Position size
        price: Entry price (fill price)
        signal_strength: Signal confidence (0.0-1.0)
        meta: Additional metadata
    """
    pair: Symbol
    position_type: PositionType
    ts: datetime
    qty: float
    price: float
    signal_strength: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClosePositionOrder:
    """Order to close an existing position.
    
    Attributes:
        position_id: ID of position to close
        ts: Order timestamp
        price: Exit price (fill price)
        reason: Exit reason (e.g., 'TP', 'SL', 'signal')
    """
    position_id: str
    ts: datetime
    price: float
    reason: str = ''