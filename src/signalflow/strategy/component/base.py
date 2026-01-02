"""Base classes for strategy components."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.core import SfComponentType, Position, Signals
from signalflow.strategy.state import StrategyState
from signalflow.strategy.types import StrategyContext, NewPositionOrder, ClosePositionOrder


@dataclass
class StrategyMetric(ABC):
    """Base class for strategy metrics.
    
    Metrics are computed FIRST in each step, making them available
    to both entry and exit rules for decision making.
    """
    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_METRIC

    @abstractmethod
    def compute(
        self, 
        *, 
        state: StrategyState, 
        context: StrategyContext
    ) -> dict[str, float]:
        """Compute metric values for current step.
        
        Args:
            state: Current strategy state with portfolio
            context: Current step context (ts, prices)
            
        Returns:
            Dictionary of metric names to values
        """
        raise NotImplementedError


@dataclass
class StrategyExitRule(ABC):
    """Base class for exit rules.
    
    Exit rules determine when to close open positions.
    They have access to computed metrics via context.metrics.
    """
    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_EXIT_RULE

    @abstractmethod
    def should_exit(
        self, 
        *, 
        position: Position, 
        state: StrategyState, 
        context: StrategyContext
    ) -> tuple[bool, str]:
        """Check if position should be closed.
        
        Args:
            position: Position to evaluate
            state: Current strategy state
            context: Current step context with metrics
            
        Returns:
            Tuple of (should_exit, reason)
        """
        raise NotImplementedError


@dataclass
class StrategyEntryRule(ABC):
    """Base class for entry rules.
    
    Entry rules generate orders for new positions based on signals.
    They have access to:
    - Current open positions (for max_positions, price distance checks)
    - Computed metrics via context.metrics
    """
    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_ENTRY_RULE

    @abstractmethod
    def build_orders(
        self, 
        *, 
        signals: pl.DataFrame, 
        state: StrategyState, 
        context: StrategyContext
    ) -> list[NewPositionOrder]:
        """Generate orders for new positions.
        
        Args:
            signals: DataFrame with signals for current timestamp.
                     Expected columns: pair, timestamp, signal_type, signal
            state: Current strategy state with portfolio
            context: Current step context with prices and metrics
            
        Returns:
            List of orders for new positions
        """
        raise NotImplementedError