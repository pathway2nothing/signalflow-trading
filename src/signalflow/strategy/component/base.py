from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar
from signalflow.core import SfComponentType, StrategyState, Position, Order, RawData, Signals
import plotly.graph_objects as go


@dataclass
class ExitRule(ABC):
    """Base class for exit rules."""

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_EXIT_RULE

    @abstractmethod
    def check_exits(self, positions: list[Position], prices: dict[str, float], state: StrategyState) -> list[Order]:
        """
        Check if any positions should be closed.

        Returns list of close orders.
        """
        ...


@dataclass
class EntryRule(ABC):
    """Base class for entry rules."""

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_ENTRY_RULE

    @abstractmethod
    def check_entries(self, signals: Signals, prices: dict[str, float], state: StrategyState) -> list[Order]:
        """
        Check signals and generate entry orders.

        Returns list of entry orders.
        """
        ...
