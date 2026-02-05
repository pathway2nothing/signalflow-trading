from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar
from signalflow.core import SfComponentType, StrategyState, Position, Order, RawData, Signals
from signalflow.strategy.broker.base import Broker
from signalflow.strategy.component.base import EntryRule, ExitRule
from signalflow.analytic import StrategyMetric


class StrategyRunner(ABC):
    """Base class for strategy runners."""

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_RUNNER
    broker: Broker
    entry_rules: list[EntryRule]
    exit_rules: list[ExitRule]
    metrics: list[StrategyMetric]

    def run(self, raw_data: RawData, signals: Signals, state: StrategyState) -> StrategyState:
        """Run the strategy."""
        ...
