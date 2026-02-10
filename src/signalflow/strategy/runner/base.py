from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar
from signalflow.core import SfComponentType, StrategyState, Position, Order, RawData, Signals
from signalflow.strategy.broker.base import Broker
from signalflow.strategy.component.base import EntryRule, ExitRule
from signalflow.analytic import StrategyMetric


class StrategyRunner(ABC):
    """Base class for strategy runners.

    Attributes:
        broker: Order execution broker.
        entry_rules: List of entry rules to apply.
        exit_rules: List of exit rules to apply.
        metrics: List of metrics to compute per bar.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_RUNNER
    broker: Broker
    entry_rules: list[EntryRule]
    exit_rules: list[ExitRule]
    metrics: list[StrategyMetric]

    @abstractmethod
    def run(self, raw_data: RawData, signals: Signals, state: StrategyState) -> StrategyState:
        """Run the strategy on historical or live data.

        Args:
            raw_data: OHLCV data container.
            signals: Pre-computed signals.
            state: Initial strategy state (or None for fresh start).

        Returns:
            Final strategy state after processing.
        """
        ...
