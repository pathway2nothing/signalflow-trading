"""External model integration for SignalFlow.

This module provides:
- Protocol for external ML/RL models
- Model-aware entry and exit rules
- Context building for model decisions

Example:
    >>> from signalflow.strategy.model import (
    ...     StrategyModel,
    ...     StrategyAction,
    ...     StrategyDecision,
    ...     ModelContext,
    ...     ModelEntryRule,
    ...     ModelExitRule,
    ... )
    >>>
    >>> # Implement your model
    >>> class MyModel:
    ...     def decide(self, context: ModelContext) -> list[StrategyDecision]:
    ...         decisions = []
    ...         for row in context.signals.value.iter_rows(named=True):
    ...             if row.get("probability", 0) > 0.7:
    ...                 decisions.append(StrategyDecision(
    ...                     action=StrategyAction.ENTER,
    ...                     pair=row["pair"],
    ...                     confidence=row["probability"],
    ...                 ))
    ...         return decisions
    >>>
    >>> # Use with backtest runner
    >>> model = MyModel()
    >>> runner = BacktestRunner(
    ...     entry_rules=[ModelEntryRule(model=model)],
    ...     exit_rules=[ModelExitRule(model=model)],
    ... )
"""

from signalflow.strategy.model.context import ModelContext
from signalflow.strategy.model.decision import StrategyAction, StrategyDecision
from signalflow.strategy.model.protocol import StrategyModel
from signalflow.strategy.model.rules import ModelEntryRule, ModelExitRule

__all__ = [
    "ModelContext",
    "ModelEntryRule",
    "ModelExitRule",
    "StrategyAction",
    "StrategyDecision",
    "StrategyModel",
]
