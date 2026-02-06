"""Protocol for external strategy models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from signalflow.strategy.model.context import ModelContext
from signalflow.strategy.model.decision import StrategyDecision


@runtime_checkable
class StrategyModel(Protocol):
    """Protocol for external strategy models.

    External models must implement this protocol to integrate with SignalFlow.
    The model receives context (signals, metrics, positions) and returns
    a list of trading decisions.

    Implementation Notes:
        - Models are called ONCE per bar (not per signal).
        - Return empty list for "no action".
        - Multiple decisions per bar are allowed.
        - Model should be stateless (use context.runtime for state).

    Example Implementation:
        >>> class MyRLModel:
        ...     '''Reinforcement learning model for trading.'''
        ...
        ...     def __init__(self, model_path: str):
        ...         self.model = load_model(model_path)
        ...
        ...     def decide(self, context: ModelContext) -> list[StrategyDecision]:
        ...         decisions = []
        ...
        ...         # Skip if drawdown too high
        ...         if context.metrics.get("max_drawdown", 0) > 0.2:
        ...             return decisions
        ...
        ...         # Process each signal
        ...         for row in context.signals.value.iter_rows(named=True):
        ...             pair = row["pair"]
        ...             prob = row.get("probability", 0.5)
        ...
        ...             features = self._build_features(row, context.metrics)
        ...             action, confidence = self.model.predict(features)
        ...
        ...             if action == "enter" and confidence > 0.6:
        ...                 decisions.append(StrategyDecision(
        ...                     action=StrategyAction.ENTER,
        ...                     pair=pair,
        ...                     size_multiplier=min(confidence, 1.5),
        ...                     confidence=confidence,
        ...                 ))
        ...
        ...         # Check if should close any positions
        ...         for pos in context.positions:
        ...             if self._should_close(pos, context):
        ...                 decisions.append(StrategyDecision(
        ...                     action=StrategyAction.CLOSE,
        ...                     pair=pos.pair,
        ...                     position_id=pos.id,
        ...                 ))
        ...
        ...         return decisions
    """

    def decide(self, context: ModelContext) -> list[StrategyDecision]:
        """Make trading decisions based on current context.

        Args:
            context: Current bar context with signals, metrics, positions.

        Returns:
            List of trading decisions (can be empty).
        """
        ...
