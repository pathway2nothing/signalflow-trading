"""Model context for strategy decision making."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from signalflow.core import Position, Signals


@dataclass(frozen=True)
class ModelContext:
    """Aggregated context passed to strategy models.

    Provides all information a model needs to make decisions:
    - Current bar signals
    - Strategy metrics (equity, drawdown, etc.)
    - Current positions and their states
    - Runtime state (for custom indicators, cooldowns, etc.)

    Attributes:
        timestamp: Current bar timestamp.
        signals: Current bar signals (Signals container).
        prices: Current prices per pair.
        positions: List of open positions.
        metrics: Current strategy metrics snapshot.
        runtime: Runtime state dict (cooldowns, custom state).

    Example:
        >>> # Model receives context each bar
        >>> def decide(self, context: ModelContext) -> list[StrategyDecision]:
        ...     # Access signals
        ...     for row in context.signals.value.iter_rows(named=True):
        ...         pair = row["pair"]
        ...         signal_type = row["signal_type"]
        ...         probability = row.get("probability", 0.5)
        ...
        ...         # Use metrics for risk management
        ...         if context.metrics.get("max_drawdown", 0) > 0.15:
        ...             continue  # Skip during high drawdown
        ...
        ...         # Check existing positions
        ...         pair_positions = [p for p in context.positions if p.pair == pair]
        ...         ...
    """

    timestamp: datetime
    signals: Signals
    prices: dict[str, float] = field(default_factory=dict)
    positions: list[Position] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
