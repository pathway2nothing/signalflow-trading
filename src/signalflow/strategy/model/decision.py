"""Strategy decision types for external model integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class StrategyAction(StrEnum):
    """Actions a strategy model can take.

    Values:
        ENTER: Open new position for pair (uses size_multiplier).
        SKIP: Skip this signal (do not enter).
        CLOSE: Close specific position (requires position_id).
        CLOSE_ALL: Close all positions for a pair.
        HOLD: Do nothing (no action).
    """

    ENTER = "enter"
    SKIP = "skip"
    CLOSE = "close"
    CLOSE_ALL = "close_all"
    HOLD = "hold"


@dataclass(frozen=True)
class StrategyDecision:
    """Model output for a single trading decision.

    Represents one decision from the model about whether to enter, exit,
    or hold positions. Multiple decisions can be returned per bar.

    Attributes:
        action: The action to take (ENTER, SKIP, CLOSE, CLOSE_ALL, HOLD).
        pair: Trading pair this decision applies to.
        position_id: For CLOSE action - specific position to close.
        size_multiplier: For ENTER action - multiplier on base position size (default 1.0).
        confidence: Model confidence in this decision (0-1).
        meta: Additional metadata (e.g., reason, model_name).

    Example:
        >>> # Enter decision
        >>> decision = StrategyDecision(
        ...     action=StrategyAction.ENTER,
        ...     pair="BTCUSDT",
        ...     size_multiplier=1.5,
        ...     confidence=0.85,
        ...     meta={"signal_type": "rise", "model": "rf_v2"}
        ... )

        >>> # Close specific position
        >>> decision = StrategyDecision(
        ...     action=StrategyAction.CLOSE,
        ...     pair="BTCUSDT",
        ...     position_id="pos_abc123",
        ...     confidence=0.92,
        ...     meta={"reason": "model_exit"}
        ... )

    Raises:
        ValueError: If CLOSE action is missing position_id.
        ValueError: If ENTER action has non-positive size_multiplier.
    """

    action: StrategyAction
    pair: str
    position_id: str | None = None
    size_multiplier: float = 1.0
    confidence: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate decision parameters."""
        if self.action == StrategyAction.CLOSE and self.position_id is None:
            raise ValueError("CLOSE action requires position_id")
        if self.action == StrategyAction.ENTER and self.size_multiplier <= 0:
            raise ValueError("size_multiplier must be positive for ENTER action")
