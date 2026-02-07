"""Signal strength position sizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from signalflow.core import sf_component
from signalflow.strategy.component.sizing.base import PositionSizer, SignalContext

if TYPE_CHECKING:
    from signalflow.core import StrategyState


@dataclass
@sf_component(name="signal_strength_sizer")
class SignalStrengthSizer(PositionSizer):
    """Size proportional to signal probability/strength.

    Higher confidence signals get larger positions.
    Essentially the current SignalEntryRule behavior extracted.

    Args:
        base_size: Base notional value.
        min_probability: Skip signals below this threshold.
        scale_factor: Multiplier for probability-based scaling.
        min_notional: Minimum trade size.
        max_notional: Maximum trade size.

    Example:
        >>> sizer = SignalStrengthSizer(base_size=100.0)
        >>> # Signal with probability=0.8 -> notional = 80
        >>> # Signal with probability=0.5 -> notional = 50
    """

    base_size: float = 100.0
    min_probability: float = 0.5
    scale_factor: float = 1.0
    min_notional: float = 10.0
    max_notional: float = float("inf")

    def compute_size(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> float:
        if signal.probability < self.min_probability:
            return 0.0

        notional = self.base_size * signal.probability * self.scale_factor

        if notional < self.min_notional:
            return 0.0
        return min(notional, self.max_notional)
