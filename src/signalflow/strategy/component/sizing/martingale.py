"""Martingale position sizer for grid strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from signalflow.core import sf_component
from signalflow.strategy.component.sizing.base import PositionSizer, SignalContext

if TYPE_CHECKING:
    from signalflow.core import StrategyState


@dataclass
@sf_component(name="martingale_sizer")
class MartingaleSizer(PositionSizer):
    """Martingale position sizing for grid strategies.

    Increases position size with each grid level filled.
    Useful for DCA (Dollar Cost Averaging) and grid trading strategies.

    Formula: notional = base_size * (multiplier ^ grid_level)

    Where grid_level = number of existing open positions in the same pair.

    Args:
        base_size: Initial position size for first grid level.
        multiplier: Size multiplier per level (e.g., 1.5 = 50% increase).
        max_grid_levels: Maximum number of grid levels to fill.
        max_notional: Maximum position size cap.
        min_notional: Minimum trade size.

    Example:
        >>> sizer = MartingaleSizer(base_size=100, multiplier=1.5)
        >>> # Level 0: $100
        >>> # Level 1: $150
        >>> # Level 2: $225
        >>> # Level 3: $337.50

    Warning:
        Martingale can lead to large losses in trending markets.
        Always use with appropriate risk limits and max_grid_levels.
    """

    base_size: float = 100.0
    multiplier: float = 1.5
    max_grid_levels: int = 5
    max_notional: float = float("inf")
    min_notional: float = 10.0

    def compute_size(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> float:
        # Count existing positions in the same pair
        open_positions = state.portfolio.open_positions()
        grid_level = sum(1 for p in open_positions if p.pair == signal.pair)

        # Check max grid levels
        if grid_level >= self.max_grid_levels:
            return 0.0

        # Calculate size with multiplier
        notional = self.base_size * (self.multiplier**grid_level)

        if notional < self.min_notional:
            return 0.0
        return min(notional, self.max_notional)
