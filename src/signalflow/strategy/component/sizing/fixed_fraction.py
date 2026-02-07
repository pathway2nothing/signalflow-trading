"""Fixed fraction position sizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from signalflow.core import sf_component
from signalflow.strategy.component.sizing.base import PositionSizer, SignalContext

if TYPE_CHECKING:
    from signalflow.core import StrategyState


@dataclass
@sf_component(name="fixed_fraction_sizer")
class FixedFractionSizer(PositionSizer):
    """Fixed percentage of equity per trade.

    Classic position sizing: risk a fixed fraction of current equity.
    Simple and consistent regardless of signal strength or volatility.

    Args:
        fraction: Fraction of equity to allocate (e.g., 0.02 = 2%).
        min_notional: Minimum trade size (skip if below).
        max_notional: Maximum trade size cap.

    Example:
        >>> sizer = FixedFractionSizer(fraction=0.02)  # 2% per trade
        >>> # With $10,000 equity: notional = $200
    """

    fraction: float = 0.02
    min_notional: float = 10.0
    max_notional: float = float("inf")

    def compute_size(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> float:
        equity = state.portfolio.equity(prices=prices)
        notional = equity * self.fraction

        if notional < self.min_notional:
            return 0.0
        return min(notional, self.max_notional)
