"""Volatility target position sizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from signalflow.core import sf_component
from signalflow.strategy.component.sizing.base import PositionSizer, SignalContext

if TYPE_CHECKING:
    from signalflow.core import StrategyState


@dataclass
@sf_component(name="volatility_target_sizer")
class VolatilityTargetSizer(PositionSizer):
    """Target specific portfolio volatility per position.

    Sizes positions to contribute equal volatility to the portfolio.
    Smaller positions in volatile assets, larger in stable ones.

    Formula: notional = (target_vol * equity) / asset_vol_pct

    Args:
        target_volatility: Target contribution to portfolio vol (e.g., 0.01 = 1%).
        volatility_source: Key in state.runtime for ATR/volatility data.
        default_volatility_pct: Default volatility if ATR not available.
        min_notional: Minimum trade size.
        max_fraction: Maximum fraction of equity per position.

    Example:
        >>> sizer = VolatilityTargetSizer(target_volatility=0.01)
        >>> # Asset with 2% daily vol -> 50% of target allocation
        >>> # Asset with 0.5% daily vol -> 200% of target allocation (capped)
    """

    target_volatility: float = 0.01
    volatility_source: str = "atr"
    default_volatility_pct: float = 0.02
    min_notional: float = 10.0
    max_fraction: float = 0.20

    def compute_size(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> float:
        # Get asset volatility from state.runtime
        vol_data = state.runtime.get(self.volatility_source, {})
        asset_vol = vol_data.get(signal.pair)

        # Calculate volatility as percentage of price
        if asset_vol is not None and asset_vol > 0 and signal.price > 0:
            vol_pct = asset_vol / signal.price
        else:
            vol_pct = self.default_volatility_pct

        if vol_pct <= 0:
            return 0.0

        equity = state.portfolio.equity(prices=prices)

        # Size to achieve target volatility contribution
        notional = (self.target_volatility * equity) / vol_pct
        notional = min(notional, equity * self.max_fraction)

        return notional if notional >= self.min_notional else 0.0
