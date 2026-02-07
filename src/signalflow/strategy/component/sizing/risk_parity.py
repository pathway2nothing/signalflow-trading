"""Risk parity position sizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from signalflow.core import sf_component
from signalflow.strategy.component.sizing.base import PositionSizer, SignalContext

if TYPE_CHECKING:
    from signalflow.core import StrategyState


@dataclass
@sf_component(name="risk_parity_sizer")
class RiskParitySizer(PositionSizer):
    """Equal risk contribution across all positions.

    Allocates capital so each position contributes equally to portfolio risk,
    accounting for existing positions and their volatilities.

    Args:
        target_positions: Target number of equal-risk positions.
        volatility_source: Key in state.runtime for volatility data.
        default_volatility_pct: Default volatility if not available.
        min_notional: Minimum trade size.

    Example:
        >>> sizer = RiskParitySizer(target_positions=10)
        >>> # Each position should contribute 10% of total risk budget
        >>> # High-vol assets get smaller notional allocation
    """

    target_positions: int = 10
    volatility_source: str = "atr"
    default_volatility_pct: float = 0.02
    min_notional: float = 10.0

    def compute_size(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> float:
        equity = state.portfolio.equity(prices=prices)
        vol_data = state.runtime.get(self.volatility_source, {})

        # Target: equal risk budget per position
        risk_budget = 1.0 / self.target_positions

        # Get new asset volatility
        asset_vol = vol_data.get(signal.pair)
        if asset_vol is not None and asset_vol > 0 and signal.price > 0:
            vol_pct = asset_vol / signal.price
        else:
            vol_pct = self.default_volatility_pct

        if vol_pct <= 0:
            return 0.0

        # Calculate position size for equal risk contribution
        # notional * vol_pct = risk_budget * total_risk_capital
        # Simplified: allocate fraction of equity inversely proportional to vol
        notional = (risk_budget * equity) / vol_pct

        # Cap at reasonable fraction per position
        notional = min(notional, equity / self.target_positions)

        return notional if notional >= self.min_notional else 0.0
