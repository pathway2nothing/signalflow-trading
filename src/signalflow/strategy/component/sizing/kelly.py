"""Kelly Criterion position sizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from signalflow.core import sf_component
from signalflow.strategy.component.sizing.base import PositionSizer, SignalContext

if TYPE_CHECKING:
    from signalflow.core import StrategyState


@dataclass
@sf_component(name="kelly_sizer")
class KellyCriterionSizer(PositionSizer):
    """Kelly Criterion position sizing.

    Formula: f* = (p * b - q) / b
    Where:
        p = win probability (from signal or historical)
        q = 1 - p (loss probability)
        b = win/loss ratio (payoff ratio)

    Half-Kelly (kelly_fraction=0.5) is recommended for practical use
    to reduce volatility while capturing most of the edge.

    Args:
        kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly recommended).
        min_trades_for_stats: Minimum closed trades before using historical stats.
        default_win_rate: Fallback win rate if insufficient history.
        default_payoff_ratio: Fallback payoff ratio if insufficient history.
        use_signal_probability: Use signal.probability as win rate proxy.
        min_notional: Minimum trade size.
        max_fraction: Maximum fraction of equity (safety cap).

    Example:
        >>> sizer = KellyCriterionSizer(kelly_fraction=0.5)  # Half-Kelly
        >>> # With 60% win rate and 1.5:1 payoff ratio:
        >>> # Full Kelly f* = (0.6 * 1.5 - 0.4) / 1.5 = 0.333
        >>> # Half Kelly = 0.167 = 16.7% of equity
    """

    kelly_fraction: float = 0.5
    min_trades_for_stats: int = 30
    default_win_rate: float = 0.5
    default_payoff_ratio: float = 1.0
    use_signal_probability: bool = True
    min_notional: float = 10.0
    max_fraction: float = 0.25

    def compute_size(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> float:
        win_rate, payoff_ratio = self._get_stats(signal, state)

        # Kelly formula
        q = 1 - win_rate
        if payoff_ratio <= 0:
            return 0.0

        kelly_f = (win_rate * payoff_ratio - q) / payoff_ratio

        # Apply Kelly fraction and cap
        kelly_f = max(0, kelly_f * self.kelly_fraction)
        kelly_f = min(kelly_f, self.max_fraction)

        equity = state.portfolio.equity(prices=prices)
        notional = equity * kelly_f

        return notional if notional >= self.min_notional else 0.0

    def _get_stats(
        self,
        signal: SignalContext,
        state: StrategyState,
    ) -> tuple[float, float]:
        """Get win rate and payoff ratio from state or defaults."""
        closed = [p for p in state.portfolio.positions.values() if p.is_closed]

        if len(closed) >= self.min_trades_for_stats:
            winners = [p for p in closed if p.realized_pnl > 0]
            losers = [p for p in closed if p.realized_pnl <= 0]

            win_rate = len(winners) / len(closed)

            avg_win = sum(p.realized_pnl for p in winners) / len(winners) if winners else 0
            avg_loss = abs(sum(p.realized_pnl for p in losers) / len(losers)) if losers else 1
            payoff_ratio = avg_win / avg_loss if avg_loss > 0 else self.default_payoff_ratio

            return win_rate, payoff_ratio

        # Use signal probability as win rate proxy
        if self.use_signal_probability and signal.probability > 0:
            return signal.probability, self.default_payoff_ratio

        return self.default_win_rate, self.default_payoff_ratio
