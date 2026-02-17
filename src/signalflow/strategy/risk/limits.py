"""Risk limits — individual rules checked by the RiskManager."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

from signalflow.core.containers.order import Order
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.decorators import sf_component
from signalflow.core.enums import SfComponentType


@dataclass
class RiskLimit(ABC):
    """Base class for a single risk limit.

    Each limit receives the proposed orders and current state and returns
    ``(allowed, reason)`` — identical pattern to ``EntryFilter``.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_RISK
    enabled: bool = True

    @abstractmethod
    def check(
        self,
        orders: list[Order],
        state: StrategyState,
        prices: dict[str, float],
        ts: datetime,
    ) -> tuple[bool, str]:
        """Return ``(True, "")`` if orders pass, ``(False, reason)`` otherwise."""
        ...


@dataclass
@sf_component(name="risk/max_leverage")
class MaxLeverageLimit(RiskLimit):
    """Reject orders that would push leverage above a threshold.

    Attributes:
        max_leverage: Maximum allowed leverage ratio.
    """

    max_leverage: float = 3.0

    def check(
        self,
        orders: list[Order],
        state: StrategyState,
        prices: dict[str, float],
        ts: datetime,
    ) -> tuple[bool, str]:
        current = state.portfolio.leverage(prices=prices)
        if current > self.max_leverage:
            return False, f"leverage={current:.2f} exceeds max={self.max_leverage}"

        # Estimate post-trade leverage
        added_notional = sum(
            prices.get(o.pair, 0) * o.qty
            for o in orders
            if o.position_id is None  # only new entries
        )
        equity = state.portfolio.equity(prices=prices)
        if equity <= 0:
            return (len(orders) == 0), "equity <= 0"
        gross = state.portfolio.gross_exposure(prices=prices)
        projected = (gross + added_notional) / equity
        if projected > self.max_leverage:
            return False, f"projected leverage={projected:.2f} exceeds max={self.max_leverage}"
        return True, ""


@dataclass
@sf_component(name="risk/max_positions")
class MaxPositionsLimit(RiskLimit):
    """Reject orders that would exceed the total open position count.

    Attributes:
        max_positions: Maximum number of open positions.
    """

    max_positions: int = 20

    def check(
        self,
        orders: list[Order],
        state: StrategyState,
        prices: dict[str, float],
        ts: datetime,
    ) -> tuple[bool, str]:
        current = len(state.portfolio.open_positions())
        new_entries = sum(1 for o in orders if o.position_id is None)
        if current + new_entries > self.max_positions:
            return False, f"positions={current}+{new_entries} exceeds max={self.max_positions}"
        return True, ""


@dataclass
@sf_component(name="risk/pair_exposure")
class PairExposureLimit(RiskLimit):
    """Limit notional exposure per pair.

    Attributes:
        max_pair_pct: Maximum exposure per pair as fraction of equity (0.25 = 25%).
    """

    max_pair_pct: float = 0.25

    def check(
        self,
        orders: list[Order],
        state: StrategyState,
        prices: dict[str, float],
        ts: datetime,
    ) -> tuple[bool, str]:
        equity = state.portfolio.equity(prices=prices)
        if equity <= 0:
            return True, ""

        # Compute per-pair exposure including proposed orders
        pair_exposure: dict[str, float] = {}
        for p in state.portfolio.open_positions():
            px = prices.get(p.pair, p.last_price)
            pair_exposure[p.pair] = pair_exposure.get(p.pair, 0.0) + px * p.qty

        for o in orders:
            if o.position_id is None:  # new entry
                px = prices.get(o.pair, 0)
                pair_exposure[o.pair] = pair_exposure.get(o.pair, 0.0) + px * o.qty

        limit = self.max_pair_pct * equity
        for pair, exposure in pair_exposure.items():
            if exposure > limit:
                return False, f"{pair} exposure={exposure:.0f} exceeds {self.max_pair_pct:.0%} of equity ({limit:.0f})"
        return True, ""


@dataclass
@sf_component(name="risk/daily_loss")
class DailyLossLimit(RiskLimit):
    """Circuit breaker — halt all trading after exceeding daily loss.

    Attributes:
        max_daily_loss_pct: Maximum daily loss as fraction of starting equity (0.05 = 5%).
    """

    max_daily_loss_pct: float = 0.05
    _day_start_equity: float = field(default=0.0, init=False, repr=False)
    _current_day: str = field(default="", init=False, repr=False)
    _halted: bool = field(default=False, init=False, repr=False)

    def check(
        self,
        orders: list[Order],
        state: StrategyState,
        prices: dict[str, float],
        ts: datetime,
    ) -> tuple[bool, str]:
        day_str = ts.strftime("%Y-%m-%d")

        # Reset at day boundary
        if day_str != self._current_day:
            self._current_day = day_str
            self._day_start_equity = state.portfolio.equity(prices=prices)
            self._halted = False

        if self._halted:
            return False, f"trading halted for {day_str} due to daily loss limit"

        if self._day_start_equity <= 0:
            return True, ""

        current_equity = state.portfolio.equity(prices=prices)
        daily_loss = (self._day_start_equity - current_equity) / self._day_start_equity

        if daily_loss >= self.max_daily_loss_pct:
            self._halted = True
            return False, (
                f"daily loss {daily_loss:.2%} exceeds max {self.max_daily_loss_pct:.2%} "
                f"(start={self._day_start_equity:.0f}, now={current_equity:.0f})"
            )
        return True, ""
