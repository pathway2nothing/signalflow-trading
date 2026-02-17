"""Central risk manager — coordinates risk limits and circuit breakers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from signalflow.core.containers.order import Order
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.decorators import sf_component
from signalflow.core.enums import SfComponentType
from signalflow.strategy.risk.limits import RiskLimit


@dataclass
class RiskCheckResult:
    """Outcome of a risk check on a batch of orders.

    Attributes:
        allowed: Whether all orders passed all limits.
        rejected_orders: Orders that were filtered out.
        passed_orders: Orders that passed all limits.
        violations: ``(limit_name, reason)`` pairs for each violation.
    """

    allowed: bool
    rejected_orders: list[Order] = field(default_factory=list)
    passed_orders: list[Order] = field(default_factory=list)
    violations: list[tuple[str, str]] = field(default_factory=list)


@dataclass
@sf_component(name="risk_manager")
class RiskManager:
    """Central risk orchestrator.

    Evaluates a set of :class:`RiskLimit` rules against proposed orders
    before they reach the broker.  Supports two modes:

    * **reject-all** (default): If any limit is violated, *all* orders
      in the batch are rejected.
    * **filter**: Each order is checked individually; only violating
      orders are removed.

    The manager also tracks a **halted** flag — when set (e.g. by a
    :class:`DailyLossLimit`), *all* orders are rejected until the halt
    is cleared.

    Attributes:
        limits: List of risk limits to apply.
        mode: ``"reject_all"`` or ``"filter"`` (per-order).
        halted: When True, all orders are rejected.

    Example::

        from signalflow.strategy.risk import (
            RiskManager,
            MaxLeverageLimit,
            DailyLossLimit,
        )

        rm = RiskManager(limits=[
            MaxLeverageLimit(max_leverage=3.0),
            DailyLossLimit(max_daily_loss_pct=0.05),
        ])

        result = rm.check(orders, state, prices, ts)
        if result.allowed:
            broker.submit_orders(result.passed_orders, prices, ts)
    """

    component_type: SfComponentType = SfComponentType.STRATEGY_RISK
    limits: list[RiskLimit] = field(default_factory=list)
    mode: str = "reject_all"  # "reject_all" | "filter"
    halted: bool = field(default=False, init=False, repr=False)
    _violation_history: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def check(
        self,
        orders: list[Order],
        state: StrategyState,
        prices: dict[str, float],
        ts: datetime,
    ) -> RiskCheckResult:
        """Evaluate all limits against proposed orders.

        Args:
            orders: Orders to validate.
            state: Current strategy state.
            prices: Current prices per pair.
            ts: Current timestamp.

        Returns:
            RiskCheckResult with passed/rejected orders and violation details.
        """
        if not orders:
            return RiskCheckResult(allowed=True, passed_orders=[], rejected_orders=[])

        if self.halted:
            return RiskCheckResult(
                allowed=False,
                rejected_orders=list(orders),
                passed_orders=[],
                violations=[("risk_manager", "trading is halted")],
            )

        if self.mode == "filter":
            return self._check_per_order(orders, state, prices, ts)

        return self._check_batch(orders, state, prices, ts)

    def _check_batch(
        self,
        orders: list[Order],
        state: StrategyState,
        prices: dict[str, float],
        ts: datetime,
    ) -> RiskCheckResult:
        """Reject entire batch if any limit is violated."""
        violations: list[tuple[str, str]] = []

        for limit in self.limits:
            if not limit.enabled:
                continue
            allowed, reason = limit.check(orders, state, prices, ts)
            if not allowed:
                name = limit.__class__.__name__
                violations.append((name, reason))
                logger.warning(f"RISK [{name}] {reason}")
                self._record_violation(name, reason, ts, orders)

        if violations:
            return RiskCheckResult(
                allowed=False,
                rejected_orders=list(orders),
                passed_orders=[],
                violations=violations,
            )

        return RiskCheckResult(allowed=True, passed_orders=list(orders), rejected_orders=[])

    def _check_per_order(
        self,
        orders: list[Order],
        state: StrategyState,
        prices: dict[str, float],
        ts: datetime,
    ) -> RiskCheckResult:
        """Check each order individually, keeping those that pass."""
        passed: list[Order] = []
        rejected: list[Order] = []
        violations: list[tuple[str, str]] = []

        for order in orders:
            order_ok = True
            for limit in self.limits:
                if not limit.enabled:
                    continue
                allowed, reason = limit.check([order], state, prices, ts)
                if not allowed:
                    order_ok = False
                    name = limit.__class__.__name__
                    violations.append((name, reason))
                    logger.warning(f"RISK [{name}] order {order.pair} rejected: {reason}")
                    self._record_violation(name, reason, ts, [order])
                    break  # first violation rejects the order

            if order_ok:
                passed.append(order)
            else:
                rejected.append(order)

        return RiskCheckResult(
            allowed=len(rejected) == 0,
            passed_orders=passed,
            rejected_orders=rejected,
            violations=violations,
        )

    def halt(self, reason: str = "") -> None:
        """Halt all trading. Call :meth:`resume` to lift."""
        self.halted = True
        logger.critical(f"RISK HALT: {reason or 'manual halt'}")

    def resume(self) -> None:
        """Resume trading after a halt."""
        self.halted = False
        logger.info("RISK RESUMED: trading resumed")

    def _record_violation(
        self,
        limit_name: str,
        reason: str,
        ts: datetime,
        orders: list[Order],
    ) -> None:
        self._violation_history.append(
            {
                "ts": ts,
                "limit": limit_name,
                "reason": reason,
                "n_orders": len(orders),
                "pairs": list({o.pair for o in orders}),
            }
        )

    @property
    def violation_history(self) -> list[dict[str, Any]]:
        return self._violation_history
