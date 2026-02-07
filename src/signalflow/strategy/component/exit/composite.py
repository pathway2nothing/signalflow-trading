from dataclasses import dataclass, field

from signalflow.core import ExitPriority, Order, Position, StrategyState, sf_component
from signalflow.strategy.component.base import ExitRule


@dataclass
@sf_component(name="composite_exit")
class CompositeExit(ExitRule):
    """Combines multiple exit rules with configurable priority.

    Allows layering exit strategies:
    - TP/SL as baseline protection
    - Trailing stop for profit maximization
    - Time-based exit as last resort

    Args:
        rules: List of (ExitRule, priority) tuples. Lower priority number = higher priority.
        priority_mode: How to handle multiple triggers (see ExitPriority).

    Example:
        >>> composite = CompositeExit(
        ...     rules=[
        ...         (TakeProfitStopLossExit(take_profit_pct=0.05), 1),  # Highest priority
        ...         (TrailingStopExit(trail_pct=0.03), 2),
        ...         (TimeBasedExit(max_bars=100), 3),  # Lowest priority
        ...     ],
        ...     priority_mode=ExitPriority.FIRST_TRIGGERED
        ... )
    """

    rules: list[tuple[ExitRule, int]] = field(default_factory=list)
    priority_mode: ExitPriority = ExitPriority.FIRST_TRIGGERED

    def check_exits(self, positions: list[Position], prices: dict[str, float], state: StrategyState) -> list[Order]:
        if not self.rules:
            return []

        if self.priority_mode == ExitPriority.FIRST_TRIGGERED:
            return self._first_triggered(positions, prices, state)
        elif self.priority_mode == ExitPriority.HIGHEST_PRIORITY:
            return self._highest_priority(positions, prices, state)
        elif self.priority_mode == ExitPriority.ALL_MUST_AGREE:
            return self._all_must_agree(positions, prices, state)
        else:
            return []

    def _first_triggered(
        self, positions: list[Position], prices: dict[str, float], state: StrategyState
    ) -> list[Order]:
        """Return orders from first rule that triggers for each position."""
        orders: list[Order] = []
        positions_handled: set[str] = set()

        # Evaluate rules in priority order (lower number = higher priority)
        for rule, priority in sorted(self.rules, key=lambda x: x[1]):
            rule_orders = rule.check_exits(positions, prices, state)
            for order in rule_orders:
                if order.position_id not in positions_handled:
                    # Add composite metadata
                    order.meta["composite_rule"] = rule.__class__.__name__
                    order.meta["composite_priority"] = priority
                    orders.append(order)
                    positions_handled.add(order.position_id)

        return orders

    def _highest_priority(
        self, positions: list[Position], prices: dict[str, float], state: StrategyState
    ) -> list[Order]:
        """Evaluate all rules, keep highest priority (lowest number) exit per position."""
        position_orders: dict[str, tuple[Order, int, str]] = {}  # position_id -> (order, priority, rule_name)

        for rule, priority in self.rules:
            rule_orders = rule.check_exits(positions, prices, state)
            for order in rule_orders:
                pos_id = order.position_id
                if pos_id not in position_orders or priority < position_orders[pos_id][1]:
                    position_orders[pos_id] = (order, priority, rule.__class__.__name__)

        # Add composite metadata to final orders
        orders: list[Order] = []
        for order, priority, rule_name in position_orders.values():
            order.meta["composite_rule"] = rule_name
            order.meta["composite_priority"] = priority
            orders.append(order)

        return orders

    def _all_must_agree(self, positions: list[Position], prices: dict[str, float], state: StrategyState) -> list[Order]:
        """Only exit if all rules agree (all return exit for same position)."""
        if not self.rules:
            return []

        # Collect positions each rule wants to exit
        exit_sets: list[set[str]] = []
        order_map: dict[str, Order] = {}  # position_id -> most recent order

        for rule, _ in self.rules:
            rule_orders = rule.check_exits(positions, prices, state)
            exit_set: set[str] = set()
            for order in rule_orders:
                exit_set.add(order.position_id)
                order_map[order.position_id] = order
            exit_sets.append(exit_set)

        # Find intersection - positions all rules agree on
        if not exit_sets:
            return []

        common_exits = set.intersection(*exit_sets)

        # Return orders for positions all rules agree on
        orders: list[Order] = []
        for pos_id in common_exits:
            if pos_id in order_map:
                order = order_map[pos_id]
                order.meta["composite_rule"] = "all_agreed"
                order.meta["composite_rules_count"] = len(self.rules)
                orders.append(order)

        return orders
