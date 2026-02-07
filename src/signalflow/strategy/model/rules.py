"""Model-aware entry and exit rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import polars as pl

from signalflow.core import Order, PositionType, Signals, SignalType, sf_component
from signalflow.strategy.component.base import EntryRule, ExitRule
from signalflow.strategy.model.context import ModelContext
from signalflow.strategy.model.decision import StrategyAction, StrategyDecision

if TYPE_CHECKING:
    from signalflow.core import Position, StrategyState
    from signalflow.strategy.model.protocol import StrategyModel

OrderSide = Literal["BUY", "SELL"]

# Cache keys for state.runtime
DECISIONS_CACHE_KEY = "_model_decisions"
BAR_SIGNALS_KEY = "_bar_signals"


def _build_model_context(
    signals: Signals,
    prices: dict[str, float],
    state: StrategyState,
) -> ModelContext:
    """Build ModelContext from current bar state."""
    # Use current time if last_ts not set (shouldn't happen in normal backtest)
    ts = state.last_ts if state.last_ts is not None else datetime.now()
    return ModelContext(
        timestamp=ts,
        signals=signals,
        prices=prices,
        positions=state.portfolio.open_positions(),
        metrics=state.metrics.copy() if state.metrics else {},
        runtime={k: v for k, v in state.runtime.items() if not k.startswith("_")},
    )


def _cache_decisions(state: StrategyState, decisions: list[StrategyDecision]) -> None:
    """Cache decisions in state.runtime for this bar."""
    state.runtime[DECISIONS_CACHE_KEY] = decisions


def _get_cached_decisions(state: StrategyState) -> list[StrategyDecision] | None:
    """Get cached decisions if they exist."""
    return state.runtime.get(DECISIONS_CACHE_KEY)


@dataclass
@sf_component(name="model_entry")
class ModelEntryRule(EntryRule):
    """Entry rule that delegates to an external model.

    The model is called once per bar. Its decisions are cached in state.runtime
    so that ModelExitRule can access exit decisions without calling the model twice.

    Attributes:
        model: External model implementing StrategyModel protocol.
        base_position_size: Base position size (multiplied by decision.size_multiplier).
        max_positions: Maximum concurrent positions.
        min_confidence: Minimum confidence to act on ENTER decisions.
        allow_shorts: Allow FALL signals to create short positions.
        pair_col: Column name for pair in signals.

    Example:
        >>> from signalflow.strategy.model import ModelEntryRule, ModelExitRule
        >>>
        >>> model = MyRLModel("model.pt")
        >>> entry_rule = ModelEntryRule(
        ...     model=model,
        ...     base_position_size=0.01,
        ...     max_positions=5,
        ...     min_confidence=0.6,
        ... )
        >>> exit_rule = ModelExitRule(model=model)
        >>>
        >>> runner = BacktestRunner(
        ...     entry_rules=[entry_rule],
        ...     exit_rules=[exit_rule],
        ...     ...
        ... )
    """

    model: StrategyModel = None  # type: ignore[assignment]
    base_position_size: float = 0.01
    max_positions: int = 10
    min_confidence: float = 0.5
    allow_shorts: bool = False
    pair_col: str = "pair"

    def check_entries(
        self,
        signals: Signals,
        prices: dict[str, float],
        state: StrategyState,
    ) -> list[Order]:
        """Generate entry orders from model decisions."""
        orders: list[Order] = []

        if signals is None or signals.value.height == 0:
            return orders

        if self.model is None:
            return orders

        # Get or compute decisions
        decisions = _get_cached_decisions(state)
        if decisions is None:
            context = _build_model_context(signals, prices, state)
            decisions = self.model.decide(context)
            _cache_decisions(state, decisions)

        # Filter to ENTER decisions
        enter_decisions = [
            d for d in decisions if d.action == StrategyAction.ENTER and d.confidence >= self.min_confidence
        ]

        open_count = len(state.portfolio.open_positions())

        for decision in enter_decisions:
            if open_count >= self.max_positions:
                break

            price = prices.get(decision.pair)
            if price is None or price <= 0:
                continue

            # Calculate position size
            qty = self.base_position_size * decision.size_multiplier

            # Determine side from signal (lookup in signals)
            side: OrderSide = self._get_side_from_signals(signals, decision.pair) or "BUY"

            order = Order(
                pair=decision.pair,
                side=side,
                order_type="MARKET",
                qty=qty,
                signal_strength=decision.confidence,
                meta={
                    "decision_action": decision.action.value,
                    "model_confidence": decision.confidence,
                    "size_multiplier": decision.size_multiplier,
                    **decision.meta,
                },
            )
            orders.append(order)
            open_count += 1

        return orders

    def _get_side_from_signals(self, signals: Signals, pair: str) -> OrderSide | None:
        """Get order side from signal type for the given pair."""
        if signals is None or signals.value.height == 0:
            return None

        signal_row = signals.value.filter(pl.col(self.pair_col) == pair).head(1)

        if signal_row.height == 0:
            return None

        signal_type = signal_row.item(0, "signal_type")

        if signal_type == SignalType.RISE.value:
            return "BUY"
        elif signal_type == SignalType.FALL.value and self.allow_shorts:
            return "SELL"
        elif signal_type == SignalType.FALL.value:
            return None  # Block shorts if not allowed

        return "BUY"  # Default


@dataclass
@sf_component(name="model_exit")
class ModelExitRule(ExitRule):
    """Exit rule that uses cached model decisions.

    NOTE: If decisions are not cached yet (exit runs before entry),
    this rule will call the model and cache the results.

    Attributes:
        model: External model implementing StrategyModel protocol.
        min_confidence: Minimum confidence to act on CLOSE/CLOSE_ALL decisions.

    Example:
        >>> exit_rule = ModelExitRule(
        ...     model=model,
        ...     min_confidence=0.7,  # Higher threshold for exits
        ... )
    """

    model: StrategyModel = None  # type: ignore[assignment]
    min_confidence: float = 0.5

    def check_exits(
        self,
        positions: list[Position],
        prices: dict[str, float],
        state: StrategyState,
    ) -> list[Order]:
        """Generate exit orders from model decisions."""
        orders: list[Order] = []

        if not positions:
            return orders

        if self.model is None:
            return orders

        # Get cached decisions or compute them
        # (ExitRule runs BEFORE EntryRule in BacktestRunner)
        decisions = _get_cached_decisions(state)
        if decisions is None:
            # Get signals from runtime (stored by runner)
            signals = state.runtime.get(BAR_SIGNALS_KEY, Signals(pl.DataFrame()))
            context = _build_model_context(signals, prices, state)
            decisions = self.model.decide(context)
            _cache_decisions(state, decisions)

        # Build position lookup
        positions_by_id = {p.id: p for p in positions if not p.is_closed}
        positions_by_pair: dict[str, list[Position]] = {}
        for p in positions:
            if not p.is_closed:
                positions_by_pair.setdefault(p.pair, []).append(p)

        # Track which positions we've already added exit orders for
        exited_position_ids: set[str] = set()

        # Process CLOSE and CLOSE_ALL decisions
        for decision in decisions:
            if decision.confidence < self.min_confidence:
                continue

            if decision.action == StrategyAction.CLOSE:
                # Close specific position
                if decision.position_id is None:
                    continue
                if decision.position_id in exited_position_ids:
                    continue

                position = positions_by_id.get(decision.position_id)
                if position and not position.is_closed:
                    order = self._create_exit_order(position, prices, decision)
                    if order:
                        orders.append(order)
                        exited_position_ids.add(position.id)

            elif decision.action == StrategyAction.CLOSE_ALL:
                # Close all positions for pair
                pair_positions = positions_by_pair.get(decision.pair, [])
                for position in pair_positions:
                    if position.id in exited_position_ids:
                        continue
                    if position.is_closed:
                        continue

                    order = self._create_exit_order(position, prices, decision)
                    if order:
                        orders.append(order)
                        exited_position_ids.add(position.id)

        return orders

    def _create_exit_order(
        self,
        position: Position,
        prices: dict[str, float],
        decision: StrategyDecision,
    ) -> Order | None:
        """Create exit order for a position."""
        price = prices.get(position.pair)
        if price is None or price <= 0:
            return None

        side: OrderSide = "SELL" if position.position_type == PositionType.LONG else "BUY"

        return Order(
            pair=position.pair,
            side=side,
            order_type="MARKET",
            qty=position.qty,
            position_id=position.id,
            meta={
                "exit_reason": "model_exit",
                "decision_action": decision.action.value,
                "model_confidence": decision.confidence,
                "entry_price": position.entry_price,
                "exit_price": price,
                **decision.meta,
            },
        )
