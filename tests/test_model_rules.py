"""Tests for ModelEntryRule and ModelExitRule."""

from datetime import datetime

import polars as pl
import pytest

from signalflow.core.containers.position import Position
from signalflow.core.containers.signals import Signals
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import PositionType
from signalflow.strategy.model.context import ModelContext
from signalflow.strategy.model.decision import StrategyAction, StrategyDecision
from signalflow.strategy.model.rules import ModelEntryRule, ModelExitRule


TS = datetime(2024, 1, 1)


def _make_state(cash: float = 10000.0) -> StrategyState:
    state = StrategyState(strategy_id="test")
    state.portfolio.cash = cash
    state.last_ts = TS
    return state


def _signals(rows: list[dict]) -> Signals:
    """Build Signals from list of dicts."""
    return Signals(pl.DataFrame(rows))


class MockModel:
    """Mock model for testing."""

    def __init__(self, decisions: list[StrategyDecision] | None = None):
        self.decisions = decisions or []
        self.call_count = 0

    def decide(self, context: ModelContext) -> list[StrategyDecision]:
        self.call_count += 1
        return self.decisions


# ── ModelEntryRule ──────────────────────────────────────────────────────────


class TestModelEntryRuleBasic:
    def test_no_model_returns_empty(self):
        rule = ModelEntryRule(model=None)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])
        orders = rule.check_entries(sigs, {"BTCUSDT": 100.0}, _make_state())
        assert orders == []

    def test_empty_signals_returns_empty(self):
        model = MockModel(decisions=[])
        rule = ModelEntryRule(model=model)
        empty = Signals(pl.DataFrame())
        orders = rule.check_entries(empty, {"BTCUSDT": 100.0}, _make_state())
        assert orders == []

    def test_enter_decision_creates_order(self):
        decisions = [
            StrategyDecision(
                action=StrategyAction.ENTER,
                pair="BTCUSDT",
                size_multiplier=1.0,
                confidence=0.8,
            )
        ]
        model = MockModel(decisions=decisions)
        rule = ModelEntryRule(model=model, base_position_size=0.01)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])

        orders = rule.check_entries(sigs, {"BTCUSDT": 50000.0}, _make_state())

        assert len(orders) == 1
        assert orders[0].pair == "BTCUSDT"
        assert orders[0].side == "BUY"
        assert orders[0].qty == 0.01
        assert orders[0].meta["model_confidence"] == 0.8

    def test_size_multiplier_affects_qty(self):
        decisions = [
            StrategyDecision(
                action=StrategyAction.ENTER,
                pair="BTCUSDT",
                size_multiplier=2.0,
                confidence=0.9,
            )
        ]
        model = MockModel(decisions=decisions)
        rule = ModelEntryRule(model=model, base_position_size=0.01)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])

        orders = rule.check_entries(sigs, {"BTCUSDT": 50000.0}, _make_state())

        assert orders[0].qty == 0.02  # 0.01 * 2.0

    def test_min_confidence_filters(self):
        decisions = [
            StrategyDecision(
                action=StrategyAction.ENTER,
                pair="BTCUSDT",
                confidence=0.4,  # Below threshold
            )
        ]
        model = MockModel(decisions=decisions)
        rule = ModelEntryRule(model=model, min_confidence=0.5)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])

        orders = rule.check_entries(sigs, {"BTCUSDT": 50000.0}, _make_state())

        assert orders == []

    def test_max_positions_limit(self):
        decisions = [
            StrategyDecision(action=StrategyAction.ENTER, pair="BTCUSDT", confidence=0.8),
            StrategyDecision(action=StrategyAction.ENTER, pair="ETHUSDT", confidence=0.8),
        ]
        model = MockModel(decisions=decisions)
        rule = ModelEntryRule(model=model, max_positions=1, base_position_size=0.01)
        sigs = _signals(
            [
                {"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9},
                {"pair": "ETHUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9},
            ]
        )

        orders = rule.check_entries(sigs, {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}, _make_state())

        assert len(orders) == 1

    def test_skip_decision_ignored(self):
        decisions = [
            StrategyDecision(action=StrategyAction.SKIP, pair="BTCUSDT"),
        ]
        model = MockModel(decisions=decisions)
        rule = ModelEntryRule(model=model)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])

        orders = rule.check_entries(sigs, {"BTCUSDT": 50000.0}, _make_state())

        assert orders == []


class TestModelEntryRuleCaching:
    def test_decisions_cached_in_state(self):
        decisions = [StrategyDecision(action=StrategyAction.ENTER, pair="BTCUSDT", confidence=0.8)]
        model = MockModel(decisions=decisions)
        rule = ModelEntryRule(model=model, base_position_size=0.01)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])
        state = _make_state()

        # First call
        rule.check_entries(sigs, {"BTCUSDT": 50000.0}, state)
        assert model.call_count == 1

        # Second call should use cached decisions
        rule.check_entries(sigs, {"BTCUSDT": 50000.0}, state)
        assert model.call_count == 1  # Still 1 - no new call


# ── ModelExitRule ───────────────────────────────────────────────────────────


class TestModelExitRuleBasic:
    def test_no_model_returns_empty(self):
        rule = ModelExitRule(model=None)
        pos = Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=50000.0, qty=0.1)
        orders = rule.check_exits([pos], {"BTCUSDT": 51000.0}, _make_state())
        assert orders == []

    def test_no_positions_returns_empty(self):
        model = MockModel(decisions=[])
        rule = ModelExitRule(model=model)
        orders = rule.check_exits([], {"BTCUSDT": 51000.0}, _make_state())
        assert orders == []

    def test_close_decision_creates_exit_order(self):
        decisions = [
            StrategyDecision(
                action=StrategyAction.CLOSE,
                pair="BTCUSDT",
                position_id="p1",
                confidence=0.9,
            )
        ]
        model = MockModel(decisions=decisions)
        rule = ModelExitRule(model=model)
        pos = Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=50000.0, qty=0.1)
        state = _make_state()
        state.portfolio.positions[pos.id] = pos
        state.runtime["_bar_signals"] = Signals(pl.DataFrame())

        orders = rule.check_exits([pos], {"BTCUSDT": 51000.0}, state)

        assert len(orders) == 1
        assert orders[0].pair == "BTCUSDT"
        assert orders[0].side == "SELL"  # Closing LONG
        assert orders[0].position_id == "p1"
        assert orders[0].meta["exit_reason"] == "model_exit"

    def test_close_all_closes_all_pair_positions(self):
        decisions = [
            StrategyDecision(
                action=StrategyAction.CLOSE_ALL,
                pair="BTCUSDT",
                confidence=0.8,
            )
        ]
        model = MockModel(decisions=decisions)
        rule = ModelExitRule(model=model)

        positions = [
            Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=50000.0, qty=0.1),
            Position(id="p2", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=49000.0, qty=0.05),
            Position(id="p3", pair="ETHUSDT", position_type=PositionType.LONG, entry_price=3000.0, qty=1.0),
        ]
        state = _make_state()
        for pos in positions:
            state.portfolio.positions[pos.id] = pos
        state.runtime["_bar_signals"] = Signals(pl.DataFrame())

        orders = rule.check_exits(positions, {"BTCUSDT": 51000.0, "ETHUSDT": 3100.0}, state)

        assert len(orders) == 2
        assert all(o.pair == "BTCUSDT" for o in orders)
        assert {o.position_id for o in orders} == {"p1", "p2"}

    def test_min_confidence_filters_exit(self):
        decisions = [
            StrategyDecision(
                action=StrategyAction.CLOSE,
                pair="BTCUSDT",
                position_id="p1",
                confidence=0.4,  # Below threshold
            )
        ]
        model = MockModel(decisions=decisions)
        rule = ModelExitRule(model=model, min_confidence=0.5)
        pos = Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=50000.0, qty=0.1)
        state = _make_state()
        state.portfolio.positions[pos.id] = pos
        state.runtime["_bar_signals"] = Signals(pl.DataFrame())

        orders = rule.check_exits([pos], {"BTCUSDT": 51000.0}, state)

        assert orders == []

    def test_hold_decision_no_action(self):
        decisions = [
            StrategyDecision(action=StrategyAction.HOLD, pair="BTCUSDT"),
        ]
        model = MockModel(decisions=decisions)
        rule = ModelExitRule(model=model)
        pos = Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=50000.0, qty=0.1)
        state = _make_state()
        state.portfolio.positions[pos.id] = pos
        state.runtime["_bar_signals"] = Signals(pl.DataFrame())

        orders = rule.check_exits([pos], {"BTCUSDT": 51000.0}, state)

        assert orders == []

    def test_short_position_exit_uses_buy(self):
        decisions = [
            StrategyDecision(
                action=StrategyAction.CLOSE,
                pair="BTCUSDT",
                position_id="p1",
                confidence=0.9,
            )
        ]
        model = MockModel(decisions=decisions)
        rule = ModelExitRule(model=model)
        pos = Position(id="p1", pair="BTCUSDT", position_type=PositionType.SHORT, entry_price=50000.0, qty=0.1)
        state = _make_state()
        state.portfolio.positions[pos.id] = pos
        state.runtime["_bar_signals"] = Signals(pl.DataFrame())

        orders = rule.check_exits([pos], {"BTCUSDT": 49000.0}, state)

        assert len(orders) == 1
        assert orders[0].side == "BUY"  # Closing SHORT


class TestModelExitRuleCaching:
    def test_exit_rule_calls_model_if_not_cached(self):
        decisions = [StrategyDecision(action=StrategyAction.HOLD, pair="BTCUSDT")]
        model = MockModel(decisions=decisions)
        rule = ModelExitRule(model=model)
        pos = Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=50000.0, qty=0.1)
        state = _make_state()
        state.portfolio.positions[pos.id] = pos
        state.runtime["_bar_signals"] = Signals(pl.DataFrame())

        # Exit rule calls model when no cache
        rule.check_exits([pos], {"BTCUSDT": 51000.0}, state)

        assert model.call_count == 1
        assert "_model_decisions" in state.runtime

    def test_exit_rule_uses_cached_decisions(self):
        decisions = [StrategyDecision(action=StrategyAction.HOLD, pair="BTCUSDT")]
        model = MockModel(decisions=decisions)
        rule = ModelExitRule(model=model)
        pos = Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=50000.0, qty=0.1)
        state = _make_state()
        state.portfolio.positions[pos.id] = pos
        state.runtime["_bar_signals"] = Signals(pl.DataFrame())
        # Pre-cache decisions
        state.runtime["_model_decisions"] = decisions

        rule.check_exits([pos], {"BTCUSDT": 51000.0}, state)

        assert model.call_count == 0  # Used cache


# ── Integrated test ─────────────────────────────────────────────────────────


class TestModelRulesIntegration:
    def test_shared_model_called_once(self):
        """Entry and exit rules sharing a model should only call it once."""
        decisions = [
            StrategyDecision(action=StrategyAction.ENTER, pair="BTCUSDT", confidence=0.8),
        ]
        model = MockModel(decisions=decisions)
        entry_rule = ModelEntryRule(model=model, base_position_size=0.01)
        exit_rule = ModelExitRule(model=model)

        state = _make_state()
        # Add a position so exit rule doesn't return early
        pos = Position(id="p1", pair="ETHUSDT", position_type=PositionType.LONG, entry_price=3000.0, qty=1.0)
        state.portfolio.positions[pos.id] = pos
        state.runtime["_bar_signals"] = Signals(pl.DataFrame())
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])

        # Simulate BacktestRunner order: exits first, then entries
        exit_rule.check_exits([pos], {"BTCUSDT": 50000.0, "ETHUSDT": 3100.0}, state)
        assert model.call_count == 1

        entry_rule.check_entries(sigs, {"BTCUSDT": 50000.0}, state)
        assert model.call_count == 1  # Still 1 - used cache
