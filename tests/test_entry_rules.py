"""Tests for entry rules: SignalEntryRule, FixedSizeEntryRule."""

from datetime import datetime

import polars as pl
import pytest

from signalflow.core.containers.position import Position
from signalflow.core.containers.signals import Signals
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import PositionType
from signalflow.strategy.component.entry.fixed_size import FixedSizeEntryRule
from signalflow.strategy.component.entry.signal import SignalEntryRule


TS = datetime(2024, 1, 1)


def _make_state(cash=10000.0):
    state = StrategyState(strategy_id="test")
    state.portfolio.cash = cash
    return state


def _signals(rows):
    """Quick helper: build Signals from list of dicts."""
    return Signals(pl.DataFrame(rows))


# ── SignalEntryRule - basics ────────────────────────────────────────────────


class TestSignalEntryRuleBasic:
    def test_none_signals(self):
        rule = SignalEntryRule()
        assert rule.check_entries(None, {"BTCUSDT": 100.0}, _make_state()) == []

    def test_empty_signals(self):
        rule = SignalEntryRule()
        empty = _signals({"pair": [], "timestamp": [], "signal_type": [], "signal": [], "probability": []})
        assert rule.check_entries(empty, {"BTCUSDT": 100.0}, _make_state()) == []

    def test_rise_generates_buy(self):
        rule = SignalEntryRule(base_position_size=100.0, min_probability=0.0)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])
        orders = rule.check_entries(sigs, {"BTCUSDT": 100.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].side == "BUY"
        assert orders[0].pair == "BTCUSDT"

    def test_fall_ignored_when_shorts_off(self):
        rule = SignalEntryRule(allow_shorts=False, min_probability=0.0)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "fall", "signal": -1, "probability": 0.9}])
        orders = rule.check_entries(sigs, {"BTCUSDT": 100.0}, _make_state())
        assert orders == []

    def test_fall_generates_sell(self):
        rule = SignalEntryRule(allow_shorts=True, min_probability=0.0)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "fall", "signal": -1, "probability": 0.8}])
        orders = rule.check_entries(sigs, {"BTCUSDT": 100.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].side == "SELL"

    def test_none_signal_skipped(self):
        rule = SignalEntryRule(min_probability=0.0)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "none", "signal": 0, "probability": 0.0}])
        assert rule.check_entries(sigs, {"BTCUSDT": 100.0}, _make_state()) == []


# ── Position limits ─────────────────────────────────────────────────────────


class TestSignalEntryRuleLimits:
    def test_max_total_positions_reached(self):
        rule = SignalEntryRule(max_total_positions=1, min_probability=0.0)
        state = _make_state()
        pos = Position(id="p1", pair="ETHUSDT", position_type=PositionType.LONG, entry_price=50.0, qty=1.0)
        state.portfolio.positions[pos.id] = pos
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])
        assert rule.check_entries(sigs, {"BTCUSDT": 100.0}, state) == []

    def test_max_positions_per_pair(self):
        rule = SignalEntryRule(max_positions_per_pair=1, min_probability=0.0)
        state = _make_state()
        pos = Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=100.0, qty=1.0)
        state.portfolio.positions[pos.id] = pos
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])
        assert rule.check_entries(sigs, {"BTCUSDT": 100.0}, state) == []

    def test_multiple_pairs_independent(self):
        rule = SignalEntryRule(max_positions_per_pair=1, min_probability=0.0)
        state = _make_state()
        pos = Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=100.0, qty=1.0)
        state.portfolio.positions[pos.id] = pos
        sigs = _signals([{"pair": "ETHUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])
        orders = rule.check_entries(sigs, {"ETHUSDT": 50.0}, state)
        assert len(orders) == 1

    def test_total_limit_across_signals(self):
        rule = SignalEntryRule(max_total_positions=2, min_probability=0.0)
        sigs = _signals(
            [
                {"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9},
                {"pair": "ETHUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.8},
                {"pair": "SOLUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.7},
            ]
        )
        orders = rule.check_entries(sigs, {"BTCUSDT": 100.0, "ETHUSDT": 50.0, "SOLUSDT": 10.0}, _make_state())
        assert len(orders) == 2


# ── Capital constraints ─────────────────────────────────────────────────────


class TestSignalEntryRuleCapital:
    def test_min_order_notional_filters(self):
        rule = SignalEntryRule(base_position_size=5.0, min_order_notional=10.0, min_probability=0.0)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 1.0}])
        assert rule.check_entries(sigs, {"BTCUSDT": 100.0}, _make_state()) == []

    def test_cash_capping(self):
        rule = SignalEntryRule(base_position_size=20000.0, min_probability=0.0)
        state = _make_state(cash=100.0)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 1.0}])
        orders = rule.check_entries(sigs, {"BTCUSDT": 10.0}, state)
        assert len(orders) == 1
        # notional = min(20000, cash*0.99=99.0, remaining_alloc=equity*0.95=95.0) = 95.0
        assert orders[0].meta["requested_notional"] == pytest.approx(95.0)

    def test_insufficient_cash(self):
        rule = SignalEntryRule(min_order_notional=10.0, min_probability=0.0)
        state = _make_state(cash=5.0)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])
        assert rule.check_entries(sigs, {"BTCUSDT": 100.0}, state) == []


# ── Probability ─────────────────────────────────────────────────────────────


class TestSignalEntryRuleProbability:
    def test_min_probability_filter(self):
        rule = SignalEntryRule(min_probability=0.7)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.5}])
        assert rule.check_entries(sigs, {"BTCUSDT": 100.0}, _make_state()) == []

    def test_probability_sorting(self):
        rule = SignalEntryRule(max_total_positions=1, min_probability=0.0)
        sigs = _signals(
            [
                {"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.6},
                {"pair": "ETHUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9},
            ]
        )
        orders = rule.check_entries(sigs, {"BTCUSDT": 100.0, "ETHUSDT": 50.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].pair == "ETHUSDT"  # higher probability first

    def test_probability_sizing(self):
        rule = SignalEntryRule(base_position_size=100.0, use_probability_sizing=True, min_probability=0.0)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.5}])
        orders = rule.check_entries(sigs, {"BTCUSDT": 10.0}, _make_state())
        assert orders[0].meta["requested_notional"] == pytest.approx(50.0)

    def test_probability_sizing_disabled(self):
        rule = SignalEntryRule(base_position_size=100.0, use_probability_sizing=False, min_probability=0.0)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.5}])
        orders = rule.check_entries(sigs, {"BTCUSDT": 10.0}, _make_state())
        assert orders[0].meta["requested_notional"] == pytest.approx(100.0)


# ── Order details ───────────────────────────────────────────────────────────


class TestSignalEntryRuleOrderDetails:
    def test_order_meta_contents(self):
        rule = SignalEntryRule(min_probability=0.0)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])
        orders = rule.check_entries(sigs, {"BTCUSDT": 100.0}, _make_state())
        assert orders[0].meta["signal_type"] == "rise"
        assert orders[0].meta["signal_probability"] == 0.9
        assert "requested_notional" in orders[0].meta

    def test_qty_equals_notional_over_price(self):
        rule = SignalEntryRule(base_position_size=100.0, use_probability_sizing=False, min_probability=0.0)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 1.0}])
        orders = rule.check_entries(sigs, {"BTCUSDT": 50.0}, _make_state())
        assert orders[0].qty == pytest.approx(2.0)

    def test_no_price_skipped(self):
        rule = SignalEntryRule(min_probability=0.0)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])
        assert rule.check_entries(sigs, {}, _make_state()) == []

    def test_zero_price_skipped(self):
        rule = SignalEntryRule(min_probability=0.0)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.9}])
        assert rule.check_entries(sigs, {"BTCUSDT": 0}, _make_state()) == []


# ── FixedSizeEntryRule ──────────────────────────────────────────────────────


class TestFixedSizeEntryRule:
    def test_rise_signal_buy(self):
        rule = FixedSizeEntryRule(position_size=0.5)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1}])
        orders = rule.check_entries(sigs, {"BTCUSDT": 100.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].side == "BUY"
        assert orders[0].qty == 0.5

    def test_fall_signal_sell(self):
        rule = FixedSizeEntryRule(position_size=0.5, signal_types=["fall"])
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "fall", "signal": -1}])
        orders = rule.check_entries(sigs, {"BTCUSDT": 100.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].side == "SELL"

    def test_max_positions_reached(self):
        rule = FixedSizeEntryRule(max_positions=0)
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1}])
        assert rule.check_entries(sigs, {"BTCUSDT": 100.0}, _make_state()) == []

    def test_no_price_skipped(self):
        rule = FixedSizeEntryRule()
        sigs = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1}])
        assert rule.check_entries(sigs, {}, _make_state()) == []

    def test_none_signals(self):
        rule = FixedSizeEntryRule()
        assert rule.check_entries(None, {"BTCUSDT": 100.0}, _make_state()) == []
