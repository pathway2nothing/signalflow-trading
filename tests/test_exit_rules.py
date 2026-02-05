"""Tests for exit rules: TakeProfitStopLossExit, TimeBasedExit."""

from datetime import datetime

import pytest

from signalflow.core.containers.position import Position
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import PositionType
from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit
from signalflow.strategy.component.exit.time_based import TimeBasedExit


def _make_state():
    return StrategyState(strategy_id="test")


# ── TakeProfitStopLossExit — LONG positions ────────────────────────────────


class TestTpSlLong:
    def test_no_positions_returns_empty(self):
        rule = TakeProfitStopLossExit()
        assert rule.check_exits([], {"BTCUSDT": 100.0}, _make_state()) == []

    def test_closed_position_skipped(self, long_position):
        long_position.is_closed = True
        rule = TakeProfitStopLossExit()
        assert rule.check_exits([long_position], {"BTCUSDT": 200.0}, _make_state()) == []

    def test_no_price_for_pair_skipped(self, long_position):
        rule = TakeProfitStopLossExit()
        assert rule.check_exits([long_position], {}, _make_state()) == []

    def test_zero_price_skipped(self, long_position):
        rule = TakeProfitStopLossExit()
        assert rule.check_exits([long_position], {"BTCUSDT": 0}, _make_state()) == []

    def test_take_profit_triggered(self, long_position):
        rule = TakeProfitStopLossExit(take_profit_pct=0.02)
        # entry=100, tp=102
        orders = rule.check_exits([long_position], {"BTCUSDT": 103.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].side == "SELL"
        assert orders[0].meta["exit_reason"] == "take_profit"

    def test_stop_loss_triggered(self, long_position):
        rule = TakeProfitStopLossExit(stop_loss_pct=0.01)
        # entry=100, sl=99
        orders = rule.check_exits([long_position], {"BTCUSDT": 98.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].side == "SELL"
        assert orders[0].meta["exit_reason"] == "stop_loss"

    def test_no_trigger_in_range(self, long_position):
        rule = TakeProfitStopLossExit(take_profit_pct=0.05, stop_loss_pct=0.05)
        orders = rule.check_exits([long_position], {"BTCUSDT": 101.0}, _make_state())
        assert orders == []

    def test_tp_exact_boundary(self, long_position):
        rule = TakeProfitStopLossExit(take_profit_pct=0.02)
        # tp = 100 * 1.02 = 102.0
        orders = rule.check_exits([long_position], {"BTCUSDT": 102.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "take_profit"

    def test_sl_exact_boundary(self, long_position):
        rule = TakeProfitStopLossExit(stop_loss_pct=0.01)
        # sl = 100 * 0.99 = 99.0
        orders = rule.check_exits([long_position], {"BTCUSDT": 99.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "stop_loss"


# ── TakeProfitStopLossExit — SHORT positions ───────────────────────────────


class TestTpSlShort:
    def test_take_profit_triggered(self, short_position):
        rule = TakeProfitStopLossExit(take_profit_pct=0.02)
        # entry=100, tp=98 (short: price must go DOWN)
        orders = rule.check_exits([short_position], {"ETHUSDT": 97.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].side == "BUY"
        assert orders[0].meta["exit_reason"] == "take_profit"

    def test_stop_loss_triggered(self, short_position):
        rule = TakeProfitStopLossExit(stop_loss_pct=0.01)
        # entry=100, sl=101
        orders = rule.check_exits([short_position], {"ETHUSDT": 102.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].side == "BUY"
        assert orders[0].meta["exit_reason"] == "stop_loss"

    def test_no_trigger_in_range(self, short_position):
        rule = TakeProfitStopLossExit(take_profit_pct=0.05, stop_loss_pct=0.05)
        orders = rule.check_exits([short_position], {"ETHUSDT": 99.5}, _make_state())
        assert orders == []


# ── TakeProfitStopLossExit — dynamic levels ────────────────────────────────


class TestTpSlDynamicLevels:
    def test_use_position_levels_tp(self, long_position):
        long_position.meta["take_profit_price"] = 110.0
        long_position.meta["stop_loss_price"] = 90.0
        rule = TakeProfitStopLossExit(use_position_levels=True)
        orders = rule.check_exits([long_position], {"BTCUSDT": 111.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "take_profit"

    def test_use_position_levels_sl(self, long_position):
        long_position.meta["take_profit_price"] = 110.0
        long_position.meta["stop_loss_price"] = 90.0
        rule = TakeProfitStopLossExit(use_position_levels=True)
        orders = rule.check_exits([long_position], {"BTCUSDT": 89.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "stop_loss"

    def test_position_levels_none_no_trigger(self, long_position):
        # No meta keys set → tp_price/sl_price are None → falsy, no trigger
        rule = TakeProfitStopLossExit(use_position_levels=True)
        orders = rule.check_exits([long_position], {"BTCUSDT": 200.0}, _make_state())
        assert orders == []


# ── TakeProfitStopLossExit — order details ─────────────────────────────────


class TestTpSlOrderMeta:
    def test_order_qty_matches_position(self, long_position):
        rule = TakeProfitStopLossExit(take_profit_pct=0.01)
        orders = rule.check_exits([long_position], {"BTCUSDT": 102.0}, _make_state())
        assert orders[0].qty == long_position.qty

    def test_order_position_id_set(self, long_position):
        rule = TakeProfitStopLossExit(take_profit_pct=0.01)
        orders = rule.check_exits([long_position], {"BTCUSDT": 102.0}, _make_state())
        assert orders[0].position_id == long_position.id

    def test_order_meta_has_prices(self, long_position):
        rule = TakeProfitStopLossExit(take_profit_pct=0.01)
        orders = rule.check_exits([long_position], {"BTCUSDT": 102.0}, _make_state())
        assert orders[0].meta["entry_price"] == 100.0
        assert orders[0].meta["exit_price"] == 102.0


# ── TimeBasedExit ───────────────────────────────────────────────────────────


class TestTimeBasedExit:
    def test_no_positions_returns_empty(self):
        rule = TimeBasedExit(max_bars=5)
        assert rule.check_exits([], {"BTCUSDT": 100.0}, _make_state()) == []

    def test_closed_position_skipped(self, long_position):
        long_position.is_closed = True
        rule = TimeBasedExit(max_bars=1)
        assert rule.check_exits([long_position], {"BTCUSDT": 100.0}, _make_state()) == []

    def test_no_price_skipped(self, long_position):
        rule = TimeBasedExit(max_bars=1)
        assert rule.check_exits([long_position], {}, _make_state()) == []

    def test_increments_bar_count(self, long_position):
        rule = TimeBasedExit(max_bars=100)
        rule.check_exits([long_position], {"BTCUSDT": 100.0}, _make_state())
        assert long_position.meta["bar_count"] == 1
        rule.check_exits([long_position], {"BTCUSDT": 100.0}, _make_state())
        assert long_position.meta["bar_count"] == 2

    def test_below_threshold_no_exit(self, long_position):
        rule = TimeBasedExit(max_bars=5)
        for _ in range(4):
            orders = rule.check_exits([long_position], {"BTCUSDT": 100.0}, _make_state())
        assert orders == []

    def test_at_threshold_triggers_exit(self, long_position):
        rule = TimeBasedExit(max_bars=3)
        for _ in range(2):
            rule.check_exits([long_position], {"BTCUSDT": 100.0}, _make_state())
        orders = rule.check_exits([long_position], {"BTCUSDT": 100.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "time_exit"
        assert orders[0].meta["bars_held"] == 3

    def test_long_exit_is_sell(self, long_position):
        rule = TimeBasedExit(max_bars=1)
        orders = rule.check_exits([long_position], {"BTCUSDT": 100.0}, _make_state())
        assert orders[0].side == "SELL"

    def test_short_exit_is_buy(self, short_position):
        rule = TimeBasedExit(max_bars=1)
        orders = rule.check_exits([short_position], {"ETHUSDT": 100.0}, _make_state())
        assert orders[0].side == "BUY"

    def test_custom_bar_col(self, long_position):
        rule = TimeBasedExit(max_bars=2, bar_col="my_counter")
        rule.check_exits([long_position], {"BTCUSDT": 100.0}, _make_state())
        assert long_position.meta["my_counter"] == 1
