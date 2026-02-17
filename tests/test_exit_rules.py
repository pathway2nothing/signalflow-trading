"""Tests for exit rules: TakeProfitStopLossExit, TimeBasedExit, TrailingStopExit, VolatilityExit, CompositeExit."""

from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import ExitPriority
from signalflow.strategy.component.exit.composite import CompositeExit
from signalflow.strategy.component.exit.time_based import TimeBasedExit
from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit
from signalflow.strategy.component.exit.trailing_stop import TrailingStopExit
from signalflow.strategy.component.exit.volatility_exit import VolatilityExit


def _make_state():
    return StrategyState(strategy_id="test")


# ── TakeProfitStopLossExit - LONG positions ────────────────────────────────


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


# ── TakeProfitStopLossExit - SHORT positions ───────────────────────────────


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


# ── TakeProfitStopLossExit - dynamic levels ────────────────────────────────


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


# ── TakeProfitStopLossExit - order details ─────────────────────────────────


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


# ── TrailingStopExit ─────────────────────────────────────────────────────────


class TestTrailingStopExit:
    def test_no_positions_returns_empty(self):
        rule = TrailingStopExit()
        assert rule.check_exits([], {"BTCUSDT": 100.0}, _make_state()) == []

    def test_closed_position_skipped(self, long_position):
        long_position.is_closed = True
        rule = TrailingStopExit()
        assert rule.check_exits([long_position], {"BTCUSDT": 50.0}, _make_state()) == []

    def test_no_price_skipped(self, long_position):
        rule = TrailingStopExit()
        assert rule.check_exits([long_position], {}, _make_state()) == []

    def test_long_peak_tracking(self, long_position):
        rule = TrailingStopExit(trail_pct=0.05)
        # Price goes up - should track peak
        rule.check_exits([long_position], {"BTCUSDT": 110.0}, _make_state())
        assert long_position.meta["_trailing_peak"] == 110.0
        # Price goes higher
        rule.check_exits([long_position], {"BTCUSDT": 120.0}, _make_state())
        assert long_position.meta["_trailing_peak"] == 120.0
        # Price drops but peak should remain
        rule.check_exits([long_position], {"BTCUSDT": 118.0}, _make_state())
        assert long_position.meta["_trailing_peak"] == 120.0

    def test_long_trailing_stop_triggered(self, long_position):
        rule = TrailingStopExit(trail_pct=0.05)
        # Set peak at 120
        rule.check_exits([long_position], {"BTCUSDT": 120.0}, _make_state())
        # Trail price = 120 * 0.95 = 114, drop below triggers exit
        orders = rule.check_exits([long_position], {"BTCUSDT": 113.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].side == "SELL"
        assert orders[0].meta["exit_reason"] == "trailing_stop"
        assert orders[0].meta["peak_price"] == 120.0

    def test_long_no_trigger_above_trail(self, long_position):
        rule = TrailingStopExit(trail_pct=0.05)
        # Peak at 120, trail at 114
        rule.check_exits([long_position], {"BTCUSDT": 120.0}, _make_state())
        orders = rule.check_exits([long_position], {"BTCUSDT": 115.0}, _make_state())
        assert orders == []

    def test_short_trough_tracking(self, short_position):
        rule = TrailingStopExit(trail_pct=0.05)
        # Price goes down - should track trough
        rule.check_exits([short_position], {"ETHUSDT": 90.0}, _make_state())
        assert short_position.meta["_trailing_trough"] == 90.0
        # Price goes lower
        rule.check_exits([short_position], {"ETHUSDT": 80.0}, _make_state())
        assert short_position.meta["_trailing_trough"] == 80.0

    def test_short_trailing_stop_triggered(self, short_position):
        rule = TrailingStopExit(trail_pct=0.05)
        # Set trough at 80
        rule.check_exits([short_position], {"ETHUSDT": 80.0}, _make_state())
        # Trail price = 80 * 1.05 = 84, rise above triggers exit
        orders = rule.check_exits([short_position], {"ETHUSDT": 85.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].side == "BUY"
        assert orders[0].meta["exit_reason"] == "trailing_stop"
        assert orders[0].meta["trough_price"] == 80.0

    def test_activation_not_reached(self, long_position):
        rule = TrailingStopExit(trail_pct=0.05, activation_pct=0.10)
        # Entry=100, need 110 to activate
        # Price at 105 - not activated yet, no trailing
        orders = rule.check_exits([long_position], {"BTCUSDT": 105.0}, _make_state())
        assert orders == []
        assert long_position.meta.get("_trailing_activated") is not True

    def test_activation_then_trails(self, long_position):
        rule = TrailingStopExit(trail_pct=0.05, activation_pct=0.10)
        # Entry=100, need 110 to activate
        # Price at 115 - activated
        rule.check_exits([long_position], {"BTCUSDT": 115.0}, _make_state())
        assert long_position.meta["_trailing_activated"] is True
        assert long_position.meta["_trailing_peak"] == 115.0
        # Now trailing is active, price drops to trigger
        # Trail = 115 * 0.95 = 109.25
        orders = rule.check_exits([long_position], {"BTCUSDT": 108.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "trailing_stop"

    def test_atr_based_trailing(self, long_position):
        rule = TrailingStopExit(use_atr=True, atr_multiplier=2.0)
        state = _make_state()
        state.runtime["atr"] = {"BTCUSDT": 5.0}  # ATR = 5, trail = 10
        # Peak at 120, trail price = 120 - 10 = 110
        rule.check_exits([long_position], {"BTCUSDT": 120.0}, state)
        # Price at 109 triggers exit
        orders = rule.check_exits([long_position], {"BTCUSDT": 109.0}, state)
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "trailing_stop"

    def test_atr_fallback_to_entry_atr(self, long_position):
        rule = TrailingStopExit(use_atr=True, atr_multiplier=2.0)
        long_position.meta["entry_atr"] = 3.0  # Trail = 6
        state = _make_state()
        # No state.runtime["atr"], should use entry_atr
        rule.check_exits([long_position], {"BTCUSDT": 110.0}, state)
        # Trail price = 110 - 6 = 104
        orders = rule.check_exits([long_position], {"BTCUSDT": 103.0}, state)
        assert len(orders) == 1


# ── VolatilityExit ───────────────────────────────────────────────────────────


class TestVolatilityExit:
    def test_no_positions_returns_empty(self):
        rule = VolatilityExit()
        assert rule.check_exits([], {"BTCUSDT": 100.0}, _make_state()) == []

    def test_closed_position_skipped(self, long_position):
        long_position.is_closed = True
        state = _make_state()
        state.runtime["atr"] = {"BTCUSDT": 5.0}
        rule = VolatilityExit()
        assert rule.check_exits([long_position], {"BTCUSDT": 200.0}, state) == []

    def test_levels_calculated_on_first_call(self, long_position):
        rule = VolatilityExit(tp_atr_mult=3.0, sl_atr_mult=1.5)
        state = _make_state()
        state.runtime["atr"] = {"BTCUSDT": 5.0}
        # Entry=100, ATR=5 -> TP=115, SL=92.5
        rule.check_exits([long_position], {"BTCUSDT": 105.0}, state)
        assert long_position.meta["_vol_tp_price"] == 115.0
        assert long_position.meta["_vol_sl_price"] == 92.5
        assert long_position.meta["_vol_atr_used"] == 5.0

    def test_long_tp_triggered(self, long_position):
        rule = VolatilityExit(tp_atr_mult=3.0, sl_atr_mult=1.5)
        state = _make_state()
        state.runtime["atr"] = {"BTCUSDT": 5.0}
        # Entry=100, TP=115
        orders = rule.check_exits([long_position], {"BTCUSDT": 116.0}, state)
        assert len(orders) == 1
        assert orders[0].side == "SELL"
        assert orders[0].meta["exit_reason"] == "volatility_tp"

    def test_long_sl_triggered(self, long_position):
        rule = VolatilityExit(tp_atr_mult=3.0, sl_atr_mult=1.5)
        state = _make_state()
        state.runtime["atr"] = {"BTCUSDT": 5.0}
        # Entry=100, SL=92.5
        orders = rule.check_exits([long_position], {"BTCUSDT": 91.0}, state)
        assert len(orders) == 1
        assert orders[0].side == "SELL"
        assert orders[0].meta["exit_reason"] == "volatility_sl"

    def test_long_no_trigger_in_range(self, long_position):
        rule = VolatilityExit(tp_atr_mult=3.0, sl_atr_mult=1.5)
        state = _make_state()
        state.runtime["atr"] = {"BTCUSDT": 5.0}
        # Entry=100, TP=115, SL=92.5
        orders = rule.check_exits([long_position], {"BTCUSDT": 105.0}, state)
        assert orders == []

    def test_short_tp_triggered(self, short_position):
        rule = VolatilityExit(tp_atr_mult=3.0, sl_atr_mult=1.5)
        state = _make_state()
        state.runtime["atr"] = {"ETHUSDT": 5.0}
        # Entry=100, TP=85 (short: price down)
        orders = rule.check_exits([short_position], {"ETHUSDT": 84.0}, state)
        assert len(orders) == 1
        assert orders[0].side == "BUY"
        assert orders[0].meta["exit_reason"] == "volatility_tp"

    def test_short_sl_triggered(self, short_position):
        rule = VolatilityExit(tp_atr_mult=3.0, sl_atr_mult=1.5)
        state = _make_state()
        state.runtime["atr"] = {"ETHUSDT": 5.0}
        # Entry=100, SL=107.5 (short: price up)
        orders = rule.check_exits([short_position], {"ETHUSDT": 108.0}, state)
        assert len(orders) == 1
        assert orders[0].side == "BUY"
        assert orders[0].meta["exit_reason"] == "volatility_sl"

    def test_missing_atr_skips(self, long_position):
        rule = VolatilityExit()
        state = _make_state()
        # No ATR data
        orders = rule.check_exits([long_position], {"BTCUSDT": 200.0}, state)
        assert orders == []

    def test_entry_atr_fallback(self, long_position):
        rule = VolatilityExit(tp_atr_mult=2.0, sl_atr_mult=1.0)
        long_position.meta["entry_atr"] = 10.0
        state = _make_state()
        # Entry=100, ATR=10 -> TP=120, SL=90
        orders = rule.check_exits([long_position], {"BTCUSDT": 121.0}, state)
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "volatility_tp"

    def test_stored_levels_used_on_subsequent_calls(self, long_position):
        rule = VolatilityExit(tp_atr_mult=2.0, sl_atr_mult=1.0)
        state = _make_state()
        state.runtime["atr"] = {"BTCUSDT": 10.0}
        # First call sets levels
        rule.check_exits([long_position], {"BTCUSDT": 105.0}, state)
        # Change ATR - should still use stored levels
        state.runtime["atr"] = {"BTCUSDT": 1.0}
        orders = rule.check_exits([long_position], {"BTCUSDT": 121.0}, state)
        assert len(orders) == 1
        # Should use original TP=120, not new TP=102


# ── CompositeExit ────────────────────────────────────────────────────────────


class TestCompositeExit:
    def test_empty_rules_returns_empty(self, long_position):
        rule = CompositeExit(rules=[])
        orders = rule.check_exits([long_position], {"BTCUSDT": 100.0}, _make_state())
        assert orders == []

    def test_single_rule_delegates(self, long_position):
        tp_sl = TakeProfitStopLossExit(take_profit_pct=0.02)
        rule = CompositeExit(rules=[(tp_sl, 1)])
        orders = rule.check_exits([long_position], {"BTCUSDT": 103.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "take_profit"
        assert orders[0].meta["composite_rule"] == "TakeProfitStopLossExit"

    def test_first_triggered_mode(self, long_position):
        # Both rules would trigger, but first in priority order wins
        tp_sl = TakeProfitStopLossExit(take_profit_pct=0.01)  # TP=101
        time_exit = TimeBasedExit(max_bars=1)
        rule = CompositeExit(
            rules=[(time_exit, 2), (tp_sl, 1)],
            priority_mode=ExitPriority.FIRST_TRIGGERED,
        )
        orders = rule.check_exits([long_position], {"BTCUSDT": 102.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "take_profit"
        assert orders[0].meta["composite_priority"] == 1

    def test_highest_priority_mode(self, long_position):
        # Both trigger, lower priority number wins
        tp_sl = TakeProfitStopLossExit(take_profit_pct=0.01)
        time_exit = TimeBasedExit(max_bars=1)
        rule = CompositeExit(
            rules=[(time_exit, 1), (tp_sl, 2)],  # Time has higher priority
            priority_mode=ExitPriority.HIGHEST_PRIORITY,
        )
        orders = rule.check_exits([long_position], {"BTCUSDT": 102.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "time_exit"
        assert orders[0].meta["composite_priority"] == 1

    def test_all_must_agree_both_trigger(self, long_position):
        tp_sl = TakeProfitStopLossExit(take_profit_pct=0.01)  # TP=101
        time_exit = TimeBasedExit(max_bars=1)
        rule = CompositeExit(
            rules=[(tp_sl, 1), (time_exit, 2)],
            priority_mode=ExitPriority.ALL_MUST_AGREE,
        )
        # Both trigger at 102 (TP hit, time hit on bar 1)
        orders = rule.check_exits([long_position], {"BTCUSDT": 102.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].meta["composite_rule"] == "all_agreed"
        assert orders[0].meta["composite_rules_count"] == 2

    def test_all_must_agree_only_one_triggers(self, long_position):
        tp_sl = TakeProfitStopLossExit(take_profit_pct=0.10)  # TP=110
        time_exit = TimeBasedExit(max_bars=1)
        rule = CompositeExit(
            rules=[(tp_sl, 1), (time_exit, 2)],
            priority_mode=ExitPriority.ALL_MUST_AGREE,
        )
        # Only time triggers (TP not hit), need both to agree
        orders = rule.check_exits([long_position], {"BTCUSDT": 102.0}, _make_state())
        assert orders == []

    def test_multiple_positions_handled(self, long_position, short_position):
        tp_sl = TakeProfitStopLossExit(take_profit_pct=0.02)
        rule = CompositeExit(rules=[(tp_sl, 1)])
        # Long: entry=100, TP=102 -> triggers at 103
        # Short: entry=100, TP=98 -> triggers at 97
        prices = {"BTCUSDT": 103.0, "ETHUSDT": 97.0}
        orders = rule.check_exits([long_position, short_position], prices, _make_state())
        assert len(orders) == 2
        position_ids = {o.position_id for o in orders}
        assert long_position.id in position_ids
        assert short_position.id in position_ids
