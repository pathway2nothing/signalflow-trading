"""Tests for GridExit — grid-level exit rule."""

from signalflow.core.containers.strategy_state import StrategyState
from signalflow.strategy.component.exit.grid_exit import GridExit


def _make_state(initial_capital: float = 10000.0) -> StrategyState:
    state = StrategyState(strategy_id="test")
    state.runtime["initial_capital"] = initial_capital
    return state


# ── Empty / edge cases ────────────────────────────────────────────────────────


class TestGridExitEmpty:
    def test_no_positions_returns_empty(self):
        rule = GridExit(grid_profit_target=0.10)
        assert rule.check_exits([], {"BTCUSDT": 100.0}, _make_state()) == []

    def test_all_closed_returns_empty(self, long_position):
        long_position.is_closed = True
        rule = GridExit(grid_profit_target=0.10)
        assert rule.check_exits([long_position], {"BTCUSDT": 200.0}, _make_state()) == []

    def test_no_price_for_pair_skipped(self, long_position):
        rule = GridExit(grid_profit_target=0.001)
        orders = rule.check_exits([long_position], {}, _make_state())
        assert orders == []

    def test_no_targets_returns_empty(self, long_position):
        rule = GridExit()
        orders = rule.check_exits([long_position], {"BTCUSDT": 200.0}, _make_state())
        assert orders == []


# ── Grid Profit Target ────────────────────────────────────────────────────────


class TestGridExitProfitTarget:
    def test_below_target_no_exit(self, long_position):
        rule = GridExit(grid_profit_target=0.10)
        # entry=100, qty=1, price=105 → PnL=5, 5/10000=0.05% < 10%
        orders = rule.check_exits([long_position], {"BTCUSDT": 105.0}, _make_state())
        assert orders == []

    def test_at_target_triggers_close_all(self, long_position):
        rule = GridExit(grid_profit_target=0.10)
        # entry=100, qty=1, price=1100 → PnL=1000, 1000/10000=10%
        orders = rule.check_exits([long_position], {"BTCUSDT": 1100.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "grid_profit_target"
        assert orders[0].side == "SELL"
        assert orders[0].position_id == long_position.id

    def test_closes_all_positions(self, long_position, short_position):
        rule = GridExit(grid_profit_target=0.05)
        # LONG: entry=100, qty=1, price=400 → PnL=300
        # SHORT: entry=100, qty=1, price=50 → PnL=50
        # total=350, 350/10000=3.5% ... need higher prices
        # LONG: entry=100, qty=1, price=600 → PnL=500
        # SHORT: entry=100, qty=1, price=50 → PnL=50
        # total=550, 550/10000=5.5% >= 5%
        orders = rule.check_exits(
            [long_position, short_position],
            {"BTCUSDT": 600.0, "ETHUSDT": 50.0},
            _make_state(),
        )
        assert len(orders) == 2
        position_ids = {o.position_id for o in orders}
        assert long_position.id in position_ids
        assert short_position.id in position_ids

    def test_meta_contains_grid_pnl_pct(self, long_position):
        rule = GridExit(grid_profit_target=0.05)
        # PnL = 1000, pct = 0.10
        orders = rule.check_exits([long_position], {"BTCUSDT": 1100.0}, _make_state())
        assert orders[0].meta["grid_pnl_pct"] >= 0.05


# ── Grid Loss Limit ──────────────────────────────────────────────────────────


class TestGridExitLossLimit:
    def test_above_limit_no_exit(self, long_position):
        rule = GridExit(grid_loss_limit=0.05)
        # entry=100, price=99 → PnL=-1, -1/10000=-0.01% > -5%
        orders = rule.check_exits([long_position], {"BTCUSDT": 99.0}, _make_state())
        assert orders == []

    def test_at_limit_triggers_close_all(self, long_position):
        rule = GridExit(grid_loss_limit=0.05)
        # entry=100, qty=1, price needs PnL=-500 → price=100-500=-400? No.
        # Use capital=1000 instead
        state = _make_state(initial_capital=1000.0)
        # PnL = (50 - 100) * 1 = -50, -50/1000 = -5% <= -5%
        orders = rule.check_exits([long_position], {"BTCUSDT": 50.0}, state)
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "grid_loss_limit"
        assert orders[0].side == "SELL"

    def test_short_loss_limit(self, short_position):
        rule = GridExit(grid_loss_limit=0.05)
        state = _make_state(initial_capital=1000.0)
        # SHORT entry=100, price=150 → PnL = (100-150)*1 = -50, -50/1000 = -5%
        orders = rule.check_exits([short_position], {"ETHUSDT": 150.0}, state)
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "grid_loss_limit"
        assert orders[0].side == "BUY"


# ── Stale Level ──────────────────────────────────────────────────────────────


class TestGridExitStaleLevel:
    def test_within_range_no_exit(self, long_position):
        rule = GridExit(max_distance_pct=0.10)
        # entry=100, price=105 → distance=5% < 10%
        orders = rule.check_exits([long_position], {"BTCUSDT": 105.0}, _make_state())
        assert orders == []

    def test_above_range_triggers_exit(self, long_position):
        rule = GridExit(max_distance_pct=0.10)
        # entry=100, price=115 → distance=15% > 10%
        orders = rule.check_exits([long_position], {"BTCUSDT": 115.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "grid_stale_level"
        assert orders[0].meta["distance_pct"] > 0.10

    def test_below_range_triggers_exit(self, long_position):
        rule = GridExit(max_distance_pct=0.10)
        # entry=100, price=85 → distance=15% > 10%
        orders = rule.check_exits([long_position], {"BTCUSDT": 85.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "grid_stale_level"

    def test_only_stale_positions_closed(self, long_position, short_position):
        rule = GridExit(max_distance_pct=0.10)
        # LONG: entry=100, price=115 → 15% > 10% → close
        # SHORT: entry=100, price=115 → 15% > 10% → close too
        orders = rule.check_exits(
            [long_position, short_position],
            {"BTCUSDT": 115.0, "ETHUSDT": 102.0},
            _make_state(),
        )
        # Only LONG at 15% distance, SHORT at 2%
        assert len(orders) == 1
        assert orders[0].position_id == long_position.id

    def test_short_stale_exit_is_buy(self, short_position):
        rule = GridExit(max_distance_pct=0.05)
        # entry=100, price=80 → 20% > 5%
        orders = rule.check_exits([short_position], {"ETHUSDT": 80.0}, _make_state())
        assert len(orders) == 1
        assert orders[0].side == "BUY"


# ── Priority: profit/loss checked before stale ────────────────────────────────


class TestGridExitPriority:
    def test_profit_target_takes_precedence_over_stale(self, long_position):
        rule = GridExit(grid_profit_target=0.05, max_distance_pct=0.10)
        state = _make_state(initial_capital=1000.0)
        # entry=100, price=200 → PnL=100, 100/1000=10% >= 5%
        # Also distance = 100% > 10%
        orders = rule.check_exits([long_position], {"BTCUSDT": 200.0}, state)
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "grid_profit_target"

    def test_loss_limit_takes_precedence_over_stale(self, long_position):
        rule = GridExit(grid_loss_limit=0.05, max_distance_pct=0.50)
        state = _make_state(initial_capital=1000.0)
        # entry=100, price=40 → PnL=-60, -60/1000=-6% <= -5%
        # Also distance=60% > 50%
        orders = rule.check_exits([long_position], {"BTCUSDT": 40.0}, state)
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "grid_loss_limit"


# ── Fallback capital calculation ─────────────────────────────────────────────


class TestGridExitCapitalFallback:
    def test_no_initial_capital_uses_notional(self, long_position):
        rule = GridExit(grid_profit_target=0.50)
        state = StrategyState(strategy_id="test")
        # No initial_capital in runtime → fallback to entry notional = 100*1=100
        # entry=100, price=200 → PnL=100, 100/100=100% >= 50%
        orders = rule.check_exits([long_position], {"BTCUSDT": 200.0}, state)
        assert len(orders) == 1
        assert orders[0].meta["exit_reason"] == "grid_profit_target"
