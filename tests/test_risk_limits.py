"""Tests for signalflow.strategy.risk.limits and risk.manager."""

from datetime import datetime

from signalflow.core.containers.order import Order
from signalflow.core.containers.portfolio import Portfolio
from signalflow.core.containers.position import Position
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import PositionType
from signalflow.strategy.risk.limits import (
    DailyLossLimit,
    MaxLeverageLimit,
    MaxPositionsLimit,
    PairExposureLimit,
)
from signalflow.strategy.risk.manager import RiskCheckResult, RiskManager


def _state(cash: float = 10000.0, positions: list[Position] | None = None) -> StrategyState:
    """Create a StrategyState with given cash and positions."""
    s = StrategyState(strategy_id="test")
    s.portfolio.cash = cash
    if positions:
        for p in positions:
            s.portfolio.positions[p.id] = p
    return s


def _long(pair: str = "BTCUSDT", qty: float = 0.1, entry_price: float = 50000.0) -> Position:
    return Position(
        pair=pair,
        position_type=PositionType.LONG,
        entry_price=entry_price,
        last_price=entry_price,
        qty=qty,
    )


def _order(pair: str = "BTCUSDT", qty: float = 0.1) -> Order:
    return Order(pair=pair, side="BUY", order_type="MARKET", qty=qty)


PRICES = {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}
TS = datetime(2024, 6, 15, 12, 0)


# ── MaxLeverageLimit ─────────────────────────────────────────────────────


class TestMaxLeverageLimit:
    def test_passes_under_limit(self):
        state = _state(cash=10000.0)
        limit = MaxLeverageLimit(max_leverage=3.0)
        ok, reason = limit.check([_order(qty=0.01)], state, PRICES, TS)
        assert ok
        assert reason == ""

    def test_rejects_when_current_leverage_exceeds(self):
        # equity = cash + pos_value = -40000 + 50000 = 10000, gross = 50000, leverage = 5x
        pos = _long(qty=1.0)
        state = _state(cash=-40000.0, positions=[pos])
        limit = MaxLeverageLimit(max_leverage=3.0)
        ok, reason = limit.check([_order()], state, PRICES, TS)
        assert not ok
        assert "leverage" in reason

    def test_rejects_when_projected_leverage_exceeds(self):
        state = _state(cash=10000.0)  # no position yet
        limit = MaxLeverageLimit(max_leverage=3.0)
        # Order for 1 BTC = 50000 notional, equity = 10000, leverage 5x projected
        ok, reason = limit.check([_order(qty=1.0)], state, PRICES, TS)
        assert not ok
        assert "projected" in reason

    def test_zero_equity(self):
        state = _state(cash=0.0)
        limit = MaxLeverageLimit(max_leverage=3.0)
        ok, reason = limit.check([_order()], state, PRICES, TS)
        assert not ok
        assert "equity" in reason

    def test_zero_equity_no_orders(self):
        state = _state(cash=0.0)
        limit = MaxLeverageLimit(max_leverage=3.0)
        ok, reason = limit.check([], state, PRICES, TS)
        assert ok


# ── MaxPositionsLimit ────────────────────────────────────────────────────


class TestMaxPositionsLimit:
    def test_passes_under_limit(self):
        state = _state(positions=[_long()])
        limit = MaxPositionsLimit(max_positions=5)
        ok, reason = limit.check([_order()], state, PRICES, TS)
        assert ok

    def test_rejects_over_limit(self):
        positions = [_long(pair=f"PAIR{i}") for i in range(5)]
        state = _state(positions=positions)
        limit = MaxPositionsLimit(max_positions=5)
        ok, reason = limit.check([_order()], state, PRICES, TS)
        assert not ok
        assert "positions" in reason

    def test_existing_position_order_doesnt_count(self):
        pos = _long()
        state = _state(positions=[pos])
        limit = MaxPositionsLimit(max_positions=1)
        # Order with position_id set = closing/modifying existing position
        order = Order(pair="BTCUSDT", side="SELL", qty=0.1, position_id=pos.id)
        ok, _ = limit.check([order], state, PRICES, TS)
        assert ok


# ── PairExposureLimit ────────────────────────────────────────────────────


class TestPairExposureLimit:
    def test_passes_under_limit(self):
        state = _state(cash=100000.0)
        limit = PairExposureLimit(max_pair_pct=0.25)
        ok, reason = limit.check([_order(qty=0.1)], state, PRICES, TS)
        assert ok

    def test_rejects_over_pair_exposure(self):
        state = _state(cash=10000.0)
        limit = PairExposureLimit(max_pair_pct=0.25)
        # Order for 1 BTC = 50000, equity = 10000, 500% > 25%
        ok, reason = limit.check([_order(qty=1.0)], state, PRICES, TS)
        assert not ok
        assert "exposure" in reason

    def test_zero_equity_passes(self):
        state = _state(cash=0.0)
        limit = PairExposureLimit(max_pair_pct=0.25)
        ok, _ = limit.check([_order()], state, PRICES, TS)
        assert ok

    def test_existing_positions_counted(self):
        pos = _long(qty=0.05)  # 2500 notional
        state = _state(cash=10000.0, positions=[pos])
        limit = PairExposureLimit(max_pair_pct=0.25)
        # New 2500 entry = total 5000 on 12500 equity => 40% > 25%
        ok, reason = limit.check([_order(qty=0.05)], state, PRICES, TS)
        assert not ok


# ── DailyLossLimit ───────────────────────────────────────────────────────


class TestDailyLossLimit:
    def test_passes_normal(self):
        state = _state(cash=10000.0)
        limit = DailyLossLimit(max_daily_loss_pct=0.05)
        ok, _ = limit.check([_order()], state, PRICES, TS)
        assert ok

    def test_halts_on_daily_loss(self):
        limit = DailyLossLimit(max_daily_loss_pct=0.05)
        state = _state(cash=10000.0)

        # First check sets the day start equity
        limit.check([], state, PRICES, TS)

        # Simulate loss: reduce cash
        state.portfolio.cash = 9400.0  # 6% loss
        ok, reason = limit.check([_order()], state, PRICES, TS)
        assert not ok
        assert "daily loss" in reason

    def test_stays_halted_same_day(self):
        limit = DailyLossLimit(max_daily_loss_pct=0.05)
        state = _state(cash=10000.0)
        limit.check([], state, PRICES, TS)
        state.portfolio.cash = 9000.0
        limit.check([_order()], state, PRICES, TS)

        # Even with recovery, still halted for the day
        state.portfolio.cash = 10000.0
        ok, reason = limit.check([_order()], state, PRICES, TS)
        assert not ok
        assert "halted" in reason

    def test_resets_next_day(self):
        limit = DailyLossLimit(max_daily_loss_pct=0.05)
        state = _state(cash=10000.0)
        limit.check([], state, PRICES, TS)
        state.portfolio.cash = 9000.0
        limit.check([_order()], state, PRICES, TS)

        # Next day: reset
        next_day = datetime(2024, 6, 16, 12, 0)
        state.portfolio.cash = 9500.0
        ok, _ = limit.check([_order()], state, PRICES, next_day)
        assert ok

    def test_zero_start_equity(self):
        limit = DailyLossLimit(max_daily_loss_pct=0.05)
        state = _state(cash=0.0)
        ok, _ = limit.check([_order()], state, PRICES, TS)
        assert ok


# ── RiskManager ──────────────────────────────────────────────────────────


class TestRiskManager:
    def test_no_orders_allowed(self):
        rm = RiskManager(limits=[MaxLeverageLimit()])
        result = rm.check([], _state(), PRICES, TS)
        assert result.allowed
        assert result.passed_orders == []

    def test_reject_all_mode(self):
        # equity = -40000 + 50000 = 10000, leverage = 5x
        pos = _long(qty=1.0)
        state = _state(cash=-40000.0, positions=[pos])
        rm = RiskManager(limits=[MaxLeverageLimit(max_leverage=3.0)])
        result = rm.check([_order()], state, PRICES, TS)
        assert not result.allowed
        assert len(result.rejected_orders) == 1
        assert len(result.violations) == 1

    def test_passes_all_limits(self):
        state = _state(cash=100000.0)
        rm = RiskManager(
            limits=[
                MaxLeverageLimit(max_leverage=3.0),
                MaxPositionsLimit(max_positions=10),
            ]
        )
        result = rm.check([_order(qty=0.01)], state, PRICES, TS)
        assert result.allowed
        assert len(result.passed_orders) == 1

    def test_filter_mode(self):
        state = _state(cash=10000.0)
        rm = RiskManager(
            limits=[PairExposureLimit(max_pair_pct=0.25)],
            mode="filter",
        )
        orders = [_order(qty=0.01), _order(qty=1.0)]  # small ok, big violates
        result = rm.check(orders, state, PRICES, TS)
        assert len(result.passed_orders) == 1
        assert len(result.rejected_orders) == 1

    def test_halted_rejects_all(self):
        rm = RiskManager(limits=[])
        rm.halt("test halt")
        assert rm.halted
        result = rm.check([_order()], _state(), PRICES, TS)
        assert not result.allowed
        assert "halted" in result.violations[0][1]

    def test_resume_after_halt(self):
        rm = RiskManager(limits=[])
        rm.halt()
        rm.resume()
        assert not rm.halted

    def test_violation_history(self):
        pos = _long(qty=1.0)
        state = _state(cash=-40000.0, positions=[pos])
        rm = RiskManager(limits=[MaxLeverageLimit(max_leverage=3.0)])
        rm.check([_order()], state, PRICES, TS)
        assert len(rm.violation_history) == 1
        assert rm.violation_history[0]["limit"] == "MaxLeverageLimit"

    def test_disabled_limit_skipped(self):
        pos = _long(qty=1.0)
        state = _state(cash=10000.0, positions=[pos])
        rm = RiskManager(limits=[MaxLeverageLimit(max_leverage=3.0, enabled=False)])
        result = rm.check([_order()], state, PRICES, TS)
        assert result.allowed

    def test_filter_mode_disabled_limit(self):
        state = _state(cash=10000.0)
        rm = RiskManager(
            limits=[PairExposureLimit(max_pair_pct=0.001, enabled=False)],
            mode="filter",
        )
        result = rm.check([_order(qty=1.0)], state, PRICES, TS)
        assert result.allowed
