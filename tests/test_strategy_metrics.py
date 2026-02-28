"""Tests for analytic.strategy.main_strategy_metrics."""

from datetime import datetime

import pytest

from signalflow.analytic.strategy.main_strategy_metrics import (
    BalanceAllocationMetric,
    DrawdownMetric,
    SharpeRatioMetric,
    TotalReturnMetric,
    WinRateMetric,
)
from signalflow.core.containers.position import Position
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import PositionType

TS = datetime(2024, 1, 1)


def _state(cash=10000.0):
    s = StrategyState(strategy_id="test")
    s.portfolio.cash = cash
    return s


def _add_position(
    state, pair="BTCUSDT", entry_price=100.0, qty=1.0, last_price=100.0, is_closed=False, realized_pnl=0.0, fees=0.0
):
    pos = Position(
        id=f"pos_{pair}_{entry_price}",
        pair=pair,
        position_type=PositionType.LONG,
        entry_price=entry_price,
        last_price=last_price,
        qty=qty,
        is_closed=is_closed,
        realized_pnl=realized_pnl,
        fees_paid=fees,
    )
    state.portfolio.positions[pos.id] = pos
    return pos


# ── TotalReturnMetric ────────────────────────────────────────────────────


class TestTotalReturnMetric:
    def test_no_positions(self):
        m = TotalReturnMetric(initial_capital=10000.0)
        s = _state(cash=10000.0)
        result = m.compute(s, prices={})
        assert result["equity"] == pytest.approx(10000.0)
        assert result["total_return"] == pytest.approx(0.0)
        assert result["open_positions"] == 0
        assert result["closed_positions"] == 0

    def test_with_open_position(self):
        m = TotalReturnMetric(initial_capital=10000.0)
        s = _state(cash=9000.0)
        _add_position(s, entry_price=100.0, qty=1.0, last_price=110.0)
        result = m.compute(s, prices={"BTCUSDT": 110.0})
        assert result["equity"] == pytest.approx(9110.0)
        assert result["open_positions"] == 1

    def test_with_closed_position(self):
        m = TotalReturnMetric(initial_capital=10000.0)
        s = _state(cash=10050.0)
        _add_position(s, is_closed=True, realized_pnl=50.0, fees=1.0)
        result = m.compute(s, prices={})
        assert result["realized_pnl"] == pytest.approx(50.0)
        assert result["total_fees"] == pytest.approx(1.0)
        assert result["closed_positions"] == 1

    def test_zero_capital(self):
        m = TotalReturnMetric(initial_capital=0.0)
        s = _state(cash=0.0)
        result = m.compute(s, prices={})
        assert result["total_return"] == 0.0


# ── BalanceAllocationMetric ──────────────────────────────────────────────


class TestBalanceAllocationMetric:
    def test_no_positions(self):
        m = BalanceAllocationMetric(initial_capital=10000.0)
        s = _state(cash=10000.0)
        result = m.compute(s, prices={})
        assert result["capital_utilization"] == pytest.approx(0.0)
        assert result["free_balance_pct"] == pytest.approx(1.0)

    def test_with_position(self):
        m = BalanceAllocationMetric(initial_capital=10000.0)
        s = _state(cash=9000.0)
        _add_position(s, entry_price=100.0, qty=1.0, last_price=100.0)
        result = m.compute(s, prices={"BTCUSDT": 100.0})
        assert result["allocated_value"] == pytest.approx(100.0)
        assert result["capital_utilization"] > 0

    def test_zero_equity(self):
        m = BalanceAllocationMetric(initial_capital=10000.0)
        s = _state(cash=0.0)
        result = m.compute(s, prices={})
        assert result["capital_utilization"] == 0.0


# ── DrawdownMetric ───────────────────────────────────────────────────────


class TestDrawdownMetric:
    def test_no_drawdown(self):
        m = DrawdownMetric()
        s = _state(cash=10000.0)
        result = m.compute(s, prices={})
        assert result["current_drawdown"] == pytest.approx(0.0)
        assert result["max_drawdown"] == pytest.approx(0.0)
        assert result["peak_equity"] == pytest.approx(10000.0)

    def test_drawdown_after_loss(self):
        m = DrawdownMetric()
        s = _state(cash=10000.0)
        m.compute(s, prices={})  # peak = 10000
        s.portfolio.cash = 9000.0
        result = m.compute(s, prices={})
        assert result["current_drawdown"] == pytest.approx(0.1)
        assert result["max_drawdown"] == pytest.approx(0.1)

    def test_recovery_resets_drawdown(self):
        m = DrawdownMetric()
        s = _state(cash=10000.0)
        m.compute(s, prices={})
        s.portfolio.cash = 9000.0
        m.compute(s, prices={})
        s.portfolio.cash = 11000.0
        result = m.compute(s, prices={})
        assert result["current_drawdown"] == pytest.approx(0.0)
        assert result["peak_equity"] == pytest.approx(11000.0)


# ── WinRateMetric ────────────────────────────────────────────────────────


class TestWinRateMetric:
    def test_no_closed_positions(self):
        m = WinRateMetric()
        s = _state()
        result = m.compute(s, prices={})
        assert result["win_rate"] == 0.0
        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 0

    def test_all_winners(self):
        m = WinRateMetric()
        s = _state()
        _add_position(s, pair="A", is_closed=True, realized_pnl=10.0)
        _add_position(s, pair="B", is_closed=True, realized_pnl=20.0)
        result = m.compute(s, prices={})
        assert result["win_rate"] == pytest.approx(1.0)
        assert result["winning_trades"] == 2

    def test_mixed(self):
        m = WinRateMetric()
        s = _state()
        _add_position(s, pair="A", is_closed=True, realized_pnl=10.0)
        _add_position(s, pair="B", is_closed=True, realized_pnl=-5.0)
        result = m.compute(s, prices={})
        assert result["win_rate"] == pytest.approx(0.5)
        assert result["winning_trades"] == 1
        assert result["losing_trades"] == 1


# ── SharpeRatioMetric ────────────────────────────────────────────────────


class TestSharpeRatioMetric:
    def test_single_observation(self):
        m = SharpeRatioMetric(initial_capital=10000.0)
        s = _state(cash=10000.0)
        result = m.compute(s, prices={})
        assert result["sharpe_ratio"] == 0.0

    def test_two_observations(self):
        m = SharpeRatioMetric(initial_capital=10000.0)
        s = _state(cash=10000.0)
        m.compute(s, prices={})
        s.portfolio.cash = 10100.0
        result = m.compute(s, prices={})
        # With 2 observations and non-zero std, should return a number
        assert isinstance(result["sharpe_ratio"], float)

    def test_constant_equity_zero_sharpe(self):
        m = SharpeRatioMetric(initial_capital=10000.0)
        s = _state(cash=10000.0)
        for _ in range(10):
            m.compute(s, prices={})
        result = m.compute(s, prices={})
        assert result["sharpe_ratio"] == 0.0

    def test_window_trimming(self):
        m = SharpeRatioMetric(initial_capital=10000.0, window_size=5)
        s = _state(cash=10000.0)
        for i in range(10):
            s.portfolio.cash = 10000.0 + i * 10
            m.compute(s, prices={})
        assert len(m._returns_history) == 5
