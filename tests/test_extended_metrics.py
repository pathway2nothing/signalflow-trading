"""Tests for analytic.strategy.extended_metrics."""

from datetime import datetime, timedelta

import pytest

from signalflow.analytic.strategy.extended_metrics import (
    AverageTradeMetric,
    CalmarRatioMetric,
    ExpectancyMetric,
    MaxConsecutiveMetric,
    ProfitFactorMetric,
    RiskRewardMetric,
    SortinoRatioMetric,
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
    state,
    pair="BTCUSDT",
    entry_price=100.0,
    qty=1.0,
    last_price=100.0,
    is_closed=False,
    realized_pnl=0.0,
    fees=0.0,
    entry_time=None,
    last_time=None,
):
    pos = Position(
        id=f"pos_{pair}_{entry_price}_{realized_pnl}",
        pair=pair,
        position_type=PositionType.LONG,
        entry_price=entry_price,
        last_price=last_price,
        qty=qty,
        is_closed=is_closed,
        realized_pnl=realized_pnl,
        fees_paid=fees,
        entry_time=entry_time,
        last_time=last_time,
    )
    state.portfolio.positions[pos.id] = pos
    return pos


# ── SortinoRatioMetric ──────────────────────────────────────────────────


class TestSortinoRatioMetric:
    def test_single_observation(self):
        m = SortinoRatioMetric(initial_capital=10000.0)
        s = _state(cash=10000.0)
        result = m.compute(s, prices={})
        assert result["sortino_ratio"] == 0.0

    def test_two_observations(self):
        m = SortinoRatioMetric(initial_capital=10000.0)
        s = _state(cash=10000.0)
        m.compute(s, prices={})
        s.portfolio.cash = 10100.0
        result = m.compute(s, prices={})
        # Only 2 observations, still not enough downside data
        assert "sortino_ratio" in result

    def test_with_downside(self):
        m = SortinoRatioMetric(initial_capital=10000.0, window_size=10)
        s = _state(cash=10000.0)

        # Simulate returns with downside
        equities = [10000, 10100, 10050, 10020, 10080, 10030, 10100, 10150]
        for eq in equities:
            s.portfolio.cash = eq
            result = m.compute(s, prices={})

        assert "sortino_ratio" in result


# ── CalmarRatioMetric ───────────────────────────────────────────────────


class TestCalmarRatioMetric:
    def test_no_drawdown(self):
        m = CalmarRatioMetric(initial_capital=10000.0)
        s = _state(cash=10000.0)
        result = m.compute(s, prices={})
        assert result["calmar_ratio"] == 0.0
        assert result["max_drawdown_calmar"] == 0.0

    def test_with_drawdown(self):
        m = CalmarRatioMetric(initial_capital=10000.0)
        s = _state(cash=10000.0)

        # Initial
        m.compute(s, prices={})

        # Go up
        s.portfolio.cash = 11000.0
        m.compute(s, prices={})

        # Drawdown
        s.portfolio.cash = 10500.0
        result = m.compute(s, prices={})

        assert result["max_drawdown_calmar"] > 0
        assert result["calmar_ratio"] >= 0


# ── ProfitFactorMetric ──────────────────────────────────────────────────


class TestProfitFactorMetric:
    def test_no_trades(self):
        m = ProfitFactorMetric()
        s = _state()
        result = m.compute(s, prices={})
        assert result["profit_factor"] == 0.0
        assert result["gross_profit"] == 0.0
        assert result["gross_loss"] == 0.0

    def test_all_winners(self):
        m = ProfitFactorMetric()
        s = _state()
        _add_position(s, is_closed=True, realized_pnl=100.0)
        _add_position(s, pair="ETHUSDT", is_closed=True, realized_pnl=50.0)
        result = m.compute(s, prices={})
        assert result["gross_profit"] == pytest.approx(150.0)
        assert result["gross_loss"] == 0.0
        assert result["profit_factor"] == float("inf")

    def test_all_losers(self):
        m = ProfitFactorMetric()
        s = _state()
        _add_position(s, is_closed=True, realized_pnl=-100.0)
        result = m.compute(s, prices={})
        assert result["gross_loss"] == pytest.approx(100.0)
        assert result["profit_factor"] == 0.0

    def test_mixed_trades(self):
        m = ProfitFactorMetric()
        s = _state()
        _add_position(s, is_closed=True, realized_pnl=200.0)
        _add_position(s, pair="ETHUSDT", is_closed=True, realized_pnl=-100.0)
        result = m.compute(s, prices={})
        assert result["profit_factor"] == pytest.approx(2.0)


# ── AverageTradeMetric ──────────────────────────────────────────────────


class TestAverageTradeMetric:
    def test_no_trades(self):
        m = AverageTradeMetric()
        s = _state()
        result = m.compute(s, prices={})
        assert result["avg_profit"] == 0.0
        assert result["avg_loss"] == 0.0
        assert result["avg_trade"] == 0.0
        assert result["avg_duration_minutes"] == 0.0

    def test_with_trades(self):
        m = AverageTradeMetric()
        s = _state()
        _add_position(s, is_closed=True, realized_pnl=100.0)
        _add_position(s, pair="ETHUSDT", is_closed=True, realized_pnl=-50.0)
        result = m.compute(s, prices={})
        assert result["avg_profit"] == pytest.approx(100.0)
        assert result["avg_loss"] == pytest.approx(-50.0)
        assert result["avg_trade"] == pytest.approx(25.0)

    def test_duration_calculation(self):
        m = AverageTradeMetric()
        s = _state()
        _add_position(
            s,
            is_closed=True,
            realized_pnl=100.0,
            entry_time=TS,
            last_time=TS + timedelta(hours=1),
        )
        result = m.compute(s, prices={})
        assert result["avg_duration_minutes"] == pytest.approx(60.0)
        assert result["avg_win_duration"] == pytest.approx(60.0)


# ── ExpectancyMetric ────────────────────────────────────────────────────


class TestExpectancyMetric:
    def test_no_trades(self):
        m = ExpectancyMetric()
        s = _state()
        result = m.compute(s, prices={})
        assert result["expectancy"] == 0.0
        assert result["expectancy_ratio"] == 0.0

    def test_positive_expectancy(self):
        m = ExpectancyMetric()
        s = _state()
        # 2 wins of 100, 1 loss of 50
        _add_position(s, is_closed=True, realized_pnl=100.0)
        _add_position(s, pair="ETHUSDT", is_closed=True, realized_pnl=100.0)
        _add_position(s, pair="SOLUSDT", is_closed=True, realized_pnl=-50.0)
        result = m.compute(s, prices={})
        # win_rate = 2/3, avg_win = 100, loss_rate = 1/3, avg_loss = 50
        # expectancy = (2/3 * 100) - (1/3 * 50) = 66.67 - 16.67 = 50
        assert result["expectancy"] == pytest.approx(50.0, rel=0.01)

    def test_negative_expectancy(self):
        m = ExpectancyMetric()
        s = _state()
        # 1 win of 50, 2 losses of 100 each
        _add_position(s, is_closed=True, realized_pnl=50.0)
        _add_position(s, pair="ETHUSDT", is_closed=True, realized_pnl=-100.0)
        _add_position(s, pair="SOLUSDT", is_closed=True, realized_pnl=-100.0)
        result = m.compute(s, prices={})
        assert result["expectancy"] < 0


# ── RiskRewardMetric ────────────────────────────────────────────────────


class TestRiskRewardMetric:
    def test_no_trades(self):
        m = RiskRewardMetric()
        s = _state()
        result = m.compute(s, prices={})
        assert result["risk_reward_ratio"] == 0.0

    def test_ratio_calculation(self):
        m = RiskRewardMetric()
        s = _state()
        _add_position(s, is_closed=True, realized_pnl=200.0)
        _add_position(s, pair="ETHUSDT", is_closed=True, realized_pnl=-100.0)
        result = m.compute(s, prices={})
        assert result["risk_reward_ratio"] == pytest.approx(2.0)
        assert result["payoff_ratio"] == pytest.approx(2.0)

    def test_no_losses(self):
        m = RiskRewardMetric()
        s = _state()
        _add_position(s, is_closed=True, realized_pnl=100.0)
        result = m.compute(s, prices={})
        assert result["risk_reward_ratio"] == 0.0


# ── MaxConsecutiveMetric ────────────────────────────────────────────────


class TestMaxConsecutiveMetric:
    def test_no_trades(self):
        m = MaxConsecutiveMetric()
        s = _state()
        result = m.compute(s, prices={})
        assert result["max_consecutive_wins"] == 0
        assert result["max_consecutive_losses"] == 0

    def test_win_streak(self):
        m = MaxConsecutiveMetric()
        s = _state()

        # Add 3 winning trades
        for i in range(3):
            _add_position(
                s,
                pair=f"PAIR{i}",
                is_closed=True,
                realized_pnl=100.0,
                last_time=TS + timedelta(minutes=i),
            )

        result = m.compute(s, prices={})
        assert result["max_consecutive_wins"] == 3
        assert result["current_win_streak"] == 3

    def test_loss_streak(self):
        m = MaxConsecutiveMetric()
        s = _state()

        # Add 2 losing trades
        for i in range(2):
            _add_position(
                s,
                pair=f"PAIR{i}",
                is_closed=True,
                realized_pnl=-50.0,
                last_time=TS + timedelta(minutes=i),
            )

        result = m.compute(s, prices={})
        assert result["max_consecutive_losses"] == 2
        assert result["current_loss_streak"] == 2

    def test_alternating(self):
        m = MaxConsecutiveMetric()
        s = _state()

        # Win, Loss, Win, Loss
        pnls = [100, -50, 100, -50]
        for i, pnl in enumerate(pnls):
            _add_position(
                s,
                pair=f"PAIR{i}",
                is_closed=True,
                realized_pnl=float(pnl),
                last_time=TS + timedelta(minutes=i),
            )

        result = m.compute(s, prices={})
        assert result["max_consecutive_wins"] == 1
        assert result["max_consecutive_losses"] == 1
