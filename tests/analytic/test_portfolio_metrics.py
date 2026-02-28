"""Tests for signalflow.analytic.strategy.portfolio_metrics module."""

from datetime import datetime

import pytest

from signalflow.analytic.strategy.portfolio_metrics import (
    PortfolioExposureMetric,
    PortfolioPnLBreakdownMetric,
)
from signalflow.core.containers.position import Position
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import PositionType


@pytest.fixture
def state_with_positions():
    """StrategyState with open positions for testing."""
    state = StrategyState(strategy_id="test")
    state.portfolio.cash = 10000.0

    # Add a LONG position in profit
    pos1 = Position(
        id="pos_1",
        pair="BTCUSDT",
        position_type=PositionType.LONG,
        entry_price=100.0,
        last_price=110.0,  # +10%
        qty=1.0,
        entry_time=datetime(2024, 1, 1),
        last_time=datetime(2024, 1, 1),
        realized_pnl=50.0,
    )

    # Add a SHORT position in loss
    pos2 = Position(
        id="pos_2",
        pair="ETHUSDT",
        position_type=PositionType.SHORT,
        entry_price=100.0,
        last_price=105.0,  # -5%
        qty=1.0,
        entry_time=datetime(2024, 1, 1),
        last_time=datetime(2024, 1, 1),
        realized_pnl=-20.0,
    )

    state.portfolio.positions = {"pos_1": pos1, "pos_2": pos2}
    return state


@pytest.fixture
def empty_state():
    """StrategyState with no positions."""
    state = StrategyState(strategy_id="test")
    state.portfolio.cash = 10000.0
    return state


class TestPortfolioExposureMetric:
    """Tests for PortfolioExposureMetric."""

    def test_compute_with_positions(self, state_with_positions):
        metric = PortfolioExposureMetric()
        prices = {"BTCUSDT": 110.0, "ETHUSDT": 105.0}

        result = metric.compute(state_with_positions, prices)

        assert "gross_exposure" in result
        assert "net_exposure" in result
        assert "leverage" in result
        assert "n_pairs" in result
        assert "max_pair_pct" in result
        assert result["n_pairs"] == 2
        assert result["gross_exposure"] > 0

    def test_compute_empty_portfolio(self, empty_state):
        metric = PortfolioExposureMetric()
        prices = {"BTCUSDT": 100.0}

        result = metric.compute(empty_state, prices)

        assert result["gross_exposure"] == 0.0
        assert result["net_exposure"] == 0.0
        assert result["leverage"] == 0.0
        assert result["n_pairs"] == 0
        assert result["max_pair_pct"] == 0.0


class TestPortfolioPnLBreakdownMetric:
    """Tests for PortfolioPnLBreakdownMetric."""

    def test_compute_with_positions(self, state_with_positions):
        metric = PortfolioPnLBreakdownMetric()
        prices = {"BTCUSDT": 110.0, "ETHUSDT": 105.0}

        result = metric.compute(state_with_positions, prices)

        assert "per_pair_realized" in result
        assert "best_pair_pnl" in result
        assert "worst_pair_pnl" in result
        assert "pair_count_open" in result

        # Best pair should be greater than worst pair
        assert result["best_pair_pnl"] >= result["worst_pair_pnl"]
        # pair_count_open should count distinct pairs
        assert result["pair_count_open"] == 2.0

    def test_compute_empty_portfolio(self, empty_state):
        metric = PortfolioPnLBreakdownMetric()
        prices = {"BTCUSDT": 100.0}

        result = metric.compute(empty_state, prices)

        assert result["per_pair_realized"] == 0.0
        assert result["best_pair_pnl"] == 0.0
        assert result["worst_pair_pnl"] == 0.0
        assert result["pair_count_open"] == 0.0

    def test_compute_single_position(self):
        """Test with single position to verify edge case."""
        state = StrategyState(strategy_id="test")
        state.portfolio.cash = 10000.0

        pos = Position(
            id="pos_1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=100.0,
            last_price=120.0,
            qty=1.0,
            entry_time=datetime(2024, 1, 1),
            last_time=datetime(2024, 1, 1),
            realized_pnl=100.0,
        )
        state.portfolio.positions = {"pos_1": pos}

        metric = PortfolioPnLBreakdownMetric()
        prices = {"BTCUSDT": 120.0}

        result = metric.compute(state, prices)

        # With single position, best == worst
        assert result["best_pair_pnl"] == result["worst_pair_pnl"]
        assert result["pair_count_open"] == 1.0
