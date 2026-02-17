"""Tests for signalflow.core.containers.portfolio module."""

from datetime import datetime

import polars as pl
import pytest

from signalflow.core.containers.portfolio import Portfolio
from signalflow.core.containers.position import Position
from signalflow.core.containers.trade import Trade
from signalflow.core.enums import PositionType


class TestPortfolioBasics:
    """Tests for basic Portfolio functionality."""

    def test_default_values(self):
        """Portfolio has correct defaults."""
        portfolio = Portfolio()
        assert portfolio.cash == 0.0
        assert portfolio.positions == {}

    def test_initialize_with_cash(self):
        """Portfolio can be initialized with cash."""
        portfolio = Portfolio(cash=10000.0)
        assert portfolio.cash == 10000.0

    def test_initialize_with_positions(self):
        """Portfolio can be initialized with positions dict."""
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        portfolio = Portfolio(cash=10000.0, positions={"pos1": pos})
        assert len(portfolio.positions) == 1


class TestOpenPositions:
    """Tests for open_positions() method."""

    def test_empty_portfolio(self):
        """Empty portfolio returns empty list."""
        portfolio = Portfolio()
        assert portfolio.open_positions() == []

    def test_all_open(self):
        """Returns all open positions."""
        pos1 = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        pos2 = Position(
            id="pos2",
            pair="ETHUSDT",
            position_type=PositionType.SHORT,
            entry_price=3000.0,
            last_price=3000.0,
            qty=1.0,
        )
        portfolio = Portfolio(positions={"pos1": pos1, "pos2": pos2})
        open_pos = portfolio.open_positions()
        assert len(open_pos) == 2

    def test_filters_closed(self):
        """Filters out closed positions."""
        pos1 = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        pos2 = Position(
            id="pos2",
            pair="ETHUSDT",
            position_type=PositionType.SHORT,
            entry_price=3000.0,
            last_price=3000.0,
            qty=1.0,
            is_closed=True,
        )
        portfolio = Portfolio(positions={"pos1": pos1, "pos2": pos2})
        open_pos = portfolio.open_positions()
        assert len(open_pos) == 1
        assert open_pos[0].id == "pos1"


class TestEquity:
    """Tests for equity() method."""

    def test_cash_only(self):
        """Equity equals cash with no positions."""
        portfolio = Portfolio(cash=10000.0)
        assert portfolio.equity(prices={}) == 10000.0

    def test_long_position(self):
        """Equity includes long position value."""
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,  # 0.1 BTC
        )
        portfolio = Portfolio(cash=5000.0, positions={"pos1": pos})
        # Equity = 5000 + (1 * 55000 * 0.1) = 5000 + 5500 = 10500
        equity = portfolio.equity(prices={"BTCUSDT": 55000.0})
        assert equity == 10500.0

    def test_short_position(self):
        """Equity includes short position value (negative sign)."""
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.SHORT,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        portfolio = Portfolio(cash=10000.0, positions={"pos1": pos})
        # Equity = 10000 + (-1 * 55000 * 0.1) = 10000 - 5500 = 4500
        equity = portfolio.equity(prices={"BTCUSDT": 55000.0})
        assert equity == 4500.0

    def test_uses_last_price_if_not_in_dict(self):
        """Falls back to position's last_price if pair not in prices."""
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=52000.0,
            qty=0.1,
        )
        portfolio = Portfolio(cash=5000.0, positions={"pos1": pos})
        # Uses last_price=52000
        equity = portfolio.equity(prices={})
        assert equity == 5000.0 + 52000.0 * 0.1


class TestGrossExposure:
    """Tests for gross_exposure() method."""

    def test_empty_portfolio(self):
        """Empty portfolio has zero gross exposure."""
        portfolio = Portfolio(cash=10000.0)
        assert portfolio.gross_exposure(prices={}) == 0.0

    def test_single_position(self):
        """Gross exposure is notional value."""
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        portfolio = Portfolio(positions={"pos1": pos})
        exposure = portfolio.gross_exposure(prices={"BTCUSDT": 55000.0})
        assert exposure == 5500.0

    def test_ignores_closed_positions(self):
        """Gross exposure ignores closed positions."""
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
            is_closed=True,
        )
        portfolio = Portfolio(positions={"pos1": pos})
        assert portfolio.gross_exposure(prices={"BTCUSDT": 55000.0}) == 0.0

    def test_sums_all_positions(self):
        """Gross exposure sums absolute values."""
        pos1 = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        pos2 = Position(
            id="pos2",
            pair="ETHUSDT",
            position_type=PositionType.SHORT,
            entry_price=3000.0,
            last_price=3000.0,
            qty=1.0,
        )
        portfolio = Portfolio(positions={"pos1": pos1, "pos2": pos2})
        exposure = portfolio.gross_exposure(prices={"BTCUSDT": 50000.0, "ETHUSDT": 3000.0})
        # 50000 * 0.1 + 3000 * 1.0 = 5000 + 3000 = 8000
        assert exposure == 8000.0


class TestNetExposure:
    """Tests for net_exposure() method."""

    def test_empty_portfolio(self):
        """Empty portfolio has zero net exposure."""
        portfolio = Portfolio(cash=10000.0)
        assert portfolio.net_exposure(prices={}) == 0.0

    def test_long_is_positive(self):
        """Long positions contribute positive exposure."""
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        portfolio = Portfolio(positions={"pos1": pos})
        assert portfolio.net_exposure(prices={"BTCUSDT": 50000.0}) == 5000.0

    def test_short_is_negative(self):
        """Short positions contribute negative exposure."""
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.SHORT,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        portfolio = Portfolio(positions={"pos1": pos})
        assert portfolio.net_exposure(prices={"BTCUSDT": 50000.0}) == -5000.0

    def test_long_short_cancel(self):
        """Long and short can cancel out."""
        pos1 = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        pos2 = Position(
            id="pos2",
            pair="BTCUSDT",
            position_type=PositionType.SHORT,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        portfolio = Portfolio(positions={"pos1": pos1, "pos2": pos2})
        assert portfolio.net_exposure(prices={"BTCUSDT": 50000.0}) == 0.0


class TestLeverage:
    """Tests for leverage() method."""

    def test_zero_equity(self):
        """Zero equity returns zero leverage."""
        portfolio = Portfolio(cash=0.0)
        assert portfolio.leverage(prices={}) == 0.0

    def test_no_positions(self):
        """No positions means zero leverage."""
        portfolio = Portfolio(cash=10000.0)
        assert portfolio.leverage(prices={}) == 0.0

    def test_leverage_calculation(self):
        """Leverage = gross exposure / equity."""
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.2,  # 10000 notional
        )
        portfolio = Portfolio(cash=5000.0, positions={"pos1": pos})
        # Gross = 50000 * 0.2 = 10000
        # Equity = 5000 + 10000 = 15000
        # Leverage = 10000 / 15000 = 0.667
        leverage = portfolio.leverage(prices={"BTCUSDT": 50000.0})
        assert abs(leverage - 0.6667) < 0.001


class TestPositionsByPair:
    """Tests for positions_by_pair() method."""

    def test_empty_portfolio(self):
        """Empty portfolio returns empty dict."""
        portfolio = Portfolio()
        assert portfolio.positions_by_pair() == {}

    def test_groups_by_pair(self):
        """Groups positions by pair name."""
        pos1 = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        pos2 = Position(
            id="pos2",
            pair="BTCUSDT",
            position_type=PositionType.SHORT,
            entry_price=51000.0,
            last_price=51000.0,
            qty=0.05,
        )
        pos3 = Position(
            id="pos3",
            pair="ETHUSDT",
            position_type=PositionType.LONG,
            entry_price=3000.0,
            last_price=3000.0,
            qty=1.0,
        )
        portfolio = Portfolio(positions={"pos1": pos1, "pos2": pos2, "pos3": pos3})
        by_pair = portfolio.positions_by_pair()
        assert len(by_pair["BTCUSDT"]) == 2
        assert len(by_pair["ETHUSDT"]) == 1

    def test_open_only_filter(self):
        """Filters closed positions when open_only=True."""
        pos1 = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        pos2 = Position(
            id="pos2",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=51000.0,
            last_price=51000.0,
            qty=0.1,
            is_closed=True,
        )
        portfolio = Portfolio(positions={"pos1": pos1, "pos2": pos2})
        by_pair = portfolio.positions_by_pair(open_only=True)
        assert len(by_pair["BTCUSDT"]) == 1

    def test_include_closed(self):
        """Includes closed positions when open_only=False."""
        pos1 = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        pos2 = Position(
            id="pos2",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=51000.0,
            last_price=51000.0,
            qty=0.1,
            is_closed=True,
        )
        portfolio = Portfolio(positions={"pos1": pos1, "pos2": pos2})
        by_pair = portfolio.positions_by_pair(open_only=False)
        assert len(by_pair["BTCUSDT"]) == 2


class TestPairExposure:
    """Tests for pair_exposure() method."""

    def test_empty_portfolio(self):
        """Empty portfolio has zero pair exposure."""
        portfolio = Portfolio()
        assert portfolio.pair_exposure("BTCUSDT", prices={}) == 0.0

    def test_single_pair(self):
        """Returns exposure for specific pair."""
        pos1 = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        pos2 = Position(
            id="pos2",
            pair="ETHUSDT",
            position_type=PositionType.LONG,
            entry_price=3000.0,
            last_price=3000.0,
            qty=1.0,
        )
        portfolio = Portfolio(positions={"pos1": pos1, "pos2": pos2})
        btc_exposure = portfolio.pair_exposure("BTCUSDT", prices={"BTCUSDT": 50000.0})
        assert btc_exposure == 5000.0

    def test_ignores_other_pairs(self):
        """Only includes positions for the specified pair."""
        pos1 = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        pos2 = Position(
            id="pos2",
            pair="ETHUSDT",
            position_type=PositionType.LONG,
            entry_price=3000.0,
            last_price=3000.0,
            qty=1.0,
        )
        portfolio = Portfolio(positions={"pos1": pos1, "pos2": pos2})
        eth_exposure = portfolio.pair_exposure("ETHUSDT", prices={"ETHUSDT": 3000.0})
        assert eth_exposure == 3000.0


class TestConcentration:
    """Tests for concentration() method."""

    def test_empty_portfolio(self):
        """Empty portfolio returns empty dict."""
        portfolio = Portfolio()
        assert portfolio.concentration(prices={}) == {}

    def test_single_pair(self):
        """Single pair has 100% concentration."""
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        portfolio = Portfolio(positions={"pos1": pos})
        conc = portfolio.concentration(prices={"BTCUSDT": 50000.0})
        assert conc["BTCUSDT"] == 1.0

    def test_multiple_pairs(self):
        """Multiple pairs split concentration."""
        pos1 = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,  # 5000 notional
        )
        pos2 = Position(
            id="pos2",
            pair="ETHUSDT",
            position_type=PositionType.LONG,
            entry_price=3000.0,
            last_price=3000.0,
            qty=1.0,  # 3000 notional
        )
        portfolio = Portfolio(positions={"pos1": pos1, "pos2": pos2})
        conc = portfolio.concentration(prices={"BTCUSDT": 50000.0, "ETHUSDT": 3000.0})
        # Total = 8000, BTC = 5000/8000 = 0.625, ETH = 3000/8000 = 0.375
        assert abs(conc["BTCUSDT"] - 0.625) < 0.001
        assert abs(conc["ETHUSDT"] - 0.375) < 0.001


class TestPositionsToPl:
    """Tests for positions_to_pl() static method."""

    def test_empty_list(self):
        """Empty list returns empty DataFrame."""
        df = Portfolio.positions_to_pl([])
        assert df.height == 0

    def test_converts_positions(self):
        """Converts positions to DataFrame."""
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=51000.0,
            qty=0.1,
            entry_time=datetime(2024, 1, 1),
            last_time=datetime(2024, 1, 2),
            fees_paid=5.0,
            realized_pnl=100.0,
        )
        df = Portfolio.positions_to_pl([pos])
        assert df.height == 1
        assert df["id"][0] == "pos1"
        assert df["pair"][0] == "BTCUSDT"
        assert df["entry_price"][0] == 50000.0
        assert df["qty"][0] == 0.1

    def test_multiple_positions(self):
        """Converts multiple positions."""
        pos1 = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        pos2 = Position(
            id="pos2",
            pair="ETHUSDT",
            position_type=PositionType.SHORT,
            entry_price=3000.0,
            last_price=3000.0,
            qty=1.0,
        )
        df = Portfolio.positions_to_pl([pos1, pos2])
        assert df.height == 2

    def test_includes_all_fields(self):
        """DataFrame includes all position fields."""
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            last_price=50000.0,
            qty=0.1,
        )
        df = Portfolio.positions_to_pl([pos])
        expected_cols = [
            "id",
            "is_closed",
            "pair",
            "position_type",
            "signal_strength",
            "entry_time",
            "last_time",
            "entry_price",
            "last_price",
            "qty",
            "fees_paid",
            "realized_pnl",
            "meta",
        ]
        for col in expected_cols:
            assert col in df.columns


class TestTradesToPl:
    """Tests for trades_to_pl() static method."""

    def test_empty_list(self):
        """Empty list returns empty DataFrame."""
        df = Portfolio.trades_to_pl([])
        assert df.height == 0

    def test_converts_trades(self):
        """Converts trades to DataFrame."""
        trade = Trade(
            id="trade1",
            position_id="pos1",
            pair="BTCUSDT",
            side="buy",
            ts=datetime(2024, 1, 1, 10, 0),
            price=50000.0,
            qty=0.1,
            fee=5.0,
        )
        df = Portfolio.trades_to_pl([trade])
        assert df.height == 1
        assert df["id"][0] == "trade1"
        assert df["pair"][0] == "BTCUSDT"
        assert df["price"][0] == 50000.0

    def test_multiple_trades(self):
        """Converts multiple trades."""
        trade1 = Trade(
            id="trade1",
            position_id="pos1",
            pair="BTCUSDT",
            side="buy",
            ts=datetime(2024, 1, 1),
            price=50000.0,
            qty=0.1,
        )
        trade2 = Trade(
            id="trade2",
            position_id="pos1",
            pair="BTCUSDT",
            side="sell",
            ts=datetime(2024, 1, 2),
            price=51000.0,
            qty=0.1,
        )
        df = Portfolio.trades_to_pl([trade1, trade2])
        assert df.height == 2

    def test_includes_all_fields(self):
        """DataFrame includes all trade fields."""
        trade = Trade(
            id="trade1",
            position_id="pos1",
            pair="BTCUSDT",
            side="buy",
            ts=datetime(2024, 1, 1),
            price=50000.0,
            qty=0.1,
        )
        df = Portfolio.trades_to_pl([trade])
        expected_cols = ["id", "position_id", "pair", "side", "ts", "price", "qty", "fee", "meta"]
        for col in expected_cols:
            assert col in df.columns
