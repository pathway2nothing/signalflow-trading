"""Tests for signalflow.core.containers.portfolio.Portfolio."""

import polars as pl
import pytest

from signalflow.core.containers.portfolio import Portfolio
from signalflow.core.containers.position import Position
from signalflow.core.containers.trade import Trade
from signalflow.core.enums import PositionType


@pytest.fixture
def portfolio_with_positions():
    pf = Portfolio(cash=10000.0)
    p1 = Position(
        id="p1",
        pair="BTCUSDT",
        position_type=PositionType.LONG,
        entry_price=100.0,
        last_price=110.0,
        qty=1.0,
    )
    p2 = Position(
        id="p2",
        pair="ETHUSDT",
        position_type=PositionType.LONG,
        entry_price=50.0,
        last_price=55.0,
        qty=2.0,
        is_closed=True,
    )
    pf.positions["p1"] = p1
    pf.positions["p2"] = p2
    return pf


class TestPortfolioBasic:
    def test_defaults(self):
        pf = Portfolio()
        assert pf.cash == 0.0
        assert pf.positions == {}

    def test_open_positions(self, portfolio_with_positions):
        opens = portfolio_with_positions.open_positions()
        assert len(opens) == 1
        assert opens[0].id == "p1"

    def test_open_positions_empty(self):
        pf = Portfolio(cash=1000.0)
        assert pf.open_positions() == []


class TestPortfolioEquity:
    def test_equity_cash_only(self):
        pf = Portfolio(cash=5000.0)
        assert pf.equity(prices={}) == 5000.0

    def test_equity_with_positions(self, portfolio_with_positions):
        eq = portfolio_with_positions.equity(prices={"BTCUSDT": 120.0, "ETHUSDT": 60.0})
        # cash=10000, BTC: 1*120=120, ETH(closed): 1*60*2=120
        # positions include closed ones in equity calc
        assert eq == pytest.approx(10000.0 + 120.0 + 60.0 * 2)

    def test_equity_uses_last_price_fallback(self, portfolio_with_positions):
        eq = portfolio_with_positions.equity(prices={})
        # uses last_price: BTC=110*1, ETH=55*2
        assert eq == pytest.approx(10000.0 + 110.0 + 110.0)


class TestPortfolioDataFrameConversion:
    def test_positions_to_pl(self, portfolio_with_positions):
        positions = list(portfolio_with_positions.positions.values())
        df = Portfolio.positions_to_pl(positions)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert "pair" in df.columns
        assert "qty" in df.columns

    def test_positions_to_pl_empty(self):
        df = Portfolio.positions_to_pl([])
        assert isinstance(df, pl.DataFrame)
        assert df.is_empty()

    def test_trades_to_pl(self):
        trades = [
            Trade(pair="BTCUSDT", side="BUY", price=100.0, qty=1.0, fee=0.1),
            Trade(pair="BTCUSDT", side="SELL", price=110.0, qty=1.0, fee=0.1),
        ]
        df = Portfolio.trades_to_pl(trades)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2

    def test_trades_to_pl_empty(self):
        df = Portfolio.trades_to_pl([])
        assert isinstance(df, pl.DataFrame)
        assert df.is_empty()
