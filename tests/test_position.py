"""Tests for signalflow.core.containers.position.Position."""

from datetime import datetime

import pytest

from signalflow.core.containers.position import Position
from signalflow.core.containers.trade import Trade
from signalflow.core.enums import PositionType


class TestPositionCreation:
    def test_defaults(self):
        p = Position()
        assert p.pair == ""
        assert p.position_type == PositionType.LONG
        assert p.qty == 0.0
        assert p.is_closed is False
        assert p.id  # uuid generated

    def test_custom(self):
        p = Position(
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=45000.0,
            qty=0.5,
        )
        assert p.pair == "BTCUSDT"
        assert p.entry_price == 45000.0
        assert p.qty == 0.5


class TestPositionProperties:
    def test_side_sign_long(self):
        p = Position(position_type=PositionType.LONG)
        assert p.side_sign == 1.0

    def test_side_sign_short(self):
        p = Position(position_type=PositionType.SHORT)
        assert p.side_sign == -1.0

    def test_unrealized_pnl_long_profit(self):
        p = Position(
            position_type=PositionType.LONG,
            entry_price=100.0,
            last_price=110.0,
            qty=2.0,
        )
        assert p.unrealized_pnl == pytest.approx(20.0)

    def test_unrealized_pnl_long_loss(self):
        p = Position(
            position_type=PositionType.LONG,
            entry_price=100.0,
            last_price=90.0,
            qty=2.0,
        )
        assert p.unrealized_pnl == pytest.approx(-20.0)

    def test_unrealized_pnl_short_profit(self):
        p = Position(
            position_type=PositionType.SHORT,
            entry_price=100.0,
            last_price=90.0,
            qty=2.0,
        )
        # short: -1 * (90 - 100) * 2 = 20
        assert p.unrealized_pnl == pytest.approx(20.0)

    def test_total_pnl(self):
        p = Position(
            position_type=PositionType.LONG,
            entry_price=100.0,
            last_price=110.0,
            qty=1.0,
            realized_pnl=5.0,
            fees_paid=2.0,
        )
        # 5 + 10 - 2 = 13
        assert p.total_pnl == pytest.approx(13.0)


class TestPositionMark:
    def test_mark_updates_price_and_time(self):
        p = Position(entry_price=100.0, qty=1.0)
        ts = datetime(2024, 6, 1)
        p.mark(ts=ts, price=105.0)
        assert p.last_price == 105.0
        assert p.last_time == ts


class TestPositionApplyTrade:
    def test_increase_long(self):
        p = Position(
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=0.0,
            qty=0.0,
        )
        trade = Trade(pair="BTCUSDT", side="BUY", price=100.0, qty=1.0, fee=0.1)
        p.apply_trade(trade)
        assert p.qty == 1.0
        assert p.entry_price == 100.0
        assert p.fees_paid == pytest.approx(0.1)

    def test_increase_averages_price(self):
        p = Position(
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=100.0,
            qty=1.0,
        )
        trade = Trade(pair="BTCUSDT", side="BUY", price=200.0, qty=1.0, fee=0.0)
        p.apply_trade(trade)
        assert p.qty == 2.0
        assert p.entry_price == pytest.approx(150.0)

    def test_decrease_long_partial(self):
        p = Position(
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=100.0,
            qty=2.0,
        )
        trade = Trade(pair="BTCUSDT", side="SELL", price=150.0, qty=1.0, fee=0.0)
        p.apply_trade(trade)
        assert p.qty == 1.0
        assert p.realized_pnl == pytest.approx(50.0)
        assert p.is_closed is False

    def test_decrease_long_full_close(self):
        p = Position(
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=100.0,
            qty=1.0,
        )
        trade = Trade(pair="BTCUSDT", side="SELL", price=120.0, qty=1.0, fee=0.5)
        p.apply_trade(trade)
        assert p.qty == 0.0
        assert p.realized_pnl == pytest.approx(20.0)
        assert p.is_closed is True
        assert p.fees_paid == pytest.approx(0.5)

    def test_short_increase(self):
        p = Position(
            pair="BTCUSDT",
            position_type=PositionType.SHORT,
            entry_price=0.0,
            qty=0.0,
        )
        trade = Trade(pair="BTCUSDT", side="SELL", price=100.0, qty=1.0, fee=0.0)
        p.apply_trade(trade)
        assert p.qty == 1.0
        assert p.entry_price == 100.0

    def test_short_decrease_profit(self):
        p = Position(
            pair="BTCUSDT",
            position_type=PositionType.SHORT,
            entry_price=100.0,
            qty=1.0,
        )
        trade = Trade(pair="BTCUSDT", side="BUY", price=80.0, qty=1.0, fee=0.0)
        p.apply_trade(trade)
        # short pnl: -1 * (80 - 100) * 1 = 20
        assert p.realized_pnl == pytest.approx(20.0)
        assert p.is_closed is True
