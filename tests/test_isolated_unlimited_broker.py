"""Tests for IsolatedBacktestBroker and UnlimitedBacktestBroker."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from signalflow.core.containers.order import Order, OrderFill
from signalflow.core.containers.position import Position
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import PositionType
from signalflow.strategy.broker.isolated_broker import IsolatedBacktestBroker
from signalflow.strategy.broker.unlimited_broker import UnlimitedBacktestBroker


TS = datetime(2024, 1, 1)


def _make_isolated_broker(pair="BTCUSDT"):
    return IsolatedBacktestBroker(pair=pair, executor=MagicMock(), store=MagicMock())


def _make_unlimited_broker():
    return UnlimitedBacktestBroker(executor=MagicMock(), store=MagicMock())


def _make_state(cash=10000.0):
    s = StrategyState(strategy_id="test")
    s.portfolio.cash = cash
    return s


# ── IsolatedBacktestBroker ───────────────────────────────────────────────


class TestIsolatedBrokerCreatePosition:
    def test_buy_creates_long(self):
        broker = _make_isolated_broker()
        order = Order(id="o1", pair="BTCUSDT", side="BUY", qty=1.0, meta={"sig": "test"})
        fill = OrderFill(id="f1", order_id="o1", pair="BTCUSDT", side="BUY", ts=TS, price=100.0, qty=1.0, fee=0.1)
        pos = broker.create_position(order, fill)
        assert pos.position_type == PositionType.LONG
        assert pos.entry_price == 100.0
        assert pos.qty == 1.0
        assert pos.fees_paid == 0.1
        assert pos.meta["sig"] == "test"

    def test_sell_creates_short(self):
        broker = _make_isolated_broker()
        order = Order(id="o1", pair="BTCUSDT", side="SELL", qty=2.0)
        fill = OrderFill(id="f1", order_id="o1", pair="BTCUSDT", side="SELL", ts=TS, price=50.0, qty=2.0, fee=0.05)
        pos = broker.create_position(order, fill)
        assert pos.position_type == PositionType.SHORT
        assert pos.qty == 2.0


class TestIsolatedBrokerProcessFills:
    def test_entry_creates_position_and_deducts_cash(self):
        broker = _make_isolated_broker()
        state = _make_state(cash=10000.0)
        order = Order(id="o1", pair="BTCUSDT", side="BUY", qty=1.0)
        fill = OrderFill(id="f1", order_id="o1", pair="BTCUSDT", side="BUY", ts=TS, price=100.0, qty=1.0, fee=0.1)
        trades = broker.process_fills([fill], [order], state)
        assert len(trades) == 1
        assert trades[0].meta["type"] == "entry"
        assert state.portfolio.cash == pytest.approx(10000.0 - 100.1)

    def test_sell_entry_adds_cash(self):
        broker = _make_isolated_broker()
        state = _make_state(cash=10000.0)
        order = Order(id="o1", pair="BTCUSDT", side="SELL", qty=1.0)
        fill = OrderFill(id="f1", order_id="o1", pair="BTCUSDT", side="SELL", ts=TS, price=100.0, qty=1.0, fee=0.1)
        broker.process_fills([fill], [order], state)
        assert state.portfolio.cash == pytest.approx(10000.0 + 99.9)

    def test_exit_sell_adds_cash(self):
        broker = _make_isolated_broker()
        state = _make_state(cash=5000.0)
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=100.0,
            last_price=100.0,
            qty=1.0,
            entry_time=TS,
            last_time=TS,
        )
        state.portfolio.positions[pos.id] = pos
        order = Order(id="o1", pair="BTCUSDT", side="SELL", qty=1.0, position_id="pos1")
        fill = OrderFill(
            id="f1",
            order_id="o1",
            pair="BTCUSDT",
            side="SELL",
            ts=TS,
            price=110.0,
            qty=1.0,
            fee=0.1,
            position_id="pos1",
        )
        trades = broker.process_fills([fill], [order], state)
        assert len(trades) == 1
        assert trades[0].meta["type"] == "exit"
        assert state.portfolio.cash == pytest.approx(5000.0 + 109.9)

    def test_exit_buy_deducts_cash(self):
        broker = _make_isolated_broker()
        state = _make_state(cash=15000.0)
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.SHORT,
            entry_price=100.0,
            last_price=100.0,
            qty=1.0,
            entry_time=TS,
            last_time=TS,
        )
        state.portfolio.positions[pos.id] = pos
        order = Order(id="o1", pair="BTCUSDT", side="BUY", qty=1.0, position_id="pos1")
        fill = OrderFill(
            id="f1", order_id="o1", pair="BTCUSDT", side="BUY", ts=TS, price=90.0, qty=1.0, fee=0.1, position_id="pos1"
        )
        trades = broker.process_fills([fill], [order], state)
        assert state.portfolio.cash == pytest.approx(15000.0 - 90.1)

    def test_wrong_pair_raises(self):
        broker = _make_isolated_broker(pair="BTCUSDT")
        state = _make_state()
        order = Order(id="o1", pair="ETHUSDT", side="BUY", qty=1.0)
        fill = OrderFill(id="f1", order_id="o1", pair="ETHUSDT", side="BUY", ts=TS, price=50.0, qty=1.0, fee=0.0)
        with pytest.raises(ValueError, match="Invalid pair"):
            broker.process_fills([fill], [order], state)

    def test_unknown_order_skipped(self):
        broker = _make_isolated_broker()
        state = _make_state()
        fill = OrderFill(id="f1", order_id="missing", pair="BTCUSDT", side="BUY", ts=TS, price=100.0, qty=1.0, fee=0.0)
        trades = broker.process_fills([fill], [], state)
        assert trades == []


class TestIsolatedBrokerHelpers:
    def test_mark_positions(self):
        broker = _make_isolated_broker()
        state = _make_state()
        pos = Position(
            id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=100.0, last_price=100.0, qty=1.0
        )
        state.portfolio.positions[pos.id] = pos
        broker.mark_positions(state, {"BTCUSDT": 120.0}, TS)
        assert pos.last_price == 120.0

    def test_get_open_position(self):
        broker = _make_isolated_broker(pair="BTCUSDT")
        state = _make_state()
        pos = Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=100.0, qty=1.0)
        state.portfolio.positions[pos.id] = pos
        assert broker.get_open_position(state) is pos

    def test_get_open_position_none(self):
        broker = _make_isolated_broker(pair="BTCUSDT")
        state = _make_state()
        assert broker.get_open_position(state) is None


# ── UnlimitedBacktestBroker ──────────────────────────────────────────────


class TestUnlimitedBrokerCreatePosition:
    def test_buy_creates_long(self):
        broker = _make_unlimited_broker()
        order = Order(id="o1", pair="BTCUSDT", side="BUY", qty=1.0)
        fill = OrderFill(id="f1", order_id="o1", pair="BTCUSDT", side="BUY", ts=TS, price=100.0, qty=1.0, fee=0.1)
        pos = broker.create_position(order, fill)
        assert pos.position_type == PositionType.LONG

    def test_sell_creates_short(self):
        broker = _make_unlimited_broker()
        order = Order(id="o1", pair="BTCUSDT", side="SELL", qty=1.0)
        fill = OrderFill(id="f1", order_id="o1", pair="BTCUSDT", side="SELL", ts=TS, price=100.0, qty=1.0, fee=0.1)
        pos = broker.create_position(order, fill)
        assert pos.position_type == PositionType.SHORT


class TestUnlimitedBrokerProcessFills:
    def test_entry_no_cash_change(self):
        broker = _make_unlimited_broker()
        state = _make_state(cash=10000.0)
        order = Order(id="o1", pair="BTCUSDT", side="BUY", qty=1.0)
        fill = OrderFill(id="f1", order_id="o1", pair="BTCUSDT", side="BUY", ts=TS, price=100.0, qty=1.0, fee=0.1)
        trades = broker.process_fills([fill], [order], state)
        assert len(trades) == 1
        assert trades[0].meta["type"] == "entry"
        assert state.portfolio.cash == 10000.0  # unchanged

    def test_exit_no_cash_change(self):
        broker = _make_unlimited_broker()
        state = _make_state(cash=10000.0)
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=100.0,
            last_price=100.0,
            qty=1.0,
            entry_time=TS,
            last_time=TS,
        )
        state.portfolio.positions[pos.id] = pos
        order = Order(id="o1", pair="BTCUSDT", side="SELL", qty=1.0, position_id="pos1")
        fill = OrderFill(
            id="f1",
            order_id="o1",
            pair="BTCUSDT",
            side="SELL",
            ts=TS,
            price=110.0,
            qty=1.0,
            fee=0.1,
            position_id="pos1",
        )
        trades = broker.process_fills([fill], [order], state)
        assert len(trades) == 1
        assert trades[0].meta["type"] == "exit"
        assert state.portfolio.cash == 10000.0  # unchanged

    def test_unknown_order_skipped(self):
        broker = _make_unlimited_broker()
        state = _make_state()
        fill = OrderFill(id="f1", order_id="missing", pair="BTCUSDT", side="BUY", ts=TS, price=100.0, qty=1.0, fee=0.0)
        trades = broker.process_fills([fill], [], state)
        assert trades == []


class TestUnlimitedBrokerHelpers:
    def test_mark_positions(self):
        broker = _make_unlimited_broker()
        state = _make_state()
        pos = Position(
            id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=100.0, last_price=100.0, qty=1.0
        )
        state.portfolio.positions[pos.id] = pos
        broker.mark_positions(state, {"BTCUSDT": 120.0}, TS)
        assert pos.last_price == 120.0

    def test_get_open_position_for_pair(self):
        broker = _make_unlimited_broker()
        state = _make_state()
        pos = Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=100.0, qty=1.0)
        state.portfolio.positions[pos.id] = pos
        assert broker.get_open_position_for_pair(state, "BTCUSDT") is pos
        assert broker.get_open_position_for_pair(state, "ETHUSDT") is None

    def test_get_open_positions_by_pair(self):
        broker = _make_unlimited_broker()
        state = _make_state()
        pos1 = Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=100.0, qty=1.0)
        pos2 = Position(id="p2", pair="ETHUSDT", position_type=PositionType.SHORT, entry_price=50.0, qty=2.0)
        state.portfolio.positions[pos1.id] = pos1
        state.portfolio.positions[pos2.id] = pos2
        result = broker.get_open_positions_by_pair(state)
        assert result["BTCUSDT"] is pos1
        assert result["ETHUSDT"] is pos2
