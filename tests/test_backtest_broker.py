"""Tests for signalflow.strategy.broker.backtest.BacktestBroker."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from signalflow.core.containers.order import Order, OrderFill
from signalflow.core.containers.position import Position
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import PositionType


def _make_broker():
    from signalflow.strategy.broker.backtest import BacktestBroker

    return BacktestBroker(executor=MagicMock(), store=MagicMock())


def _make_state(cash=10000.0):
    state = StrategyState(strategy_id="test")
    state.portfolio.cash = cash
    return state


TS = datetime(2024, 1, 1)


# ── create_position ─────────────────────────────────────────────────────────


class TestCreatePosition:
    def test_buy_creates_long(self):
        broker = _make_broker()
        order = Order(id="o1", pair="BTCUSDT", side="BUY", qty=1.0, meta={"signal": "test"})
        fill = OrderFill(id="f1", order_id="o1", pair="BTCUSDT", side="BUY", ts=TS, price=100.0, qty=1.0, fee=0.1)
        pos = broker.create_position(order, fill)
        assert pos.position_type == PositionType.LONG
        assert pos.pair == "BTCUSDT"
        assert pos.entry_price == 100.0
        assert pos.qty == 1.0
        assert pos.fees_paid == 0.1

    def test_sell_creates_short(self):
        broker = _make_broker()
        order = Order(id="o1", pair="ETHUSDT", side="SELL", qty=2.0)
        fill = OrderFill(id="f1", order_id="o1", pair="ETHUSDT", side="SELL", ts=TS, price=50.0, qty=2.0, fee=0.05)
        pos = broker.create_position(order, fill)
        assert pos.position_type == PositionType.SHORT
        assert pos.qty == 2.0

    def test_meta_includes_order_meta(self):
        broker = _make_broker()
        order = Order(id="o1", pair="BTCUSDT", side="BUY", qty=1.0, meta={"custom": "value"})
        fill = OrderFill(id="f1", order_id="o1", pair="BTCUSDT", side="BUY", ts=TS, price=100.0, qty=1.0, fee=0.0)
        pos = broker.create_position(order, fill)
        assert pos.meta["custom"] == "value"
        assert pos.meta["order_id"] == "o1"
        assert pos.meta["fill_id"] == "f1"


# ── process_fills — entry flow ──────────────────────────────────────────────


class TestProcessFillsEntry:
    def test_new_position_created(self):
        broker = _make_broker()
        state = _make_state()
        order = Order(id="o1", pair="BTCUSDT", side="BUY", qty=1.0)
        fill = OrderFill(id="f1", order_id="o1", pair="BTCUSDT", side="BUY", ts=TS, price=100.0, qty=1.0, fee=0.1)
        trades = broker.process_fills([fill], [order], state)
        assert len(trades) == 1
        assert trades[0].meta["type"] == "entry"
        assert len(state.portfolio.positions) == 1

    def test_cash_decreased_on_buy(self):
        broker = _make_broker()
        state = _make_state(cash=10000.0)
        order = Order(id="o1", pair="BTCUSDT", side="BUY", qty=1.0)
        fill = OrderFill(id="f1", order_id="o1", pair="BTCUSDT", side="BUY", ts=TS, price=100.0, qty=1.0, fee=0.1)
        broker.process_fills([fill], [order], state)
        # cash -= notional + fee = 100 + 0.1 = 100.1
        assert state.portfolio.cash == pytest.approx(10000.0 - 100.1)

    def test_cash_increased_on_sell_entry(self):
        broker = _make_broker()
        state = _make_state(cash=10000.0)
        order = Order(id="o1", pair="BTCUSDT", side="SELL", qty=1.0)
        fill = OrderFill(id="f1", order_id="o1", pair="BTCUSDT", side="SELL", ts=TS, price=100.0, qty=1.0, fee=0.1)
        broker.process_fills([fill], [order], state)
        # cash += notional - fee = 100 - 0.1 = 99.9
        assert state.portfolio.cash == pytest.approx(10000.0 + 99.9)


# ── process_fills — exit flow ───────────────────────────────────────────────


class TestProcessFillsExit:
    def _state_with_position(self):
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
        return state, pos

    def test_existing_position_updated(self):
        broker = _make_broker()
        state, pos = self._state_with_position()
        order = Order(id="o1", pair="BTCUSDT", side="SELL", qty=1.0, position_id="pos1")
        fill = OrderFill(
            id="f1", order_id="o1", pair="BTCUSDT", side="SELL", ts=TS, price=110.0, qty=1.0, fee=0.1, position_id="pos1"
        )
        trades = broker.process_fills([fill], [order], state)
        assert len(trades) == 1
        assert trades[0].meta["type"] == "exit"
        assert pos.is_closed

    def test_cash_increased_on_sell_exit(self):
        broker = _make_broker()
        state, _ = self._state_with_position()
        initial_cash = state.portfolio.cash
        order = Order(id="o1", pair="BTCUSDT", side="SELL", qty=1.0, position_id="pos1")
        fill = OrderFill(
            id="f1", order_id="o1", pair="BTCUSDT", side="SELL", ts=TS, price=110.0, qty=1.0, fee=0.1, position_id="pos1"
        )
        broker.process_fills([fill], [order], state)
        # cash += notional - fee = 110 - 0.1
        assert state.portfolio.cash == pytest.approx(initial_cash + 109.9)

    def test_unknown_order_skipped(self):
        broker = _make_broker()
        state = _make_state()
        fill = OrderFill(id="f1", order_id="missing", pair="BTCUSDT", side="BUY", ts=TS, price=100.0, qty=1.0, fee=0.0)
        trades = broker.process_fills([fill], [], state)
        assert trades == []


# ── mark_positions ──────────────────────────────────────────────────────────


class TestMarkPositions:
    def test_marks_all_open(self):
        broker = _make_broker()
        state = _make_state()
        pos = Position(
            id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=100.0, last_price=100.0, qty=1.0
        )
        state.portfolio.positions[pos.id] = pos
        broker.mark_positions(state, {"BTCUSDT": 110.0}, TS)
        assert pos.last_price == 110.0
        assert pos.last_time == TS

    def test_skips_missing_price(self):
        broker = _make_broker()
        state = _make_state()
        pos = Position(
            id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=100.0, last_price=100.0, qty=1.0
        )
        state.portfolio.positions[pos.id] = pos
        broker.mark_positions(state, {}, TS)
        assert pos.last_price == 100.0  # unchanged

    def test_skips_zero_price(self):
        broker = _make_broker()
        state = _make_state()
        pos = Position(
            id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=100.0, last_price=100.0, qty=1.0
        )
        state.portfolio.positions[pos.id] = pos
        broker.mark_positions(state, {"BTCUSDT": 0}, TS)
        assert pos.last_price == 100.0


# ── Position helpers ────────────────────────────────────────────────────────


class TestPositionHelpers:
    def test_get_open_position_for_pair(self):
        broker = _make_broker()
        state = _make_state()
        pos = Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=100.0, qty=1.0)
        state.portfolio.positions[pos.id] = pos
        assert broker.get_open_position_for_pair(state, "BTCUSDT") is pos

    def test_get_open_position_for_pair_none(self):
        broker = _make_broker()
        assert broker.get_open_position_for_pair(_make_state(), "BTCUSDT") is None

    def test_get_open_positions_by_pair(self):
        broker = _make_broker()
        state = _make_state()
        pos = Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=100.0, qty=1.0)
        state.portfolio.positions[pos.id] = pos
        result = broker.get_open_positions_by_pair(state)
        assert "BTCUSDT" in result
        assert result["BTCUSDT"] is pos
