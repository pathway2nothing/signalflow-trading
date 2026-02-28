"""Tests for signalflow.core.containers.order and trade."""

from datetime import datetime

import pytest

from signalflow.core.containers.order import Order, OrderFill
from signalflow.core.containers.trade import Trade


class TestOrder:
    def test_defaults(self):
        o = Order()
        assert o.side == "BUY"
        assert o.order_type == "MARKET"
        assert o.status == "NEW"
        assert o.qty == 0.0
        assert o.id  # uuid generated

    def test_is_buy(self):
        o = Order(side="BUY")
        assert o.is_buy is True
        assert o.is_sell is False

    def test_is_sell(self):
        o = Order(side="SELL")
        assert o.is_sell is True
        assert o.is_buy is False

    def test_is_market(self):
        o = Order(order_type="MARKET")
        assert o.is_market is True

    def test_is_limit(self):
        o = Order(order_type="LIMIT", price=100.0)
        assert o.is_market is False

    def test_mutable_status(self):
        o = Order()
        o.status = "FILLED"
        assert o.status == "FILLED"


class TestOrderFill:
    def test_notional(self):
        f = OrderFill(price=45000.0, qty=0.5)
        assert f.notional == pytest.approx(22500.0)

    def test_frozen(self):
        f = OrderFill(price=100.0, qty=1.0)
        with pytest.raises(AttributeError):
            f.price = 200.0  # type: ignore


class TestTrade:
    def test_defaults(self):
        t = Trade()
        assert t.side == "BUY"
        assert t.price == 0.0
        assert t.qty == 0.0
        assert t.fee == 0.0
        assert t.id  # uuid generated

    def test_notional(self):
        t = Trade(price=45000.0, qty=0.5)
        assert t.notional == pytest.approx(22500.0)

    def test_frozen(self):
        t = Trade(price=100.0, qty=1.0)
        with pytest.raises(AttributeError):
            t.price = 200.0  # type: ignore

    def test_custom_fields(self):
        ts = datetime(2024, 6, 1)
        t = Trade(
            pair="BTCUSDT",
            side="SELL",
            ts=ts,
            price=50000.0,
            qty=0.1,
            fee=5.0,
            meta={"reason": "tp"},
        )
        assert t.pair == "BTCUSDT"
        assert t.side == "SELL"
        assert t.ts == ts
        assert t.fee == 5.0
        assert t.meta["reason"] == "tp"
