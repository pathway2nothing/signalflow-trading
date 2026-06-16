"""Limit-order execution (SimBroker bar-cross fill) + Run.target surfacing."""

from datetime import datetime

import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.engine.broker import SimBroker
from signalflow.engine.types import Intent, Order
from signalflow.enums import IntentKind, OrderType, Side
from signalflow.flow.loop import _orders
from signalflow.flow.run import Run


def _bar(high: float, low: float, close: float, pair: str = "BTCUSDT"):
    frame = pl.DataFrame(
        {
            "pair": [pair],
            "ts": [datetime(2024, 1, 1)],
            "open": [close],
            "high": [high],
            "low": [low],
            "close": [close],
            "volume": [0.0],
        }
    ).with_columns(pl.col("ts").cast(pl.Datetime("ms")))
    return next(Dataset(frame=frame).iter_bars())


def test_limit_buy_fills_at_limit_when_bar_trades_through():
    bar = _bar(high=105, low=95, close=100)
    order = Order("BTCUSDT", Side.BUY, qty=1.0, type=OrderType.LIMIT, limit_price=96.0)
    fills = SimBroker().execute([order], bar)
    assert len(fills) == 1
    assert fills[0].price == 96.0


def test_limit_buy_no_fill_when_price_never_reaches():
    bar = _bar(high=105, low=95, close=100)
    order = Order("BTCUSDT", Side.BUY, qty=1.0, type=OrderType.LIMIT, limit_price=94.0)
    assert SimBroker().execute([order], bar) == []


def test_limit_sell_fills_when_bar_trades_up():
    bar = _bar(high=105, low=95, close=100)
    order = Order("BTCUSDT", Side.SELL, qty=1.0, type=OrderType.LIMIT, limit_price=104.0)
    fills = SimBroker().execute([order], bar)
    assert len(fills) == 1 and fills[0].price == 104.0


def test_market_order_unaffected():
    bar = _bar(high=105, low=95, close=100)
    order = Order("BTCUSDT", Side.BUY, qty=1.0)
    fills = SimBroker(slippage=0.0).execute([order], bar)
    assert len(fills) == 1 and fills[0].price == 100.0


def test_intent_limit_price_produces_limit_order():
    intents = [Intent("BTCUSDT", IntentKind.OPEN, Side.BUY, notional=1000.0, limit_price=50.0)]
    orders = _orders(intents, {"BTCUSDT": 100.0}, datetime(2024, 1, 1))
    assert len(orders) == 1
    o = orders[0]
    assert o.type == OrderType.LIMIT and o.limit_price == 50.0
    assert o.qty == 20.0  # notional / limit_price


def test_run_target_in_scorecard():
    curve = pl.DataFrame({"ts": [datetime(2024, 1, 1)], "equity": [100.0]})
    run = Run("n", "backtest", curve, [], "ETH")
    assert run.scorecard()["target"] == "ETH"
