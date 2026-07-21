"""BinanceBroker: filter quantization, min-notional skip, idempotence, retry, IOC."""

import json
import urllib.error
import urllib.parse
import urllib.request
from types import SimpleNamespace

from signalflow.engine.broker import BinanceBroker
from signalflow.engine.types import Order
from signalflow.enums import OrderType, Side

_EXCHANGE_INFO = {
    "symbols": [
        {
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "NOTIONAL", "minNotional": "10.0"},
            ]
        }
    ]
}

_ORDER_RESP = {
    "executedQty": "0.117",
    "cummulativeQuoteQty": "11.7",
    "fills": [{"commission": "0.0001", "commissionAsset": "BTC"}],
}


class _Resp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _bar():
    return SimpleNamespace(ts="2024-01-01T00:00:00", prices={"BTCUSDT": 100.0})


def _broker():
    return BinanceBroker(api_key="k", api_secret="s", retry_delay=0.0)


def _install(monkeypatch, order_handler):
    calls = {"order": []}

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        if "exchangeInfo" in url:
            return _Resp(_EXCHANGE_INFO)
        calls["order"].append(url)
        return order_handler(url, len(calls["order"]))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def _query(url):
    return dict(urllib.parse.parse_qsl(url.split("?", 1)[1]))


def test_quantizes_quantity(monkeypatch):
    calls = _install(monkeypatch, lambda url, n: _Resp(_ORDER_RESP))
    order = Order("BTCUSDT", Side.BUY, 0.11764705882352941, ts="2024-01-01T00:00:00")
    fills = _broker().execute([order], _bar())
    assert len(fills) == 1
    assert _query(calls["order"][0])["quantity"] == "0.117"


def test_below_min_notional_skipped(monkeypatch):
    calls = _install(monkeypatch, lambda url, n: _Resp(_ORDER_RESP))
    order = Order("BTCUSDT", Side.BUY, 0.05, ts="2024-01-01T00:00:00")
    fills = _broker().execute([order], _bar())
    assert fills == []
    assert calls["order"] == []


def test_idempotent_client_order_id():
    b = _broker()
    o1 = Order("BTCUSDT", Side.BUY, 0.117, ts="2024-01-01T00:00:00")
    o2 = Order("BTCUSDT", Side.BUY, 0.117, ts="2024-01-01T00:00:00")
    assert b.client_order_id(o1) == b.client_order_id(o2)
    assert b.client_order_id(o1).startswith("sf-")
    assert len(b.client_order_id(o1)) <= 36


def test_retry_then_success(monkeypatch):
    def handler(url, n):
        if n == 1:
            raise urllib.error.URLError("connection reset")
        return _Resp(_ORDER_RESP)

    calls = _install(monkeypatch, handler)
    order = Order("BTCUSDT", Side.BUY, 0.117, ts="2024-01-01T00:00:00")
    fills = _broker().execute([order], _bar())
    assert len(fills) == 1
    assert len(calls["order"]) == 2


def test_limit_order_is_ioc(monkeypatch):
    calls = _install(monkeypatch, lambda url, n: _Resp(_ORDER_RESP))
    order = Order("BTCUSDT", Side.BUY, 0.117, type=OrderType.LIMIT, limit_price=100.0, ts="2024-01-01T00:00:00")
    _broker().execute([order], _bar())
    q = _query(calls["order"][0])
    assert q["timeInForce"] == "IOC"
    assert q["type"] == "LIMIT"
