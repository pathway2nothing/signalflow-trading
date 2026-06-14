"""Brokers - turn orders into fills."""


import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from signalflow.decorators import broker
from signalflow.engine.types import Fill, Order
from signalflow.enums import Side


@runtime_checkable
class Broker(Protocol):
    def execute(self, orders: list[Order], bar) -> list[Fill]: ...


@broker("sim")
@dataclass
class SimBroker(Broker):
    """Simulated fills with flat fee + slippage; next-available close pricing."""

    fee_rate: float = 0.001
    slippage: float = 0.0005
    quote: str = "USDT"

    def execute(self, orders: list[Order], bar) -> list[Fill]:
        fills: list[Fill] = []
        for o in orders:
            price = bar.prices.get(o.pair)
            if price is None or o.qty <= 0:
                continue
            exec_price = price * (1 + self.slippage) if o.side == Side.BUY else price * (1 - self.slippage)
            fee = o.qty * exec_price * self.fee_rate
            fills.append(
                Fill(
                    pair=o.pair,
                    ts=bar.ts,
                    side=o.side,
                    qty=o.qty,
                    price=exec_price,
                    fee=fee,
                    fee_asset=self.quote,
                )
            )
        return fills


@broker("exchange")
@dataclass
class ExchangeBroker(Broker):
    """Base for live venue brokers. Subclass and implement ``execute``."""

    quote: str = "USDT"

    def execute(self, orders: list[Order], bar) -> list[Fill]:
        raise NotImplementedError(
            "ExchangeBroker is abstract; use SimBroker for backtest/paper or a "
            "configured live venue client (e.g. BinanceBroker) for armed live trading."
        )


@broker("binance")
@dataclass
class BinanceBroker(ExchangeBroker):
    """Live Binance spot venue: signed REST market orders (testnet by default).

    Set ``base_url`` to ``https://api.binance.com`` for production. Requires
    ``api_key``/``api_secret``; only reachable through ``Flow.live(armed=True)``.
    """

    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://testnet.binance.vision"
    recv_window: int = 5000
    timeout: float = 20.0
    quote: str = "USDT"

    def execute(self, orders: list[Order], bar) -> list[Fill]:
        if not (self.api_key and self.api_secret):
            raise ValueError("BinanceBroker requires api_key and api_secret for armed trading")
        fills: list[Fill] = []
        for o in orders:
            if o.qty <= 0:
                continue
            resp = self._place(o)
            fill = self._to_fill(o, resp, bar)
            if fill is not None:
                fills.append(fill)
        return fills

    def _place(self, order: Order) -> dict:
        params = {
            "symbol": order.pair,
            "side": order.side.name,
            "type": "MARKET",
            "quantity": repr(order.qty),
            "newOrderRespType": "FULL",
            "recvWindow": self.recv_window,
            "timestamp": int(time.time() * 1000),
        }
        query = urllib.parse.urlencode(params)
        signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        url = f"{self.base_url}/api/v3/order?{query}&signature={signature}"
        req = urllib.request.Request(url, method="POST", headers={"X-MBX-APIKEY": self.api_key})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _to_fill(self, order: Order, resp: dict, bar) -> "Fill | None":
        executed = float(resp.get("executedQty", 0.0) or 0.0)
        if executed <= 0:
            return None
        quote_spent = float(resp.get("cummulativeQuoteQty", 0.0) or 0.0)
        avg_price = quote_spent / executed if executed else 0.0
        legs = resp.get("fills") or []
        fee = sum(float(leg.get("commission", 0.0)) for leg in legs)
        fee_asset = legs[0].get("commissionAsset", self.quote) if legs else self.quote
        return Fill(
            pair=order.pair,
            ts=bar.ts,
            side=order.side,
            qty=executed,
            price=avg_price,
            fee=fee,
            fee_asset=fee_asset,
        )
