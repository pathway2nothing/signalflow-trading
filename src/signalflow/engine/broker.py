"""Brokers - turn orders into fills."""

import decimal
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl
from loguru import logger

from signalflow.decorators import broker
from signalflow.engine.types import Fill, Order
from signalflow.enums import OrderType, Side


@runtime_checkable
class Broker(Protocol):
    def execute(self, orders: list[Order], bar) -> list[Fill]: ...


def sim_client_order_id(order: Order) -> str:
    """Deterministic client-order id for non-exchange brokers (same recipe as Binance)."""
    raw = f"{order.pair}|{order.ts}|{order.side.name}|{order.qty}"
    return "sf-" + hashlib.sha256(raw.encode()).hexdigest()[:29]


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
            if o.type == OrderType.LIMIT:
                exec_price = self._limit_fill(o, bar, price)
                if exec_price is None:
                    continue
            else:
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

    def _limit_fill(self, order: Order, bar, close: float) -> "float | None":
        """Fill a resting limit at its price if the bar traded through it, else skip."""
        if order.limit_price is None:
            return None
        high, low = self._bar_high_low(bar, order.pair, close)
        if order.side == Side.BUY:
            return order.limit_price if low <= order.limit_price else None
        return order.limit_price if high >= order.limit_price else None

    @staticmethod
    def _bar_high_low(bar, pair: str, close: float) -> "tuple[float, float]":
        frame = getattr(bar, "frame", None)
        if frame is not None and {"high", "low"} <= set(frame.columns):
            row = frame.filter(pl.col("pair") == pair)
            if row.height:
                return float(row.get_column("high")[0]), float(row.get_column("low")[0])
        return close, close


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
    """Live Binance spot venue: signed REST orders, quantized to exchange filters (testnet by default).

    Set ``base_url`` to ``https://api.binance.com`` for production. Requires
    ``api_key``/``api_secret``; only reachable through ``Flow.live(armed=True)``.
    Quantities/prices are floored/rounded to the symbol's ``stepSize``/``tickSize``
    and orders below ``minQty``/``minNotional`` are skipped (never sent). Every order
    carries a deterministic ``newClientOrderId`` so a retried send never double-fills;
    LIMIT orders are IOC (bar-synchronous fill-or-gone), matching ``SimBroker``.
    """

    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://testnet.binance.vision"
    recv_window: int = 5000
    timeout: float = 20.0
    quote: str = "USDT"
    max_retries: int = 2
    retry_delay: float = 1.0

    def __post_init__(self) -> None:
        self._filter_cache: dict = {}

    def execute(self, orders: list[Order], bar) -> list[Fill]:
        if not (self.api_key and self.api_secret):
            raise ValueError("BinanceBroker requires api_key and api_secret for armed trading")
        fills: list[Fill] = []
        for o in orders:
            if o.qty <= 0:
                continue
            fill = self._execute_one(o, bar)
            if fill is not None:
                fills.append(fill)
        return fills

    def _execute_one(self, order: Order, bar) -> "Fill | None":
        filters = self._filters(order.pair)
        qty = self._floor_step(order.qty, filters.get("stepSize"))
        ref_price = order.limit_price if order.limit_price is not None else bar.prices.get(order.pair)
        price = self._round_tick(order.limit_price, filters.get("tickSize")) if order.limit_price is not None else None
        notional = float(qty) * float(ref_price or 0.0)
        min_qty = filters.get("minQty", 0.0)
        min_notional = filters.get("minNotional", 0.0)
        if float(qty) < min_qty or (min_notional and notional < min_notional):
            logger.error(
                f"BinanceBroker: skipping {order.side.name} {order.pair} qty={qty} notional={notional}: "
                f"below exchange minimums (minQty={min_qty}, minNotional={min_notional})"
            )
            return None
        resp = self._place(order, qty, price)
        if resp is None:
            return None
        return self._resp_to_fill(resp, order)

    def _filters(self, pair: str) -> dict:
        """Exchange LOT_SIZE/PRICE_FILTER/NOTIONAL limits for ``pair`` (fetched once, cached)."""
        if pair in self._filter_cache:
            return self._filter_cache[pair]
        try:
            req = urllib.request.Request(f"{self.base_url}/api/v3/exchangeInfo?symbol={pair}", method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                info = json.loads(resp.read().decode())
            filters = self._parse_filters(info)
        except Exception as e:
            logger.warning(f"BinanceBroker: could not fetch exchange filters for {pair}: {e}; sending unquantized")
            filters = {}
        self._filter_cache[pair] = filters
        return filters

    @staticmethod
    def _parse_filters(info: dict) -> dict:
        symbols = info.get("symbols") or []
        if not symbols:
            return {}
        by_type = {f.get("filterType"): f for f in symbols[0].get("filters", [])}
        lot = by_type.get("LOT_SIZE", {})
        price = by_type.get("PRICE_FILTER", {})
        notional = by_type.get("NOTIONAL") or by_type.get("MIN_NOTIONAL") or {}
        return {
            "stepSize": lot.get("stepSize"),
            "minQty": float(lot.get("minQty", 0.0) or 0.0),
            "tickSize": price.get("tickSize"),
            "minNotional": float(notional.get("minNotional", 0.0) or 0.0),
        }

    @staticmethod
    def _floor_step(value: float, step: "str | None") -> decimal.Decimal:
        d = decimal.Decimal(str(value))
        if not step:
            return d
        s = decimal.Decimal(str(step))
        return (d / s).to_integral_value(rounding=decimal.ROUND_DOWN) * s

    @staticmethod
    def _round_tick(value: float, step: "str | None") -> decimal.Decimal:
        d = decimal.Decimal(str(value))
        if not step:
            return d
        s = decimal.Decimal(str(step))
        return (d / s).to_integral_value(rounding=decimal.ROUND_HALF_UP) * s

    @staticmethod
    def client_order_id(order: Order) -> str:
        """Deterministic id so a retried send dedupes venue-side instead of double-filling."""
        return sim_client_order_id(order)

    def query_order(self, pair: str, client_order_id: str) -> dict:
        """Signed GET of an order's current venue state by its client order id."""
        params = {
            "symbol": pair,
            "origClientOrderId": client_order_id,
            "recvWindow": self.recv_window,
            "timestamp": int(time.time() * 1000),
        }
        query = urllib.parse.urlencode(params)
        signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        url = f"{self.base_url}/api/v3/order?{query}&signature={signature}"
        req = urllib.request.Request(url, method="GET", headers={"X-MBX-APIKEY": self.api_key})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _place(self, order: Order, qty: decimal.Decimal, price: "decimal.Decimal | None") -> "dict | None":
        params = {
            "symbol": order.pair,
            "side": order.side.name,
            "quantity": f"{qty:f}",
            "newClientOrderId": self.client_order_id(order),
            "newOrderRespType": "FULL",
            "recvWindow": self.recv_window,
            "timestamp": int(time.time() * 1000),
        }
        if order.type == OrderType.LIMIT and price is not None:
            params["type"] = "LIMIT"
            params["timeInForce"] = "IOC"
            params["price"] = f"{price:f}"
        else:
            params["type"] = "MARKET"
        return self._signed_post(params)

    def _signed_post(self, params: dict) -> "dict | None":
        query = urllib.parse.urlencode(params)
        signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        url = f"{self.base_url}/api/v3/order?{query}&signature={signature}"
        req = urllib.request.Request(url, method="POST", headers={"X-MBX-APIKEY": self.api_key})
        attempt = 0
        while True:
            attempt += 1
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace") if hasattr(e, "read") else str(e)
                logger.error(f"BinanceBroker: order for {params['symbol']} rejected (HTTP {e.code}): {body}")
                return None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt > self.max_retries:
                    logger.error(f"BinanceBroker: order for {params['symbol']} failed after {attempt} attempts: {e}")
                    return None
                logger.warning(
                    f"BinanceBroker: transient error for {params['symbol']} (attempt {attempt}): {e}; retrying"
                )
                time.sleep(self.retry_delay)

    def _resp_to_fill(self, resp: dict, order_like) -> "Fill | None":
        """Build a Fill from a venue order response (reused by execute and reconciliation)."""
        executed = float(resp.get("executedQty", 0.0) or 0.0)
        if executed <= 0:
            return None
        quote_spent = float(resp.get("cummulativeQuoteQty", 0.0) or 0.0)
        avg_price = quote_spent / executed if executed else 0.0
        legs = resp.get("fills") or []
        fee = sum(float(leg.get("commission", 0.0)) for leg in legs)
        fee_asset = legs[0].get("commissionAsset", self.quote) if legs else self.quote
        return Fill(
            pair=order_like.pair,
            ts=order_like.ts,
            side=order_like.side,
            qty=executed,
            price=avg_price,
            fee=fee,
            fee_asset=fee_asset,
        )
