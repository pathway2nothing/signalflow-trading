"""Brokers - turn orders into fills."""

import decimal
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import polars as pl
from loguru import logger

from signalflow.decorators import broker
from signalflow.engine.quantize import SymbolFilters, quantize_order
from signalflow.engine.types import Fill, Order, client_order_id
from signalflow.enums import OrderType, Side


@runtime_checkable
class Broker(Protocol):
    def execute(self, orders: list[Order], bar) -> list[Fill]: ...


@broker("sim")
@dataclass
class SimBroker(Broker):
    """Simulated fills with flat fee + slippage.

    ``fill="close"`` (default) executes an order at the close of the bar that
    produced it; ``fill="next_open"`` defers it to the next bar's open, which is
    what a live loop deciding on a closed candle actually gets. ``filters`` maps
    pair -> ``{stepSize, tickSize, minQty, minNotional}`` and applies the same
    quantization as :class:`BinanceBroker`, so paper and armed runs size orders
    identically; orders below the minimums are skipped.
    """

    fee_rate: float = 0.001
    slippage: float = 0.0005
    quote: str = "USDT"
    fill: str = "close"
    filters: "dict | None" = None

    def __post_init__(self) -> None:
        if self.fill not in ("close", "next_open"):
            raise ValueError(f"SimBroker.fill must be 'close' or 'next_open', got {self.fill!r}")
        self._filters = {k: SymbolFilters.from_mapping(v) for k, v in (self.filters or {}).items()}

    def execute(
        self, orders: list[Order], bar, at: str = "close", prices: "dict[str, float] | None" = None
    ) -> list[Fill]:
        """Fill ``orders`` against ``bar``: at its close (``at="close"``), at its open (``at="open"``),
        or at explicit ``prices`` (pair -> price, e.g. a live ticker) with the same slippage and fees."""
        if prices is None:
            prices = bar.prices if at == "close" or not getattr(bar, "open", None) else bar.open
        fills: list[Fill] = []
        for o in orders:
            price = prices.get(o.pair)
            if price is None or o.qty <= 0:
                continue
            qty, limit_price = o.qty, o.limit_price
            spec = self._filters.get(o.pair)
            if spec is not None:
                qty, limit_price, ok = quantize_order(o.qty, price, o.limit_price, spec)
                if not ok:
                    logger.debug(f"SimBroker: skipping {o.side.name} {o.pair} qty={o.qty}: below symbol minimums")
                    continue
            if o.type == OrderType.LIMIT:
                exec_price = self._limit_fill(o, bar, price, limit_price)
                if exec_price is None:
                    continue
            else:
                exec_price = price * (1 + self.slippage) if o.side == Side.BUY else price * (1 - self.slippage)
            fee = qty * exec_price * self.fee_rate
            fills.append(
                Fill(pair=o.pair, ts=bar.ts, side=o.side, qty=qty, price=exec_price, fee=fee, fee_asset=self.quote)
            )
        return fills

    def _limit_fill(self, order: Order, bar, close: float, limit_price: "float | None") -> "float | None":
        """Fill a resting limit at its price if the bar traded through it, else skip."""
        if limit_price is None:
            return None
        high, low = self._bar_high_low(bar, order.pair, close)
        if order.side == Side.BUY:
            return limit_price if low <= limit_price else None
        return limit_price if high >= limit_price else None

    @staticmethod
    def _bar_high_low(bar, pair: str, close: float) -> "tuple[float, float]":
        highs, lows = getattr(bar, "high", None), getattr(bar, "low", None)
        if highs and lows and pair in highs and pair in lows:
            return float(highs[pair]), float(lows[pair])
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

    api_key: str = field(default="", repr=False)
    api_secret: str = field(default="", repr=False)
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
        spec = self._filters(order.pair)
        ref_price = order.limit_price if order.limit_price is not None else bar.prices.get(order.pair)
        qty, price, ok = quantize_order(order.qty, float(ref_price or 0.0), order.limit_price, spec)
        if not ok:
            logger.error(
                f"BinanceBroker: skipping {order.side.name} {order.pair} qty={qty}: below exchange minimums "
                f"(minQty={spec.min_qty}, minNotional={spec.min_notional})"
            )
            return None
        resp = self._place(order, qty, price)
        if resp is None:
            # The send failed or timed out after the venue may have accepted it: ask before assuming nothing filled.
            resp = self._query_after_failure(order)
            if resp is None:
                return None
        return self._resp_to_fill(resp, order)

    def _query_after_failure(self, order: Order) -> "dict | None":
        cid = client_order_id(order)
        try:
            resp = self.query_order(order.pair, cid)
        except Exception as exc:
            logger.error(f"BinanceBroker: could not query {cid} after a failed send: {exc}; treating as not filled")
            return None
        if float(resp.get("executedQty", 0.0) or 0.0) > 0:
            logger.warning(f"BinanceBroker: order {cid} was executed by the venue despite the failed send")
            return resp
        return None

    def _filters(self, pair: str) -> SymbolFilters:
        """Exchange LOT_SIZE/PRICE_FILTER/NOTIONAL limits for ``pair`` (fetched once, cached)."""
        if pair in self._filter_cache:
            return self._filter_cache[pair]
        try:
            req = urllib.request.Request(f"{self.base_url}/api/v3/exchangeInfo?symbol={pair}", method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                info = json.loads(resp.read().decode())
            spec = SymbolFilters.from_binance(info)
        except Exception as e:
            logger.warning(f"BinanceBroker: could not fetch exchange filters for {pair}: {e}; sending unquantized")
            spec = SymbolFilters()
        self._filter_cache[pair] = spec
        return spec

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

    def _place(self, order: Order, qty: float, price: "float | None") -> "dict | None":
        params = {
            "symbol": order.pair,
            "side": order.side.name,
            "quantity": f"{decimal.Decimal(str(qty)):f}",
            "newClientOrderId": client_order_id(order),
            "newOrderRespType": "FULL",
            "recvWindow": self.recv_window,
        }
        if order.type == OrderType.LIMIT and price is not None:
            params["type"] = "LIMIT"
            params["timeInForce"] = "IOC"
            params["price"] = f"{decimal.Decimal(str(price)):f}"
        else:
            params["type"] = "MARKET"
        return self._signed_post(params)

    def _signed_post(self, params: dict) -> "dict | None":
        """POST a signed order; every retry re-signs with a fresh ``timestamp`` so it stays inside ``recvWindow``."""
        attempt = 0
        while True:
            attempt += 1
            query = urllib.parse.urlencode({**params, "timestamp": int(time.time() * 1000)})
            signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            url = f"{self.base_url}/api/v3/order?{query}&signature={signature}"
            req = urllib.request.Request(url, method="POST", headers={"X-MBX-APIKEY": self.api_key})
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
