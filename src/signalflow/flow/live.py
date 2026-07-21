"""Real-time decision loop - the live/paper sibling of the backtest replay.

Backtest precomputes signals over a finished Dataset; live cannot see the future,
so it warms a rolling buffer from history (no trading) and then decides on each
freshly closed bar as it arrives. The decision core (decide -> risk -> broker ->
engine) is identical to the backtest.
"""

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC
from typing import Protocol, runtime_checkable

import polars as pl
from loguru import logger

from signalflow.data.dataset import Bar, Dataset
from signalflow.engine.clock import Clock
from signalflow.engine.engine import Engine
from signalflow.engine.types import Fill, OrderEvent, Position
from signalflow.enums import RunMode, Side
from signalflow.flow.loop import _EMPTY_SIGNALS_SCHEMA, _orders, enriched_signals
from signalflow.strategy.observation import Observation


def _order_id(broker, order) -> str:
    """Client-order id for an order: the broker's own recipe for a Binance venue, else the sim recipe."""
    from signalflow.engine.broker import BinanceBroker, sim_client_order_id

    return broker.client_order_id(order) if isinstance(broker, BinanceBroker) else sim_client_order_id(order)


def _order_status(order, fills) -> str:
    """Fill status of ``order`` given this bar's fills (filled/partial/skipped)."""
    for f in fills:
        if f.pair == order.pair and f.side == order.side:
            if abs(f.qty - order.qty) < 1e-9:
                return "filled"
            if 0 < f.qty < order.qty:
                return "partial"
    return "skipped"


def _reconcile_on_resume(engine, broker) -> None:
    """Query the venue for any 'placed' order lacking a 'result' and apply the missing fill."""
    resolved = {e.client_order_id for e in engine.order_log if e.kind == "result"}
    dangling = [e for e in engine.order_log if e.kind == "placed" and e.client_order_id not in resolved]
    for ev in dangling:
        try:
            resp = broker.query_order(ev.pair, ev.client_order_id)
        except Exception as exc:
            logger.error(f"live: could not reconcile order {ev.client_order_id} on resume: {exc}")
            continue
        fill = broker._resp_to_fill(resp, ev)
        if fill is not None:
            engine.apply([fill])
            engine.record_order(OrderEvent(ev.client_order_id, ev.pair, ev.side, ev.qty, ev.ts, "result", "filled"))
            logger.warning(f"live: reconciled dangling order {ev.client_order_id}; synthesized fill qty={fill.qty}")
        else:
            engine.record_order(OrderEvent(ev.client_order_id, ev.pair, ev.side, ev.qty, ev.ts, "result", "unknown"))


_INTERVAL_S = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def _closed_only(frame: pl.DataFrame, step_s: int, now_s: float) -> pl.DataFrame:
    """Drop the still-forming candle: keep only bars whose close <= now."""
    if frame.height == 0:
        return frame
    cutoff_ms = int(now_s * 1000) - step_s * 1000
    return frame.filter(pl.col("ts").dt.epoch("ms") <= cutoff_ms)


@runtime_checkable
class LiveFeed(Protocol):
    """Warmup history (preloaded, no trades) + a stream of newly closed bars."""

    quote: str
    interval: "str | None"

    def warmup(self) -> Dataset: ...

    def stream(self) -> Iterator[Bar]: ...


@dataclass
class ReplayFeed(LiveFeed):
    """Replays a finished Dataset bar-by-bar - drives the live loop at full speed.

    ``warmup_bars`` splits off a leading lookback window that fills the buffer
    without trading; decisions then run incrementally over the remaining bars -
    a faithful walk-forward of the live cycle.
    """

    data: Dataset
    interval: "str | None" = None
    warmup_bars: int = 0
    quote: str = field(init=False, default="USDT")

    def __post_init__(self) -> None:
        self.quote = self.data.quote

    def _cutoff(self):
        if self.warmup_bars <= 0:
            return None
        ts = self.data.frame.get_column("ts").unique().sort()
        if self.warmup_bars >= ts.len():
            return ts[-1]
        return ts[self.warmup_bars - 1]

    def warmup(self) -> Dataset:
        cutoff = self._cutoff()
        frame = self.data.frame.head(0) if cutoff is None else self.data.frame.filter(pl.col("ts") <= cutoff)
        return Dataset(frame=frame, quote=self.quote)

    def stream(self) -> Iterator[Bar]:
        cutoff = self._cutoff()
        frame = self.data.frame if cutoff is None else self.data.frame.filter(pl.col("ts") > cutoff)
        yield from Dataset(frame=frame, quote=self.quote).iter_bars()


@dataclass
class PollingFeed(LiveFeed):
    """Polls a Source for newly closed klines, one yield per new bar boundary.

    ``warmup()`` fetches a closed-bar history prefix to fill the buffer. ``stream()``
    sleeps to each interval boundary, fetches the latest CLOSED bar(s), and never
    yields the still-forming current candle.
    """

    source: object
    pairs: list[str]
    interval: str = "1m"
    quote: str = "USDT"
    warmup_bars: "int | None" = None
    max_bars: "int | None" = None
    lag_seconds: float = 3.0
    backfill_bars: int = 5000
    clock: Clock = field(default_factory=lambda: Clock(RunMode.LIVE))

    def __post_init__(self) -> None:
        self._last_ts = None

    def warmup(self) -> Dataset:
        step = _INTERVAL_S[self.interval]
        now = float(self.clock.now() or self.clock.wall())
        start = int(now) - (self.warmup_bars + 1) * step
        frame = _closed_only(self.source.fetch(self.pairs, start=start, interval=self.interval), step, now)
        ds = Dataset(frame=frame, quote=self.quote)
        if frame.height:
            self._last_ts = ds.frame.get_column("ts").max()
        return ds

    def stream(self) -> Iterator[Bar]:
        step = _INTERVAL_S[self.interval]
        emitted = 0
        while self.max_bars is None or emitted < self.max_bars:
            now = self.clock.wall()
            self.clock.sleep((now // step + 1) * step + self.lag_seconds - now)
            now = self.clock.wall()
            if self._last_ts is not None:
                start = max(
                    int(self._last_ts.replace(tzinfo=UTC).timestamp()) + step, int(now - self.backfill_bars * step)
                )
            else:
                start = int(now - 2 * step)
            fresh = _closed_only(self.source.fetch(self.pairs, start=start, interval=self.interval), step, now)
            new_bars = [
                b
                for b in Dataset(frame=fresh, quote=self.quote).iter_bars()
                if self._last_ts is None or b.ts > self._last_ts
            ]
            if self._last_ts is not None and new_bars and (new_bars[0].ts - self._last_ts).total_seconds() > step:
                logger.warning(f"PollingFeed: backfilled {len(new_bars)} missed bars after a gap")
            for bar in new_bars:
                self._last_ts = bar.ts
                emitted += 1
                yield bar
                if self.max_bars is not None and emitted >= self.max_bars:
                    break


def _iso(ts) -> "str | None":
    """Serialize a timestamp to ISO text, tolerating already-string values."""
    if ts is None:
        return None
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


def _parse_ts(value):
    """Parse an ISO timestamp back to a datetime, leaving non-ISO values unchanged."""
    if value is None:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return value


def save_state(engine: Engine, path: str, peak: "float | None" = None) -> None:
    """Persist balances/positions/marks, the fill log, and the drawdown peak so a restart resumes."""
    state = {
        "balances": engine.balances,
        "positions": {
            k: {"pair": v.pair, "qty": v.qty, "avg_price": v.avg_price, "opened_ts": _iso(v.opened_ts)}
            for k, v in engine.positions.items()
        },
        "marks": engine.marks,
        "peak": peak,
        "event_log": [
            {
                "pair": f.pair,
                "ts": _iso(f.ts),
                "side": f.side.value,
                "qty": f.qty,
                "price": f.price,
                "fee": f.fee,
                "fee_asset": f.fee_asset,
            }
            for f in engine.event_log
        ],
        "order_log": [
            {
                "client_order_id": e.client_order_id,
                "pair": e.pair,
                "side": e.side.value,
                "qty": e.qty,
                "ts": _iso(e.ts),
                "kind": e.kind,
                "status": e.status,
            }
            for e in engine.order_log
        ],
    }
    with open(path, "w") as fh:
        json.dump(state, fh)


def load_state(engine: Engine, path: str) -> "dict | None":
    """Restore a book saved by :func:`save_state`; return the parsed state dict (``None`` if absent)."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        state = json.load(fh)
    engine.balances = dict(state.get("balances", {}))
    engine.marks = dict(state.get("marks", {}))
    engine.positions = {
        k: Position(pair=p["pair"], qty=p["qty"], avg_price=p["avg_price"], opened_ts=_parse_ts(p.get("opened_ts")))
        for k, p in state.get("positions", {}).items()
    }
    engine.event_log = [
        Fill(
            pair=e["pair"],
            ts=_parse_ts(e.get("ts")),
            side=Side(e["side"]),
            qty=e["qty"],
            price=e["price"],
            fee=e.get("fee", 0.0),
            fee_asset=e.get("fee_asset", "USDT"),
        )
        for e in state.get("event_log", [])
    ]
    engine.order_log = [
        OrderEvent(
            client_order_id=e["client_order_id"],
            pair=e["pair"],
            side=Side(e["side"]),
            qty=e["qty"],
            ts=_parse_ts(e.get("ts")),
            kind=e["kind"],
            status=e.get("status", ""),
        )
        for e in state.get("order_log", [])
    ]
    return state


def _close_epoch(bar_ts, step_s: int) -> float:
    """Wall-clock epoch (UTC) at which ``bar_ts``'s candle closed."""
    return bar_ts.replace(tzinfo=UTC).timestamp() + step_s


def resolve_warmup_bars(flow, feed: LiveFeed) -> int:
    """Fill a feed's unset (``None``) warmup coverage from the flow's requirement.

    Returns the flow's required warmup so callers can log and bound-check it.
    """
    required = flow.required_warmup
    if getattr(feed, "warmup_bars", 0) is None:
        feed.warmup_bars = required
    return required


def run_live_loop(
    flow,
    feed: LiveFeed,
    capital,
    broker,
    target: "str | None" = None,
    maxlen: int = 5000,
    max_bars: "int | None" = None,
    mandate: "dict | None" = None,
    state_path: "str | None" = None,
    on_bar=None,
    max_latency_s: float = 10.0,
    armed: bool = False,
    compute_window: "int | None" = None,
):
    """Drive a Flow over a live (or replayed) feed.

    Warmup history fills the buffer without trading; decisions begin on the first
    streamed bar. Per live bar, decision latency (execution wall-clock minus the
    bar's close time) is measured and a breach of ``max_latency_s`` is logged.
    Runs are not promotable. Per-bar signals are recomputed over the trailing
    ``required_warmup + 1`` bars (``compute_window`` overrides); a transform that
    under-declares its ``warmup`` will surface as ``simulate != backtest``.
    """
    from signalflow.engine.broker import ExchangeBroker
    from signalflow.errors import FlowConfigError
    from signalflow.flow.run import Run

    if isinstance(broker, ExchangeBroker) and not armed:
        raise FlowConfigError("a real ExchangeBroker requires Flow.live(..., armed=True); use SimBroker for paper")
    if armed and flow.risk.max_drawdown >= 1.0 and flow.risk.max_notional_per_pair >= 1.0:
        logger.warning("risk layer is effectively disabled (default limits) while trading armed")
    required_warmup = resolve_warmup_bars(flow, feed)
    window = compute_window if compute_window is not None else required_warmup + 1
    logger.info(f"live: resolved warmup = {required_warmup} bars")
    if maxlen < required_warmup:
        raise ValueError(
            f"maxlen={maxlen} is below the flow's required warmup of {required_warmup} bars; "
            f"raise maxlen to at least {required_warmup}"
        )
    coverage = getattr(feed, "warmup_bars", None)
    if coverage is not None and coverage < required_warmup:
        logger.warning(
            f"live: feed warmup coverage {coverage} < required {required_warmup}; "
            f"outputs may be NaN-driven until the buffer grows"
        )

    target = target or feed.quote
    engine = Engine(capital, target=target, quote=feed.quote)
    peak = float("-inf")
    if state_path:
        state = load_state(engine, state_path)
        if state is not None:
            logger.info(f"live: resumed book from {state_path}")
            if state.get("peak") is not None:
                peak = float(state["peak"])
            if armed and isinstance(broker, ExchangeBroker) and hasattr(broker, "query_order"):
                _reconcile_on_resume(engine, broker)

    buf: list = []
    for wbar in feed.warmup().iter_bars():
        buf.append(wbar.frame)
    step_s = _INTERVAL_S.get(getattr(feed, "interval", None) or "")

    eq_ts, eq_val = [], []
    started = False
    for n, bar in enumerate(feed.stream()):
        buf.append(bar.frame)
        if len(buf) > maxlen:
            buf = buf[-maxlen:]
        tail = buf[-window:] if len(buf) > window else buf
        hist = Dataset(frame=pl.concat(tail), quote=feed.quote)
        signals = enriched_signals(flow, hist)
        sig_frame = (
            signals.filter(pl.col("ts") == bar.ts) if signals.height else pl.DataFrame(schema=_EMPTY_SIGNALS_SCHEMA)
        )
        snap = engine.snapshot(bar.ts, bar.prices)
        if not started:
            eq_ts.append(bar.ts)
            eq_val.append(snap.equity)
            started = True
        peak = max(peak, snap.equity)
        obs = Observation(bar.ts, sig_frame, snap, mandate or {})
        intents = flow.risk.clip(flow.strategy.decide(obs), snap, peak, raise_on_trip=armed)
        orders = _orders(intents, bar.prices, bar.ts)
        for o in orders:
            engine.record_order(OrderEvent(_order_id(broker, o), o.pair, o.side, o.qty, bar.ts, "placed"))
        fills = broker.execute(orders, bar)
        for o in orders:
            engine.record_order(
                OrderEvent(_order_id(broker, o), o.pair, o.side, o.qty, bar.ts, "result", _order_status(o, fills))
            )
        engine.apply(fills)
        eq_ts.append(bar.ts)
        eq_val.append(engine.equity(bar.prices))

        latency = (time.time() - _close_epoch(bar.ts, step_s)) if step_s else None
        if fills and latency is not None and latency > max_latency_s:
            logger.warning(f"live: order latency {latency:.1f}s exceeds {max_latency_s}s budget at close {bar.ts}")

        if state_path:
            save_state(engine, state_path, peak=peak)
        if on_bar is not None:
            on_bar(engine, bar, fills, latency)
        if max_bars is not None and n + 1 >= max_bars:
            break

    curve = pl.DataFrame({"ts": eq_ts, "equity": eq_val})
    return Run(flow.name, RunMode.LIVE.value, curve, engine.event_log, target, promotable=False)
