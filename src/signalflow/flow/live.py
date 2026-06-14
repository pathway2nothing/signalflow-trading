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
from signalflow.engine.types import Position
from signalflow.enums import RunMode
from signalflow.flow.loop import _EMPTY_SIGNALS_SCHEMA, _orders, enriched_signals
from signalflow.strategy.observation import Observation

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
    quote: str = "USDT"
    interval: "str | None" = None
    warmup_bars: int = 0

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
    warmup_bars: int = 500
    max_bars: "int | None" = None
    lag_seconds: float = 3.0
    clock: Clock = field(default_factory=lambda: Clock(RunMode.LIVE))

    def __post_init__(self) -> None:
        self._last_ts = None

    def warmup(self) -> Dataset:
        step = _INTERVAL_S[self.interval]
        now = float(self.clock.now() or time.time())
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
            now = time.time()
            time.sleep(max(0.0, (now // step + 1) * step + self.lag_seconds - now))
            now = time.time()
            fresh = _closed_only(
                self.source.fetch(self.pairs, start=int(now - 2 * step), interval=self.interval), step, now
            )
            for bar in Dataset(frame=fresh, quote=self.quote).iter_bars():
                if self._last_ts is not None and bar.ts <= self._last_ts:
                    continue
                self._last_ts = bar.ts
                emitted += 1
                yield bar
                if self.max_bars is not None and emitted >= self.max_bars:
                    break


def save_state(engine: Engine, path: str) -> None:
    """Persist balances/positions/marks so a restart resumes the same book."""
    state = {
        "balances": engine.balances,
        "positions": {
            k: {"pair": v.pair, "qty": v.qty, "avg_price": v.avg_price, "opened_ts": str(v.opened_ts)}
            for k, v in engine.positions.items()
        },
        "marks": engine.marks,
    }
    with open(path, "w") as fh:
        json.dump(state, fh)


def load_state(engine: Engine, path: str) -> bool:
    """Restore a book saved by :func:`save_state`. Returns False if absent."""
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        state = json.load(fh)
    engine.balances = dict(state.get("balances", {}))
    engine.marks = dict(state.get("marks", {}))
    engine.positions = {
        k: Position(pair=p["pair"], qty=p["qty"], avg_price=p["avg_price"])
        for k, p in state.get("positions", {}).items()
    }
    return True


def _close_epoch(bar_ts, step_s: int) -> float:
    """Wall-clock epoch (UTC) at which ``bar_ts``'s candle closed."""
    return bar_ts.replace(tzinfo=UTC).timestamp() + step_s


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
):
    """Drive a Flow over a live (or replayed) feed.

    Warmup history fills the buffer without trading; decisions begin on the first
    streamed bar. Per live bar, decision latency (execution wall-clock minus the
    bar's close time) is measured and a breach of ``max_latency_s`` is logged.
    Runs are not promotable.
    """
    from signalflow.flow.run import Run

    target = target or feed.quote
    engine = Engine(capital, target=target, quote=feed.quote)
    if state_path and load_state(engine, state_path):
        logger.info(f"live: resumed book from {state_path}")

    buf: list = []
    for wbar in feed.warmup().iter_bars():
        buf.append(wbar.frame)
    step_s = _INTERVAL_S.get(getattr(feed, "interval", None) or "")

    eq_ts, eq_val = [], []
    peak = float("-inf")
    started = False
    for n, bar in enumerate(feed.stream()):
        buf.append(bar.frame)
        if len(buf) > maxlen:
            buf = buf[-maxlen:]
        hist = Dataset(frame=pl.concat(buf), quote=feed.quote)
        signals = enriched_signals(flow, hist)
        sig_frame = (
            signals.filter(pl.col("ts") == bar.ts)
            if signals.height
            else pl.DataFrame(schema=_EMPTY_SIGNALS_SCHEMA)
        )
        snap = engine.snapshot(bar.ts, bar.prices)
        if not started:
            eq_ts.append(bar.ts)
            eq_val.append(snap.equity)
            started = True
        peak = max(peak, snap.equity)
        obs = Observation(bar.ts, sig_frame, snap, mandate or {})
        intents = flow.risk.clip(flow.strategy.decide(obs), snap, peak)
        fills = broker.execute(_orders(intents, bar.prices, bar.ts), bar)
        engine.apply(fills)
        eq_ts.append(bar.ts)
        eq_val.append(engine.equity(bar.prices))

        latency = (time.time() - _close_epoch(bar.ts, step_s)) if step_s else None
        if fills and latency is not None and latency > max_latency_s:
            logger.warning(f"live: order latency {latency:.1f}s exceeds {max_latency_s}s budget at close {bar.ts}")

        if state_path:
            save_state(engine, state_path)
        if on_bar is not None:
            on_bar(engine, bar, fills, latency)
        if max_bars is not None and n + 1 >= max_bars:
            break

    curve = pl.DataFrame({"ts": eq_ts, "equity": eq_val})
    return Run(flow.name, RunMode.LIVE.value, curve, engine.event_log, target, promotable=False)
