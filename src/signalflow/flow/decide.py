"""The live loop's per-bar body as a public, stateless call - plus the trailing-window buffer it reads.

A runner that owns its own feed, portfolio store and executor (or a paper run that
wants a different fill model) needs exactly one thing from a Flow per closed bar:
*given this history and this portfolio, what would you do now?* :func:`decide`
answers that without touching any state; :class:`Buffer` keeps the trailing
``required_warmup + 1`` bars so nothing is refetched. ``run_live_loop`` is built
from the same two pieces, so ``simulate``/``live`` and an external loop agree.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.flow.loop import EMPTY_SIGNALS_SCHEMA, enriched_signals, orders_from_intents
from signalflow.strategy.observation import Observation


class Buffer:
    """The trailing ``window`` closed bars as one frame ordered by (ts, pair).

    ``push`` accepts one closed bar (all pairs at one ``ts``) or any larger frame;
    the buffer keeps the last ``window`` distinct timestamps. Appends concatenate
    without copying and trims are zero-copy slices, so a live loop pays nothing
    per bar for the history it decides on.
    """

    def __init__(self, window: int) -> None:
        self.window = max(int(window), 1)
        self.frame: pl.DataFrame | None = None
        self._lengths: deque[int] = deque()

    def push(self, frame: pl.DataFrame) -> "Buffer":
        if frame.height == 0:
            return self
        ts = frame.get_column("ts")
        if ts.n_unique() == 1:
            chunks = [frame.height]
        else:
            frame = frame.sort(["ts", "pair"]) if "pair" in frame.columns else frame.sort("ts")
            chunks = frame.group_by("ts", maintain_order=True).len().get_column("len").to_list()
        self.frame = frame if self.frame is None else pl.concat([self.frame, frame], rechunk=False)
        self._lengths.extend(chunks)
        while len(self._lengths) > self.window:
            self.frame = self.frame.slice(self._lengths.popleft())
        if self.frame.n_chunks() > 2 * self.window + 8:
            self.frame = self.frame.rechunk()
        return self

    @property
    def bars(self) -> int:
        """Distinct timestamps currently held."""
        return len(self._lengths)

    @property
    def ts(self) -> Any:
        """Timestamp of the newest bar, or ``None`` when empty."""
        return None if self.frame is None or self.frame.height == 0 else self.frame.get_column("ts").max()

    def dataset(self, quote: str = "USDT") -> Dataset:
        frame = (
            self.frame if self.frame is not None else pl.DataFrame(schema={"pair": pl.Utf8, "ts": pl.Datetime("ms")})
        )
        return Dataset(frame=frame, quote=quote)

    def __repr__(self) -> str:
        return f"Buffer(window={self.window}, bars={self.bars}, ts={self.ts})"


@dataclass(frozen=True)
class Decision:
    """What a flow would do on one bar: the signals it saw, the intents after risk, the orders to send."""

    ts: Any
    signals: pl.DataFrame
    intents: list = field(default_factory=list)
    orders: list = field(default_factory=list)


def decide(
    flow: Any,
    history: Any,
    snapshot: Any,
    ts: Any = None,
    *,
    peak: float | None = None,
    mandate: dict | None = None,
    raise_on_trip: bool = False,
    prices: dict | None = None,
    quote: str | None = None,
) -> Decision:
    """One bar of the live loop with no side effects.

    ``history`` is a :class:`Buffer`, a Dataset or a bare frame holding at least
    ``flow.required_warmup + 1`` closed bars; ``snapshot`` is the portfolio as the
    strategy should see it (an ``Engine.snapshot`` or a ``PortfolioSnapshot`` built
    from the runner's own books). ``ts`` defaults to the newest bar; ``peak`` is the
    equity high-water mark for the drawdown limit (defaults to the snapshot's
    equity); ``prices`` are the reference prices for sizing (default: the
    snapshot's, else the newest closes). Nothing is mutated: call it twice and you
    get the same :class:`Decision`.
    """
    if isinstance(history, Buffer):
        frame = history.frame
    elif isinstance(history, Dataset):
        frame = history.frame
        quote = quote or history.quote
    else:
        frame = history
    if frame is None or frame.height == 0:
        raise ValueError("decide: history is empty")
    ts = frame.get_column("ts").max() if ts is None else ts
    data = Dataset(frame=frame, quote=quote or getattr(flow, "quote", "USDT"))

    signals = enriched_signals(flow, data, log=False)
    sig_frame = signals.filter(pl.col("ts") == ts) if signals.height else pl.DataFrame(schema=EMPTY_SIGNALS_SCHEMA)
    obs = Observation(ts, sig_frame, snapshot, mandate or {})
    equity = float(getattr(snapshot, "equity", 0.0) or 0.0)
    high_water = equity if peak is None else max(float(peak), equity)
    intents = flow.risk.clip(flow.strategy.decide(obs), snapshot, high_water, raise_on_trip=raise_on_trip)
    if prices is None:
        prices = dict(getattr(snapshot, "prices", None) or {}) or _last_closes(frame, ts)
    orders = orders_from_intents(intents, prices, ts)
    return Decision(ts=ts, signals=sig_frame, intents=list(intents), orders=orders)


def _last_closes(frame: pl.DataFrame, ts: Any) -> dict[str, float]:
    rows = frame.filter(pl.col("ts") == ts).select(["pair", "close"])
    return dict(zip(rows.get_column("pair").to_list(), rows.get_column("close").to_list(), strict=True))


__all__ = ["Buffer", "Decision", "decide"]
