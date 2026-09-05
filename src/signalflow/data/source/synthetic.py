"""Deterministic synthetic source - for tests, examples, and offline work."""

import math
from dataclasses import dataclass

import polars as pl

from signalflow.data.source.base import Source, interval_seconds, parse_time, validate_frame
from signalflow.decorators import source


def _seed_for(pair: str, base: int) -> int:
    return base + sum(ord(c) for c in pair) * 2654435761 & 0x7FFFFFFF


@source("synthetic")
@dataclass
class SyntheticSource(Source):
    """Synthetic OHLCV generator: a deterministic random walk, not market data.

    Prices start near ``start_price`` (offset by a hash of the pair name) and
    follow a log-normal walk with per-bar ``drift`` and ``vol``; volume is noise
    around 100. Pair names only seed the generator - ``BTCUSDT`` here has nothing
    to do with real BTC quotes. Use ``binance`` for real candles.
    """

    name: str = "synthetic"
    seed: int = 7
    drift: float = 0.00001
    vol: float = 0.002
    start_price: float = 100.0

    def fetch(
        self,
        pairs: list[str],
        start: str,
        end: str | None = None,
        interval: str = "1h",
    ) -> pl.DataFrame:
        step = interval_seconds(interval)
        start_dt = parse_time(start)
        end_dt = parse_time(end) if end else start_dt + 5000 * step
        n = max(1, int((end_dt - start_dt) // step))

        frames: list[pl.DataFrame] = []
        for pair in pairs:
            rng = _Lcg(_seed_for(pair, self.seed))
            ts: list[int] = []
            o: list[float] = []
            h: list[float] = []
            lo: list[float] = []
            c: list[float] = []
            v: list[float] = []
            price = self.start_price * (1.0 + (sum(ord(c) for c in pair) % 50) / 100.0)
            t = start_dt
            for _ in range(n):
                ret = self.drift + self.vol * rng.normal()
                new_price = max(0.01, price * math.exp(ret))
                hi = max(price, new_price) * (1.0 + abs(rng.normal()) * self.vol)
                low = min(price, new_price) * (1.0 - abs(rng.normal()) * self.vol)
                ts.append(t * 1000)
                o.append(price)
                h.append(hi)
                lo.append(low)
                c.append(new_price)
                v.append(100.0 + abs(rng.normal()) * 50.0)
                price = new_price
                t += step
            frames.append(
                pl.DataFrame(
                    {
                        "pair": [pair] * n,
                        "ts": ts,
                        "open": o,
                        "high": h,
                        "low": lo,
                        "close": c,
                        "volume": v,
                    }
                ).with_columns(pl.col("ts").cast(pl.Datetime("ms")))
            )
        return validate_frame(pl.concat(frames))


MemorySource = SyntheticSource
"""Deprecated alias for :class:`SyntheticSource`; the registry name ``memory`` maps to ``synthetic``."""


class _Lcg:
    """Tiny deterministic RNG (no global state, reproducible across platforms)."""

    def __init__(self, seed: int) -> None:
        self.state = (seed or 1) & 0xFFFFFFFF
        self._spare: float | None = None

    def _uniform(self) -> float:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state / 0x7FFFFFFF

    def normal(self) -> float:
        if self._spare is not None:
            s, self._spare = self._spare, None
            return s
        u1 = max(1e-12, self._uniform())
        u2 = self._uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)
