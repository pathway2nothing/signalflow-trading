"""Deterministic synthetic source - for tests, examples, and offline work."""

import math
from dataclasses import dataclass

import polars as pl

from signalflow._time import interval_seconds, to_epoch
from signalflow.data.source.base import Source, validate_frame
from signalflow.decorators import source

_DAY = 86_400.0

# Approximate spot level (USD) and daily quote turnover (USD) of well-known base
# assets, so that ``BTCUSDT`` starts near a BTC-like price and trades BTC-like
# volume. Anything else starts at ``start_price`` (offset by a hash of the pair
# name) and trades ``turnover`` per day. Orders of magnitude, not quotes.
_KNOWN_ASSETS: dict[str, tuple[float, float]] = {
    "BTC": (100_000.0, 2.0e9),
    "ETH": (4_000.0, 1.0e9),
    "BNB": (800.0, 2.0e8),
    "SOL": (200.0, 5.0e8),
    "XRP": (2.5, 3.0e8),
    "DOGE": (0.3, 2.0e8),
    "ADA": (0.8, 1.0e8),
    "TRX": (0.3, 5.0e7),
    "AVAX": (40.0, 8.0e7),
    "LINK": (20.0, 1.0e8),
    "DOT": (7.0, 5.0e7),
    "LTC": (100.0, 1.0e8),
    "TON": (6.0, 5.0e7),
    "SUI": (3.0, 1.0e8),
    "POL": (0.4, 3.0e7),
    "MATIC": (0.4, 3.0e7),
}
_STABLE_QUOTES = ("USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI", "USD", "EUR")


def _seed_for(pair: str, base: int) -> int:
    return base + sum(ord(c) for c in pair) * 2654435761 & 0x7FFFFFFF


def _split_pair(pair: str) -> tuple[str, str]:
    """``ETHBTC`` -> (``ETH``, ``BTC``); unrecognised layouts -> (pair, "")."""
    quotes = (*_STABLE_QUOTES, *sorted(_KNOWN_ASSETS, key=len, reverse=True))
    for quote in quotes:
        if pair.endswith(quote) and len(pair) > len(quote):
            return pair[: -len(quote)], quote
    return pair, ""


def _level(pair: str, start_price: float, turnover: float) -> tuple[float, float]:
    """Starting price and daily turnover of ``pair``, both in quote units."""
    base, quote = _split_pair(pair)
    quote_usd = _KNOWN_ASSETS[quote][0] if quote in _KNOWN_ASSETS else 1.0
    if base in _KNOWN_ASSETS:
        price_usd, turnover_usd = _KNOWN_ASSETS[base]
    elif base in _STABLE_QUOTES:
        price_usd, turnover_usd = 1.0, turnover
    else:
        price_usd = start_price * (1.0 + (sum(ord(c) for c in pair) % 50) / 100.0)
        turnover_usd = turnover
    return price_usd / quote_usd, turnover_usd / quote_usd


@source("synthetic")
@dataclass
class SyntheticSource(Source):
    """Synthetic OHLCV generator: a deterministic random walk, not market data.

    Prices follow a log-normal walk whose ``drift`` and ``vol`` are quoted **per
    day** and rescaled to the requested interval (``vol * sqrt(bar / day)``,
    ``drift * bar / day``), so 1m, 1h and 1d series of one pair share the same
    annualised volatility. Well-known base assets start near a realistic level
    and trade a realistic daily turnover (``BTCUSDT`` ~ 100k and ~2e9 USDT/day,
    ``ETHBTC`` ~ 0.04); anything else starts at ``start_price`` (offset by a hash
    of the pair name) and trades ``turnover`` per day. Volume scales with the
    interval and with the size of each bar's move. Pair names only seed the
    generator and pick the level - nothing here is a real quote. Use ``binance``
    for real candles. ``ts`` is each bar's close time, like every source.
    """

    name: str = "synthetic"
    seed: int = 7
    drift: float = 0.0002  # log-drift per day (~7 % a year)
    vol: float = 0.03  # sigma of daily log returns (BTC-like)
    start_price: float = 100.0  # pairs outside the built-in table
    turnover: float = 1.0e7  # quote volume per day for pairs outside the table

    def fetch(
        self,
        pairs: list[str],
        start: str,
        end: str | None = None,
        interval: str = "1h",
    ) -> pl.DataFrame:
        step = interval_seconds(interval)
        start_dt = to_epoch(start)
        end_dt = to_epoch(end) if end else start_dt + 5000 * step
        n = max(1, int((end_dt - start_dt) // step))
        frac = step / _DAY
        vol_bar = self.vol * math.sqrt(frac)
        drift_bar = self.drift * frac

        frames: list[pl.DataFrame] = []
        for pair in pairs:
            rng = _Lcg(_seed_for(pair, self.seed))
            price, turnover = _level(pair, self.start_price, self.turnover)
            bar_volume = turnover / price * frac  # base units per bar
            ts: list[int] = []
            o: list[float] = []
            h: list[float] = []
            lo: list[float] = []
            c: list[float] = []
            v: list[float] = []
            t = start_dt
            for _ in range(n):
                z = rng.normal()
                new_price = max(1e-9, price * math.exp(drift_bar + vol_bar * z))
                hi = max(price, new_price) * (1.0 + abs(rng.normal()) * vol_bar * 0.5)
                low = min(price, new_price) * (1.0 - abs(rng.normal()) * vol_bar * 0.5)
                # busier bars on bigger moves; log-normal noise with unit mean
                activity = (0.6 + 0.5 * abs(z)) * math.exp(0.4 * rng.normal() - 0.08)
                ts.append((t + step) * 1000)  # ts is the bar's close
                o.append(price)
                h.append(hi)
                lo.append(low)
                c.append(new_price)
                v.append(bar_volume * activity)
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
