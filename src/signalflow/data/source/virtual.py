"""Virtual data provider — generates synthetic OHLCV bars for testing."""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from signalflow.core.decorators import sf_component
from signalflow.data.source.base import RawDataLoader

_TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
}


def generate_ohlcv(
    pair: str,
    start: datetime,
    n_bars: int,
    timeframe: str = "1m",
    base_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seed: int | None = None,
) -> list[dict]:
    """Generate synthetic OHLCV bars with a random walk + optional trend.

    The generated data has realistic properties:
    - Prices follow a geometric random walk with drift
    - High/low are derived from close with spread
    - Volume varies with volatility

    Args:
        pair: Trading pair name.
        start: Timestamp of the first bar.
        n_bars: Number of bars to generate.
        timeframe: Candle interval.
        base_price: Starting price.
        volatility: Per-bar return standard deviation.
        trend: Per-bar drift (positive = uptrend).
        seed: Random seed for reproducibility.

    Returns:
        List of kline dicts ready for ``insert_klines()``.
    """
    rng = random.Random(seed)
    tf_minutes = _TIMEFRAME_MINUTES.get(timeframe, 1)
    delta = timedelta(minutes=tf_minutes)

    bars: list[dict] = []
    price = base_price

    for i in range(n_bars):
        ts = start + delta * i

        # Geometric random walk
        ret = trend + volatility * rng.gauss(0, 1)
        close = price * (1 + ret)
        close = max(close, 0.01)  # floor

        # Derive OHLCV
        spread = abs(ret) + volatility * 0.5
        high = close * (1 + abs(rng.gauss(0, spread * 0.5)))
        low = close * (1 - abs(rng.gauss(0, spread * 0.5)))
        open_ = price  # open = previous close

        high = max(high, open_, close)
        low = min(low, open_, close)
        low = max(low, 0.01)

        volume = 1000.0 * (1 + abs(ret) / volatility) * (0.5 + rng.random())

        bars.append(
            {
                "timestamp": ts,
                "open": round(open_, 8),
                "high": round(high, 8),
                "low": round(low, 8),
                "close": round(close, 8),
                "volume": round(volume, 2),
                "trades": rng.randint(10, 500),
            }
        )
        price = close

    return bars


def generate_crossover_data(
    pair: str,
    start: datetime,
    n_bars: int,
    timeframe: str = "1m",
    base_price: float = 100.0,
    crossover_at: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Generate data with a guaranteed SMA crossover for signal testing.

    Creates a price series that trends down then sharply up, producing
    a fast-SMA / slow-SMA crossover around ``crossover_at``.

    Args:
        pair: Trading pair name.
        start: Timestamp of the first bar.
        n_bars: Number of bars.
        timeframe: Candle interval.
        base_price: Starting price.
        crossover_at: Bar index where the crossover should occur.
            Defaults to ``n_bars * 2 // 3``.
        seed: Random seed.

    Returns:
        List of kline dicts.
    """
    if crossover_at is None:
        crossover_at = n_bars * 2 // 3

    rng = random.Random(seed)
    tf_minutes = _TIMEFRAME_MINUTES.get(timeframe, 1)
    delta = timedelta(minutes=tf_minutes)

    bars: list[dict] = []
    price = base_price

    for i in range(n_bars):
        ts = start + delta * i

        # Downtrend then uptrend to create crossover
        if i < crossover_at:
            drift = -0.001
        else:
            drift = 0.003  # sharp reversal

        noise = 0.005 * rng.gauss(0, 1)
        ret = drift + noise
        close = price * (1 + ret)
        close = max(close, 0.01)

        spread = abs(ret) + 0.003
        high = close * (1 + abs(rng.gauss(0, spread * 0.3)))
        low = close * (1 - abs(rng.gauss(0, spread * 0.3)))
        open_ = price

        high = max(high, open_, close)
        low = min(low, open_, close)
        low = max(low, 0.01)

        volume = 1000.0 * (0.5 + rng.random())

        bars.append(
            {
                "timestamp": ts,
                "open": round(open_, 8),
                "high": round(high, 8),
                "low": round(low, 8),
                "close": round(close, 8),
                "volume": round(volume, 2),
                "trades": rng.randint(10, 200),
            }
        )
        price = close

    return bars


@dataclass
@sf_component(name="virtual/spot")
class VirtualDataProvider(RawDataLoader):
    """Generates and streams synthetic OHLCV data into a store.

    Drop-in replacement for ``BinanceSpotLoader`` in tests and paper
    trading dry-runs.  Data is generated deterministically when a seed
    is provided.

    Attributes:
        store: Any raw data store with ``insert_klines(pair, klines)``.
        timeframe: Candle interval.
        base_prices: Starting price per pair (defaults to 100.0).
        volatility: Per-bar return std dev.
        trend: Per-bar drift.
        seed: Random seed for reproducibility.
    """

    store: Any = None
    timeframe: str = "1m"
    base_prices: dict[str, float] = field(default_factory=dict)
    volatility: float = 0.02
    trend: float = 0.0001
    seed: int | None = 42

    # Track last generated price per pair for continuation
    _last_prices: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _bars_generated: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def download(
        self,
        pairs: list[str] | None = None,
        n_bars: int = 200,
        start: datetime | None = None,
        **kwargs: Any,
    ) -> None:
        """Pre-populate store with historical data.

        Args:
            pairs: Trading pairs to generate.
            n_bars: Number of bars per pair.
            start: Start timestamp.  Defaults to ``n_bars`` intervals
                before now.
        """
        pairs = pairs or []
        tf_minutes = _TIMEFRAME_MINUTES.get(self.timeframe, 1)

        if start is None:
            start = datetime(2024, 1, 1)

        for pair in pairs:
            base = self.base_prices.get(pair, 100.0)
            pair_seed = None if self.seed is None else self.seed + hash(pair) % 10000

            klines = generate_ohlcv(
                pair=pair,
                start=start,
                n_bars=n_bars,
                timeframe=self.timeframe,
                base_price=base,
                volatility=self.volatility,
                trend=self.trend,
                seed=pair_seed,
            )
            self.store.insert_klines(pair, klines)

            # Track state for continuation in sync()
            if klines:
                self._last_prices[pair] = klines[-1]["close"]
                self._bars_generated[pair] = n_bars

            logger.info(f"VirtualDataProvider: generated {n_bars} bars for {pair}")

    async def sync(
        self,
        pairs: list[str],
        update_interval_sec: int = 60,
    ) -> None:
        """Continuously generate new bars at a fixed interval.

        Mimics ``BinanceSpotLoader.sync()`` — runs forever, writing
        new bars to the store each cycle.

        Args:
            pairs: Trading pairs to stream.
            update_interval_sec: Seconds between new bars.
        """
        logger.info(f"VirtualDataProvider sync started pairs={pairs} interval={update_interval_sec}s")

        tf_minutes = _TIMEFRAME_MINUTES.get(self.timeframe, 1)
        delta = timedelta(minutes=tf_minutes)
        rng = random.Random(self.seed)

        while True:
            for pair in pairs:
                price = self._last_prices.get(pair, self.base_prices.get(pair, 100.0))
                n = self._bars_generated.get(pair, 0)

                # Get last timestamp from store
                _, max_ts = self.store.get_time_bounds(pair)
                if max_ts is None:
                    ts = datetime(2024, 1, 1)
                else:
                    ts = max_ts + delta

                # Generate one new bar
                ret = self.trend + self.volatility * rng.gauss(0, 1)
                close = price * (1 + ret)
                close = max(close, 0.01)

                spread = abs(ret) + self.volatility * 0.5
                high = close * (1 + abs(rng.gauss(0, spread * 0.3)))
                low = close * (1 - abs(rng.gauss(0, spread * 0.3)))
                open_ = price

                high = max(high, open_, close)
                low = min(low, open_, close)
                low = max(low, 0.01)

                volume = 1000.0 * (0.5 + rng.random())

                kline = {
                    "timestamp": ts,
                    "open": round(open_, 8),
                    "high": round(high, 8),
                    "low": round(low, 8),
                    "close": round(close, 8),
                    "volume": round(volume, 2),
                    "trades": rng.randint(10, 200),
                }

                self.store.insert_klines(pair, [kline])
                self._last_prices[pair] = close
                self._bars_generated[pair] = n + 1

            logger.debug(f"VirtualDataProvider: synced {len(pairs)} pairs")
            await asyncio.sleep(update_interval_sec)
