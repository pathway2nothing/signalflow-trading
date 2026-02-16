"""Hyperliquid data source - async REST client and loaders for perpetuals."""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiohttp
from loguru import logger

from signalflow.core import sf_component
from signalflow.data.raw_store import DuckDbSpotStore
from signalflow.data.source._helpers import (
    TIMEFRAME_MS,
    dt_to_ms_utc,
    ensure_utc_naive,
    ms_to_dt_utc_naive,
    normalize_hyperliquid_pair,
    to_hyperliquid_coin,
)
from signalflow.data.source.base import RawDataLoader, RawDataSource

# Hyperliquid interval mapping.
_HYPERLIQUID_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
}


@dataclass
@sf_component(name="hyperliquid")
class HyperliquidClient(RawDataSource):
    """Async client for Hyperliquid REST API.

    Provides methods for fetching OHLCV kline data with retries
    and rate-limit handling.

    Returned timestamps are candle **close** times (UTC-naive).
    Hyperliquid is a perpetual DEX (no spot trading).

    Attributes:
        base_url: Hyperliquid API base URL.
        max_retries: Maximum retry attempts.
        timeout_sec: Request timeout in seconds.
        min_delay_sec: Minimum delay between requests.
    """

    base_url: str = "https://api.hyperliquid.xyz/info"
    max_retries: int = 3
    timeout_sec: int = 30
    min_delay_sec: float = 0.1

    _session: aiohttp.ClientSession | None = field(default=None, init=False)

    async def __aenter__(self) -> "HyperliquidClient":
        timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def get_pairs(self) -> list[str]:
        """Get list of available perpetual coins from Hyperliquid.

        Returns:
            list[str]: List of coin symbols (e.g., ["BTC", "ETH", "SOL"]).

        Example:
            ```python
            async with HyperliquidClient() as client:
                coins = await client.get_pairs()
                # ['BTC', 'ETH', 'SOL', ...]
            ```
        """
        if self._session is None:
            raise RuntimeError("HyperliquidClient must be used as an async context manager.")

        payload = {"type": "meta"}

        for attempt in range(self.max_retries):
            try:
                async with self._session.post(self.base_url, json=payload) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Hyperliquid API HTTP {resp.status}: {text}")

                    body = await resp.json()

                if "error" in body:
                    raise RuntimeError(f"Hyperliquid API error: {body['error']}")

                coins: list[str] = []
                universe = body.get("universe", [])
                for asset in universe:
                    name = asset.get("name", "")
                    if name:
                        coins.append(name)

                return sorted(coins)

            except (TimeoutError, aiohttp.ClientError) as e:
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(f"Request failed, retrying in {wait}s: {e}")
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(f"Failed to get pairs: {e}") from e

        return []

    async def get_klines(
        self,
        coin: str,
        timeframe: str = "1m",
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Fetch OHLCV klines from Hyperliquid.

        Returned ``timestamp`` is the candle **close** time (UTC-naive).

        Args:
            coin: Coin symbol (e.g., "BTC", "ETH").
            timeframe: Interval in SignalFlow format (1m, 1h, ...).
            start_time: Fetch candles after this time.
            end_time: Fetch candles before this time.
            limit: Max candles per request (Hyperliquid max: 500).

        Returns:
            List of OHLCV dicts sorted ascending by timestamp.

        Raises:
            RuntimeError: If not in async context or API error.
        """
        if self._session is None:
            raise RuntimeError("HyperliquidClient must be used as an async context manager.")

        interval = _HYPERLIQUID_INTERVAL_MAP.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        req: dict[str, object] = {
            "coin": coin.upper(),
            "interval": interval,
        }
        if start_time is not None:
            req["startTime"] = dt_to_ms_utc(start_time)
        if end_time is not None:
            req["endTime"] = dt_to_ms_utc(end_time)

        payload = {"type": "candleSnapshot", "req": req}
        last_err: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                async with self._session.post(self.base_url, json=payload) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s (coin={coin})")
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Hyperliquid API HTTP {resp.status}: {text}")

                    body = await resp.json()

                if isinstance(body, dict) and "error" in body:
                    raise RuntimeError(f"Hyperliquid API error: {body['error']}")

                # Response is a list of candle objects
                out: list[dict] = []
                for candle in body:
                    # T = close time in milliseconds
                    close_time_ms = int(candle.get("T", candle.get("t", 0)))
                    out.append(
                        {
                            "timestamp": ms_to_dt_utc_naive(close_time_ms),
                            "open": float(candle.get("o", 0)),
                            "high": float(candle.get("h", 0)),
                            "low": float(candle.get("l", 0)),
                            "close": float(candle.get("c", 0)),
                            "volume": float(candle.get("v", 0)),
                            "trades": int(candle.get("n", 0)),
                        }
                    )

                # Sort ascending by timestamp
                out.sort(key=lambda x: x["timestamp"])
                return out

            except (TimeoutError, aiohttp.ClientError, RuntimeError) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(f"Request failed, retrying in {wait}s (coin={coin}): {e}")
                    await asyncio.sleep(wait)
                else:
                    break

        raise last_err or RuntimeError("Unknown error while fetching Hyperliquid klines.")

    async def get_klines_range(
        self,
        coin: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        *,
        limit: int = 500,
    ) -> list[dict]:
        """Download all klines for a period with automatic pagination.

        Note: Hyperliquid has a limit of 5000 total historical candles.

        Args:
            coin: Coin symbol.
            timeframe: Interval (must be in ``TIMEFRAME_MS``).
            start_time: Range start (inclusive).
            end_time: Range end (inclusive).
            limit: Candles per request (max 500).

        Returns:
            Deduplicated, ascending OHLCV dicts.
        """
        if timeframe not in TIMEFRAME_MS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        start_time = ensure_utc_naive(start_time)
        end_time = ensure_utc_naive(end_time)

        if start_time >= end_time:
            return []

        tf_ms = TIMEFRAME_MS[timeframe]
        window = timedelta(milliseconds=tf_ms * limit)

        all_klines: list[dict] = []
        current_start = start_time

        # Hyperliquid has a 5000 candle limit total
        max_candles = 5000
        max_loops = max_candles // limit + 10
        loops = 0

        while current_start < end_time and len(all_klines) < max_candles:
            loops += 1
            if loops > max_loops:
                logger.warning(f"Hyperliquid: reached pagination limit for {coin}")
                break

            req_end = min(current_start + window, end_time)

            klines = await self.get_klines(
                coin=coin,
                timeframe=timeframe,
                start_time=current_start,
                end_time=req_end,
                limit=limit,
            )

            if not klines:
                current_start = req_end + timedelta(milliseconds=1)
                await asyncio.sleep(self.min_delay_sec)
                continue

            for k in klines:
                ts = k["timestamp"]
                if start_time <= ts <= end_time:
                    all_klines.append(k)

            last_ts = klines[-1]["timestamp"]
            next_start = last_ts

            if next_start <= current_start:
                current_start = current_start + timedelta(milliseconds=1)
            else:
                current_start = next_start

            if len(all_klines) and len(all_klines) % 2000 == 0:
                logger.info(f"{coin}: loaded {len(all_klines):,} candles...")

            await asyncio.sleep(self.min_delay_sec)

        uniq: dict[datetime, dict] = {}
        for k in all_klines:
            uniq[k["timestamp"]] = k

        out = list(uniq.values())
        out.sort(key=lambda x: x["timestamp"])
        return out


@dataclass
@sf_component(name="hyperliquid/futures")
class HyperliquidFuturesLoader(RawDataLoader):
    """Downloads and stores Hyperliquid perpetual OHLCV data.

    Pairs are provided in compact format (e.g., "BTCUSD") and
    automatically converted to Hyperliquid coin symbols (e.g., "BTC").

    Attributes:
        store: Storage backend.
        timeframe: Fixed timeframe for all data.
    """

    store: DuckDbSpotStore = field(
        default_factory=lambda: DuckDbSpotStore(db_path=Path("raw_data_hyperliquid_futures.duckdb"))
    )
    timeframe: str = "1m"

    async def get_pairs(self) -> list[str]:
        """Get list of available Hyperliquid perpetual coins.

        Returns:
            list[str]: List of coin symbols (e.g., ["BTC", "ETH"]).

        Example:
            ```python
            loader = HyperliquidFuturesLoader(store=store)
            coins = await loader.get_pairs()
            # ['BTC', 'ETH', 'SOL', ...]
            ```
        """
        async with HyperliquidClient() as client:
            return await client.get_pairs()

    async def download(
        self,
        pairs: list[str],
        days: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        fill_gaps: bool = True,
    ) -> None:
        """Download historical Hyperliquid perpetual data.

        Note: Hyperliquid only keeps ~5000 candles of history.

        Args:
            pairs: Trading pairs (e.g., ["BTCUSD"]) or coins (e.g., ["BTC"]).
            days: Number of days back from *end*. Default: 7.
            start: Range start (overrides *days*).
            end: Range end. Default: now.
            fill_gaps: Detect and fill gaps. Default: True.
        """
        now = datetime.now(UTC).replace(tzinfo=None)
        if end is None:
            end = now
        else:
            end = ensure_utc_naive(end)

        if start is None:
            start = end - timedelta(days=days if days else 7)
        else:
            start = ensure_utc_naive(start)

        tf_minutes = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "8h": 480,
            "12h": 720,
            "1d": 1440,
        }.get(self.timeframe, 1)

        async def download_pair(client: HyperliquidClient, pair: str) -> None:
            # Convert compact pair to coin symbol if needed
            coin = to_hyperliquid_coin(pair)
            store_pair = normalize_hyperliquid_pair(coin)

            logger.info(f"Processing {pair} -> {coin} (hyperliquid/futures) from {start} to {end}")

            db_min, db_max = self.store.get_time_bounds(store_pair)
            ranges_to_download: list[tuple[datetime, datetime]] = []

            if db_min is None:
                ranges_to_download.append((start, end))
            else:
                if start < db_min:
                    pre_end = min(end, db_min - timedelta(minutes=tf_minutes))
                    if start < pre_end:
                        ranges_to_download.append((start, pre_end))
                if end > db_max:
                    post_start = max(start, db_max + timedelta(minutes=tf_minutes))
                    if post_start < end:
                        ranges_to_download.append((post_start, end))
                if fill_gaps:
                    overlap_start = max(start, db_min)
                    overlap_end = min(end, db_max)
                    if overlap_start < overlap_end:
                        gaps = self.store.find_gaps(store_pair, overlap_start, overlap_end, tf_minutes)
                        ranges_to_download.extend(gaps)

            for range_start, range_end in ranges_to_download:
                if range_start >= range_end:
                    continue
                logger.info(f"{store_pair}: downloading {range_start} -> {range_end}")
                try:
                    klines = await client.get_klines_range(
                        coin=coin,
                        timeframe=self.timeframe,
                        start_time=range_start,
                        end_time=range_end,
                    )
                    self.store.insert_klines(store_pair, klines)
                except Exception as e:
                    logger.error(f"Error downloading {store_pair}: {e}")

        async with HyperliquidClient() as client:
            await asyncio.gather(*[download_pair(client, pair) for pair in pairs])

        self.store.close()

    async def sync(
        self,
        pairs: list[str],
        update_interval_sec: int = 60,
    ) -> None:
        """Continuously sync latest Hyperliquid perpetual data.

        Args:
            pairs: Trading pairs or coins to sync.
            update_interval_sec: Update interval in seconds.
        """
        logger.info(f"Starting real-time sync (hyperliquid/futures) for {pairs}")
        logger.info(f"Update interval: {update_interval_sec}s (timeframe={self.timeframe})")

        async def fetch_and_store(client: HyperliquidClient, pair: str) -> None:
            coin = to_hyperliquid_coin(pair)
            store_pair = normalize_hyperliquid_pair(coin)
            try:
                klines = await client.get_klines(
                    coin=coin,
                    timeframe=self.timeframe,
                    limit=5,
                )
                self.store.insert_klines(store_pair, klines)
            except Exception as e:
                logger.error(f"Error syncing {store_pair}: {e}")

        async with HyperliquidClient() as client:
            while True:
                await asyncio.gather(*[fetch_and_store(client, pair) for pair in pairs])
                logger.debug(f"Synced {len(pairs)} pairs (hyperliquid/futures)")
                await asyncio.sleep(update_interval_sec)
