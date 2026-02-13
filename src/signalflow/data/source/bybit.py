"""Bybit data source - async REST client and loaders for spot & futures."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiohttp
from loguru import logger

from signalflow.core import sf_component
from signalflow.data.raw_store import DuckDbSpotStore
from signalflow.data.source.base import RawDataSource, RawDataLoader
from signalflow.data.source._helpers import (
    TIMEFRAME_MS,
    dt_to_ms_utc,
    ms_to_dt_utc_naive,
    ensure_utc_naive,
)

# Bybit-specific interval mapping (their API uses different names).
_BYBIT_INTERVAL_MAP: dict[str, str] = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
}


@dataclass
@sf_component(name="bybit")
class BybitClient(RawDataSource):
    """Async client for Bybit v5 REST API.

    Provides methods for fetching OHLCV kline data with automatic retries,
    rate-limit handling, and pagination.

    Returned timestamps are candle **close** times (open + 1 tf), UTC-naive.

    Attributes:
        base_url: Bybit API base URL.
        max_retries: Maximum retry attempts.
        timeout_sec: Request timeout in seconds.
        min_delay_sec: Minimum delay between requests.
    """

    base_url: str = "https://api.bybit.com"
    max_retries: int = 3
    timeout_sec: int = 30
    min_delay_sec: float = 0.05

    _session: Optional[aiohttp.ClientSession] = field(default=None, init=False)

    async def __aenter__(self) -> "BybitClient":
        timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def get_pairs(
        self,
        category: str = "spot",
        quote: str | None = None,
    ) -> list[str]:
        """Get list of available instruments from Bybit.

        Args:
            category (str): Market category. One of:
                - "spot" for spot trading
                - "linear" for USDT perpetuals
                - "inverse" for coin-margined contracts
            quote (str | None): Filter by quote/settlement currency (e.g., "USDT").
                If None, returns all instruments.

        Returns:
            list[str]: List of symbols (e.g., ["BTCUSDT", "ETHUSDT"]).

        Example:
            ```python
            async with BybitClient() as client:
                # All spot USDT pairs
                spot = await client.get_pairs(category="spot", quote="USDT")

                # All linear perpetuals
                perps = await client.get_pairs(category="linear")
            ```
        """
        if self._session is None:
            raise RuntimeError("BybitClient must be used as an async context manager.")

        url = f"{self.base_url}/v5/market/instruments-info"
        params: dict[str, str] = {"category": category}

        for attempt in range(self.max_retries):
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Bybit API HTTP {resp.status}: {text}")

                    body = await resp.json()

                if body.get("retCode") != 0:
                    raise RuntimeError(f"Bybit API error {body.get('retCode')}: {body.get('retMsg')}")

                pairs: list[str] = []
                for inst in body.get("result", {}).get("list", []):
                    if inst.get("status") != "Trading":
                        continue
                    symbol = inst.get("symbol", "")
                    # Quote filtering: quoteCoin for spot, settleCoin for derivatives
                    ccy_field = "quoteCoin" if category == "spot" else "settleCoin"
                    if quote is None or inst.get(ccy_field) == quote:
                        pairs.append(symbol)

                return sorted(pairs)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(f"Request failed, retrying in {wait}s: {e}")
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(f"Failed to get instruments: {e}") from e

        return []

    async def get_klines(
        self,
        pair: str,
        category: str = "spot",
        timeframe: str = "1m",
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Fetch OHLCV klines from Bybit.

        Returned ``timestamp`` is the candle **close** time (open + 1 tf, UTC-naive).

        Args:
            pair: Trading pair (e.g. ``"BTCUSDT"``).
            category: Market category - ``"spot"``, ``"linear"`` or ``"inverse"``.
            timeframe: Interval (1m, 5m, 1h, 1d, etc.).
            start_time: Range start (naive=UTC or aware).
            end_time: Range end (naive=UTC or aware).
            limit: Max candles per request (max 1000).

        Returns:
            List of OHLCV dicts sorted ascending by timestamp.

        Raises:
            RuntimeError: If not in async context or API error.
        """
        if self._session is None:
            raise RuntimeError("BybitClient must be used as an async context manager.")

        interval = _BYBIT_INTERVAL_MAP.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        params: dict[str, object] = {
            "category": category,
            "symbol": pair,
            "interval": interval,
            "limit": int(limit),
        }
        if start_time is not None:
            params["start"] = dt_to_ms_utc(start_time)
        if end_time is not None:
            params["end"] = dt_to_ms_utc(end_time)

        url = f"{self.base_url}/v5/market/kline"
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s (pair={pair})")
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Bybit API HTTP {resp.status}: {text}")

                    body = await resp.json()

                if body.get("retCode") != 0:
                    raise RuntimeError(f"Bybit API error {body.get('retCode')}: {body.get('retMsg')}")

                rows = body.get("result", {}).get("list", [])

                tf_ms = TIMEFRAME_MS.get(timeframe, 60_000)
                out: list[dict] = []
                for row in reversed(rows):  # Bybit returns descending - reverse
                    out.append(
                        {
                            "timestamp": ms_to_dt_utc_naive(int(row[0]) + tf_ms),
                            "open": float(row[1]),
                            "high": float(row[2]),
                            "low": float(row[3]),
                            "close": float(row[4]),
                            "volume": float(row[5]),
                            "trades": 0,
                        }
                    )
                return out

            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(f"Request failed, retrying in {wait}s (pair={pair}): {e}")
                    await asyncio.sleep(wait)
                else:
                    break

        raise last_err or RuntimeError("Unknown error while fetching Bybit klines.")

    async def get_klines_range(
        self,
        pair: str,
        category: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        *,
        limit: int = 1000,
    ) -> list[dict]:
        """Download all klines for a period with automatic pagination.

        Args:
            pair: Trading pair.
            category: Market category.
            timeframe: Interval (must be in ``TIMEFRAME_MS``).
            start_time: Range start (inclusive).
            end_time: Range end (inclusive).
            limit: Candles per request.

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

        max_loops = 2_000_000
        loops = 0

        while current_start < end_time:
            loops += 1
            if loops > max_loops:
                raise RuntimeError("Pagination guard triggered (too many loops).")

            req_end = min(current_start + window, end_time)

            klines = await self.get_klines(
                pair=pair,
                category=category,
                timeframe=timeframe,
                start_time=current_start,
                end_time=req_end,
                limit=limit,
            )

            if not klines:
                current_start = req_end + timedelta(milliseconds=1)
                await asyncio.sleep(self.min_delay_sec)
                continue

            klines.sort(key=lambda x: x["timestamp"])

            for k in klines:
                ts = k["timestamp"]
                if start_time <= ts <= end_time:
                    all_klines.append(k)

            last_ts = klines[-1]["timestamp"]  # close time (open + tf)
            next_start = last_ts  # close time = next candle's open time

            if next_start <= current_start:
                current_start = current_start + timedelta(milliseconds=1)
            else:
                current_start = next_start

            if len(all_klines) and len(all_klines) % 10000 == 0:
                logger.info(f"{pair}: loaded {len(all_klines):,} candles...")

            await asyncio.sleep(self.min_delay_sec)

        uniq: dict[datetime, dict] = {}
        for k in all_klines:
            uniq[k["timestamp"]] = k

        out = list(uniq.values())
        out.sort(key=lambda x: x["timestamp"])
        return out


@dataclass
@sf_component(name="bybit/spot")
class BybitSpotLoader(RawDataLoader):
    """Downloads and stores Bybit spot OHLCV data.

    Attributes:
        store: Storage backend.
        timeframe: Fixed timeframe for all data.
    """

    store: DuckDbSpotStore = field(default_factory=lambda: DuckDbSpotStore(db_path=Path("raw_data_bybit_spot.duckdb")))
    timeframe: str = "1m"

    async def get_pairs(self, quote: str | None = None) -> list[str]:
        """Get list of available Bybit spot trading pairs.

        Args:
            quote (str | None): Filter by quote currency (e.g., "USDT").
                If None, returns all spot pairs.

        Returns:
            list[str]: List of symbols (e.g., ["BTCUSDT", "ETHUSDT"]).

        Example:
            ```python
            loader = BybitSpotLoader(store=store)
            usdt_pairs = await loader.get_pairs(quote="USDT")
            # ['BTCUSDT', 'ETHUSDT', ...]
            ```
        """
        async with BybitClient() as client:
            return await client.get_pairs(category="spot", quote=quote)

    async def download(
        self,
        pairs: list[str],
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        fill_gaps: bool = True,
    ) -> None:
        """Download historical Bybit spot data.

        Args:
            pairs: Trading pairs to download.
            days: Number of days back from *end*. Default: 7.
            start: Range start (overrides *days*).
            end: Range end. Default: now.
            fill_gaps: Detect and fill gaps. Default: True.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
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
            "6h": 360,
            "8h": 480,
            "12h": 720,
            "1d": 1440,
        }.get(self.timeframe, 1)

        async def download_pair(client: BybitClient, pair: str) -> None:
            logger.info(f"Processing {pair} (bybit/spot) from {start} to {end}")
            db_min, db_max = self.store.get_time_bounds(pair)
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
                        gaps = self.store.find_gaps(pair, overlap_start, overlap_end, tf_minutes)
                        ranges_to_download.extend(gaps)

            for range_start, range_end in ranges_to_download:
                if range_start >= range_end:
                    continue
                logger.info(f"{pair}: downloading {range_start} -> {range_end}")
                try:
                    klines = await client.get_klines_range(
                        pair=pair,
                        category="spot",
                        timeframe=self.timeframe,
                        start_time=range_start,
                        end_time=range_end,
                    )
                    self.store.insert_klines(pair, klines)
                except Exception as e:
                    logger.error(f"Error downloading {pair}: {e}")

        async with BybitClient() as client:
            await asyncio.gather(*[download_pair(client, pair) for pair in pairs])

        self.store.close()

    async def sync(
        self,
        pairs: list[str],
        update_interval_sec: int = 60,
    ) -> None:
        """Continuously sync latest Bybit spot data.

        Args:
            pairs: Trading pairs to sync.
            update_interval_sec: Update interval in seconds.
        """
        logger.info(f"Starting real-time sync (bybit/spot) for {pairs}")
        logger.info(f"Update interval: {update_interval_sec}s (timeframe={self.timeframe})")

        async def fetch_and_store(client: BybitClient, pair: str) -> None:
            try:
                klines = await client.get_klines(
                    pair=pair,
                    category="spot",
                    timeframe=self.timeframe,
                    limit=5,
                )
                self.store.insert_klines(pair, klines)
            except Exception as e:
                logger.error(f"Error syncing {pair}: {e}")

        async with BybitClient() as client:
            while True:
                await asyncio.gather(*[fetch_and_store(client, pair) for pair in pairs])
                logger.debug(f"Synced {len(pairs)} pairs (bybit/spot)")
                await asyncio.sleep(update_interval_sec)


@dataclass
@sf_component(name="bybit/futures")
class BybitFuturesLoader(RawDataLoader):
    """Downloads and stores Bybit futures OHLCV data.

    Attributes:
        store: Storage backend.
        timeframe: Fixed timeframe for all data.
        category: Bybit market category - ``"linear"`` (USDT perps) or
            ``"inverse"`` (coin-margined).
    """

    store: DuckDbSpotStore = field(
        default_factory=lambda: DuckDbSpotStore(db_path=Path("raw_data_bybit_futures.duckdb"))
    )
    timeframe: str = "1m"
    category: str = "linear"

    async def get_pairs(self, quote: str | None = None) -> list[str]:
        """Get list of available Bybit futures/perpetual pairs.

        Args:
            quote (str | None): Filter by settlement currency (e.g., "USDT").
                If None, returns all pairs for the configured category.

        Returns:
            list[str]: List of symbols (e.g., ["BTCUSDT", "ETHUSDT"]).

        Example:
            ```python
            loader = BybitFuturesLoader(store=store, category="linear")
            usdt_perps = await loader.get_pairs(quote="USDT")
            # ['BTCUSDT', 'ETHUSDT', ...]
            ```
        """
        async with BybitClient() as client:
            return await client.get_pairs(category=self.category, quote=quote)

    async def download(
        self,
        pairs: list[str],
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        fill_gaps: bool = True,
    ) -> None:
        """Download historical Bybit futures data.

        Args:
            pairs: Trading pairs to download.
            days: Number of days back from *end*. Default: 7.
            start: Range start (overrides *days*).
            end: Range end. Default: now.
            fill_gaps: Detect and fill gaps. Default: True.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
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
            "6h": 360,
            "8h": 480,
            "12h": 720,
            "1d": 1440,
        }.get(self.timeframe, 1)

        async def download_pair(client: BybitClient, pair: str) -> None:
            logger.info(f"Processing {pair} (bybit/futures/{self.category}) from {start} to {end}")
            db_min, db_max = self.store.get_time_bounds(pair)
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
                        gaps = self.store.find_gaps(pair, overlap_start, overlap_end, tf_minutes)
                        ranges_to_download.extend(gaps)

            for range_start, range_end in ranges_to_download:
                if range_start >= range_end:
                    continue
                logger.info(f"{pair}: downloading {range_start} -> {range_end}")
                try:
                    klines = await client.get_klines_range(
                        pair=pair,
                        category=self.category,
                        timeframe=self.timeframe,
                        start_time=range_start,
                        end_time=range_end,
                    )
                    self.store.insert_klines(pair, klines)
                except Exception as e:
                    logger.error(f"Error downloading {pair}: {e}")

        async with BybitClient() as client:
            await asyncio.gather(*[download_pair(client, pair) for pair in pairs])

        self.store.close()

    async def sync(
        self,
        pairs: list[str],
        update_interval_sec: int = 60,
    ) -> None:
        """Continuously sync latest Bybit futures data.

        Args:
            pairs: Trading pairs to sync.
            update_interval_sec: Update interval in seconds.
        """
        logger.info(f"Starting real-time sync (bybit/futures/{self.category}) for {pairs}")
        logger.info(f"Update interval: {update_interval_sec}s (timeframe={self.timeframe})")

        async def fetch_and_store(client: BybitClient, pair: str) -> None:
            try:
                klines = await client.get_klines(
                    pair=pair,
                    category=self.category,
                    timeframe=self.timeframe,
                    limit=5,
                )
                self.store.insert_klines(pair, klines)
            except Exception as e:
                logger.error(f"Error syncing {pair}: {e}")

        async with BybitClient() as client:
            while True:
                await asyncio.gather(*[fetch_and_store(client, pair) for pair in pairs])
                logger.debug(f"Synced {len(pairs)} pairs (bybit/futures)")
                await asyncio.sleep(update_interval_sec)
