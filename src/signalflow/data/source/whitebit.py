"""WhiteBIT data source - async REST client and loaders for spot trading."""

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
    dt_to_sec_utc,
    sec_to_dt_utc_naive,
    ensure_utc_naive,
    normalize_whitebit_pair,
    to_whitebit_symbol,
)

# WhiteBIT interval mapping.
_WHITEBIT_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
}

# Timeframe in seconds for shifting to close time
_TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


@dataclass
@sf_component(name="whitebit")
class WhitebitClient(RawDataSource):
    """Async client for WhiteBIT REST API.

    Provides methods for fetching OHLCV kline data with retries
    and rate-limit handling.

    Returned timestamps are candle **close** times (UTC-naive).
    WhiteBIT uses SECONDS for timestamps (not milliseconds).

    Attributes:
        base_url: WhiteBIT API base URL.
        max_retries: Maximum retry attempts.
        timeout_sec: Request timeout in seconds.
        min_delay_sec: Minimum delay between requests.
    """

    base_url: str = "https://whitebit.com"
    max_retries: int = 3
    timeout_sec: int = 30
    min_delay_sec: float = 0.05  # 1000 req/10 sec limit

    _session: Optional[aiohttp.ClientSession] = field(default=None, init=False)

    async def __aenter__(self) -> "WhitebitClient":
        timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def get_pairs(self, quote: Optional[str] = None) -> list[str]:
        """Get list of available trading pairs from WhiteBIT.

        Args:
            quote: Optional quote currency filter (e.g., "USDT").

        Returns:
            list[str]: List of market symbols (e.g., ["BTC_USDT", "ETH_USDT"]).

        Example:
            ```python
            async with WhitebitClient() as client:
                pairs = await client.get_pairs(quote="USDT")
                # ['BTC_USDT', 'ETH_USDT', ...]
            ```
        """
        if self._session is None:
            raise RuntimeError("WhitebitClient must be used as an async context manager.")

        url = f"{self.base_url}/api/v4/public/markets"

        for attempt in range(self.max_retries):
            try:
                async with self._session.get(url) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 10))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"WhiteBIT API HTTP {resp.status}: {text}")

                    body = await resp.json()

                pairs: list[str] = []
                for market in body:
                    name = market.get("name", "")
                    if not name:
                        continue
                    # Filter by quote currency if specified
                    if quote:
                        if name.endswith(f"_{quote.upper()}"):
                            pairs.append(name)
                    else:
                        pairs.append(name)

                return sorted(pairs)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(f"Request failed, retrying in {wait}s: {e}")
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(f"Failed to get pairs: {e}") from e

        return []

    async def get_klines(
        self,
        market: str,
        timeframe: str = "1m",
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1440,
    ) -> list[dict]:
        """Fetch OHLCV klines from WhiteBIT.

        Returned ``timestamp`` is the candle **close** time (UTC-naive).
        WhiteBIT returns open time, so we add one timeframe to get close time.

        Args:
            market: Market symbol (e.g., "BTC_USDT", "ETH_USDT").
            timeframe: Interval in SignalFlow format (1m, 1h, ...).
            start_time: Fetch candles after this time.
            end_time: Fetch candles before this time.
            limit: Max candles per request (WhiteBIT max: 1440).

        Returns:
            List of OHLCV dicts sorted ascending by timestamp.

        Raises:
            RuntimeError: If not in async context or API error.
        """
        if self._session is None:
            raise RuntimeError("WhitebitClient must be used as an async context manager.")

        interval = _WHITEBIT_INTERVAL_MAP.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        tf_seconds = _TIMEFRAME_SECONDS.get(timeframe, 60)

        # Build URL with query params
        url = f"{self.base_url}/api/v1/public/kline"
        params: dict[str, object] = {
            "market": market.upper(),
            "interval": interval,
            "limit": min(limit, 1440),
        }
        if start_time is not None:
            params["start"] = dt_to_sec_utc(start_time)
        if end_time is not None:
            params["end"] = dt_to_sec_utc(end_time)

        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 10))
                        logger.warning(f"Rate limited, waiting {retry_after}s (market={market})")
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"WhiteBIT API HTTP {resp.status}: {text}")

                    body = await resp.json()

                # Check for API errors
                if isinstance(body, dict) and ("error" in body or "message" in body):
                    err_msg = body.get("error") or body.get("message", "Unknown error")
                    raise RuntimeError(f"WhiteBIT API error: {err_msg}")

                # Check for success response format
                if isinstance(body, dict) and not body.get("success", True):
                    raise RuntimeError(f"WhiteBIT API error: {body}")

                # Extract result if wrapped
                result = body.get("result", body) if isinstance(body, dict) else body

                # Response format: [[timestamp, open, close, high, low, volume_stock, volume_money], ...]
                # Note: WhiteBIT order is open, CLOSE, high, low (different from most exchanges)
                out: list[dict] = []
                for candle in result:
                    if not isinstance(candle, list) or len(candle) < 6:
                        continue
                    # timestamp is open time, add tf_seconds to get close time
                    open_time_sec = int(candle[0])
                    close_time_sec = open_time_sec + tf_seconds
                    out.append(
                        {
                            "timestamp": sec_to_dt_utc_naive(close_time_sec),
                            "open": float(candle[1]),
                            "high": float(candle[3]),
                            "low": float(candle[4]),
                            "close": float(candle[2]),  # close is at index 2!
                            "volume": float(candle[5]),
                            "trades": 0,  # Not available in WhiteBIT API
                        }
                    )

                # Sort ascending by timestamp
                out.sort(key=lambda x: x["timestamp"])
                return out

            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(f"Request failed, retrying in {wait}s (market={market}): {e}")
                    await asyncio.sleep(wait)
                else:
                    break

        raise last_err or RuntimeError("Unknown error while fetching WhiteBIT klines.")

    async def get_klines_range(
        self,
        market: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        *,
        limit: int = 1440,
    ) -> list[dict]:
        """Download all klines for a period with automatic pagination.

        Args:
            market: Market symbol.
            timeframe: Interval (must be in ``TIMEFRAME_MS``).
            start_time: Range start (inclusive).
            end_time: Range end (inclusive).
            limit: Candles per request (max 1440).

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

        max_loops = 1000
        loops = 0

        while current_start < end_time:
            loops += 1
            if loops > max_loops:
                logger.warning(f"WhiteBIT: reached pagination limit for {market}")
                break

            req_end = min(current_start + window, end_time)

            klines = await self.get_klines(
                market=market,
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

            if len(all_klines) and len(all_klines) % 5000 == 0:
                logger.info(f"{market}: loaded {len(all_klines):,} candles...")

            await asyncio.sleep(self.min_delay_sec)

        uniq: dict[datetime, dict] = {}
        for k in all_klines:
            uniq[k["timestamp"]] = k

        out = list(uniq.values())
        out.sort(key=lambda x: x["timestamp"])
        return out


@dataclass
@sf_component(name="whitebit/spot")
class WhitebitSpotLoader(RawDataLoader):
    """Downloads and stores WhiteBIT spot OHLCV data.

    Pairs are provided in compact format (e.g., "BTCUSDT") and
    automatically converted to WhiteBIT symbols (e.g., "BTC_USDT").

    Attributes:
        store: Storage backend.
        timeframe: Fixed timeframe for all data.
    """

    store: DuckDbSpotStore = field(
        default_factory=lambda: DuckDbSpotStore(db_path=Path("raw_data_whitebit_spot.duckdb"))
    )
    timeframe: str = "1m"

    async def get_pairs(self, quote: Optional[str] = "USDT") -> list[str]:
        """Get list of available WhiteBIT spot pairs.

        Args:
            quote: Optional quote currency filter (e.g., "USDT").

        Returns:
            list[str]: List of market symbols (e.g., ["BTC_USDT", "ETH_USDT"]).

        Example:
            ```python
            loader = WhitebitSpotLoader(store=store)
            pairs = await loader.get_pairs(quote="USDT")
            # ['BTC_USDT', 'ETH_USDT', ...]
            ```
        """
        async with WhitebitClient() as client:
            return await client.get_pairs(quote=quote)

    async def download(
        self,
        pairs: list[str],
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        fill_gaps: bool = True,
    ) -> None:
        """Download historical WhiteBIT spot data.

        Args:
            pairs: Trading pairs (e.g., ["BTCUSDT", "BTC_USDT"]).
            days: Number of days back from *end*. Default: 30.
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
            start = end - timedelta(days=days if days else 30)
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

        async def download_pair(client: WhitebitClient, pair: str) -> None:
            # Convert compact pair to WhiteBIT symbol if needed
            market = to_whitebit_symbol(pair)
            store_pair = normalize_whitebit_pair(market)

            logger.info(f"Processing {pair} -> {market} (whitebit/spot) from {start} to {end}")

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
                        market=market,
                        timeframe=self.timeframe,
                        start_time=range_start,
                        end_time=range_end,
                    )
                    self.store.insert_klines(store_pair, klines)
                except Exception as e:
                    logger.error(f"Error downloading {store_pair}: {e}")

        async with WhitebitClient() as client:
            await asyncio.gather(*[download_pair(client, pair) for pair in pairs])

        self.store.close()

    async def sync(
        self,
        pairs: list[str],
        update_interval_sec: int = 60,
    ) -> None:
        """Continuously sync latest WhiteBIT spot data.

        Args:
            pairs: Trading pairs to sync.
            update_interval_sec: Update interval in seconds.
        """
        logger.info(f"Starting real-time sync (whitebit/spot) for {pairs}")
        logger.info(f"Update interval: {update_interval_sec}s (timeframe={self.timeframe})")

        async def fetch_and_store(client: WhitebitClient, pair: str) -> None:
            market = to_whitebit_symbol(pair)
            store_pair = normalize_whitebit_pair(market)
            try:
                klines = await client.get_klines(
                    market=market,
                    timeframe=self.timeframe,
                    limit=5,
                )
                self.store.insert_klines(store_pair, klines)
            except Exception as e:
                logger.error(f"Error syncing {store_pair}: {e}")

        async with WhitebitClient() as client:
            while True:
                await asyncio.gather(*[fetch_and_store(client, pair) for pair in pairs])
                logger.debug(f"Synced {len(pairs)} pairs (whitebit/spot)")
                await asyncio.sleep(update_interval_sec)
