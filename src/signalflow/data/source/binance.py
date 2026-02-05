# IMPORTANT
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiohttp
from loguru import logger

from signalflow.data.raw_store import DuckDbSpotStore
from signalflow.core import sf_component
from signalflow.data.source.base import RawDataSource, RawDataLoader

_TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def _dt_to_ms_utc(dt: datetime) -> int:
    """Convert datetime to UNIX milliseconds in UTC.

    Accepts naive (assumed UTC) or aware (converted to UTC) datetimes.

    Args:
        dt (datetime): Input datetime.

    Returns:
        int: UNIX timestamp in milliseconds (UTC).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_dt_utc_naive(ms: int) -> datetime:
    """Convert UNIX milliseconds to UTC-naive datetime.

    Args:
        ms (int): UNIX timestamp in milliseconds.

    Returns:
        datetime: UTC datetime without timezone info.
    """
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None)


def _ensure_utc_naive(dt: datetime) -> datetime:
    """Normalize to UTC-naive datetime.

    Args:
        dt (datetime): Input datetime (naive or aware).

    Returns:
        datetime: UTC-naive datetime.
    """
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


@dataclass
@sf_component(name="binance")
class BinanceClient(RawDataSource):
    """Async client for Binance REST API.

    Provides async methods for fetching OHLCV candlestick data with automatic
    retries, rate limit handling, and pagination.

    IMPORTANT: Returned timestamps are candle CLOSE times (Binance k[6]), UTC-naive.

    Attributes:
        base_url (str): Binance API base URL. Default: "https://api.binance.com".
        max_retries (int): Maximum retry attempts. Default: 3.
        timeout_sec (int): Request timeout in seconds. Default: 30.
        min_delay_sec (float): Minimum delay between requests. Default: 0.05.
    """

    base_url: str = "https://api.binance.com"
    max_retries: int = 3
    timeout_sec: int = 30
    min_delay_sec: float = 0.05

    _session: Optional[aiohttp.ClientSession] = field(default=None, init=False)

    async def __aenter__(self) -> "BinanceClient":
        """Enter async context - creates session."""
        timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *args) -> None:
        """Exit async context - closes session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def get_klines(
        self,
        pair: str,
        timeframe: str = "1m",
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Fetch OHLCV klines from Binance.

        IMPORTANT: Returned "timestamp" is CANDLE CLOSE TIME (UTC-naive).

        Args:
            pair (str): Trading pair (e.g., "BTCUSDT").
            timeframe (str): Interval (1m, 5m, 1h, 1d, etc.). Default: "1m".
            start_time (datetime | None): Range start (naive=UTC or aware).
            end_time (datetime | None): Range end (naive=UTC or aware).
            limit (int): Max candles (max 1000). Default: 1000.

        Returns:
            list[dict]: OHLCV dicts with keys: timestamp, open, high, low,
                close, volume, trades.

        Raises:
            RuntimeError: If not in async context or API error.
        """
        if self._session is None:
            raise RuntimeError("BinanceClient must be used as an async context manager.")

        params: dict[str, object] = {"symbol": pair, "interval": timeframe, "limit": int(limit)}
        if start_time is not None:
            params["startTime"] = _dt_to_ms_utc(start_time)
        if end_time is not None:
            params["endTime"] = _dt_to_ms_utc(end_time)

        url = f"{self.base_url}/api/v3/klines"
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s (pair={pair}, tf={timeframe})")
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Binance API error {resp.status}: {text}")

                    data = await resp.json()

                out: list[dict] = []
                for k in data:
                    close_ms = int(k[6])
                    out.append(
                        {
                            "timestamp": _ms_to_dt_utc_naive(close_ms),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[7]),
                            "trades": int(k[8]),
                        }
                    )

                return out

            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(f"Request failed, retrying in {wait}s (pair={pair}, tf={timeframe}): {e}")
                    await asyncio.sleep(wait)
                else:
                    break

        raise last_err or RuntimeError("Unknown error while fetching klines.")

    async def get_klines_range(
        self,
        pair: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        *,
        limit: int = 1000,
    ) -> list[dict]:
        """Download all klines for period with automatic pagination.

        Semantics:
            - Range by CANDLE CLOSE TIME: [start_time, end_time] inclusive
            - Returns UTC-naive timestamps
            - Automatic deduplication

        Pagination strategy:
            - Request windows of size limit * timeframe
            - Advance based on last returned close time + 1ms
            - Additional dedup at end for safety

        Args:
            pair (str): Trading pair.
            timeframe (str): Interval (must be in _TIMEFRAME_MS).
            start_time (datetime): Range start (inclusive).
            end_time (datetime): Range end (inclusive).
            limit (int): Candles per request. Default: 1000.

        Returns:
            list[dict]: Deduplicated, sorted OHLCV dicts.

        Raises:
            ValueError: If timeframe unsupported.
            RuntimeError: If pagination exceeds safety limit (2M loops).
        """
        if timeframe not in _TIMEFRAME_MS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        start_time = _ensure_utc_naive(start_time)
        end_time = _ensure_utc_naive(end_time)

        if start_time >= end_time:
            return []

        tf_ms = _TIMEFRAME_MS[timeframe]
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

            last_close = klines[-1]["timestamp"]
            next_start = last_close + timedelta(milliseconds=1)

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
@sf_component(name="binance/spot")
class BinanceSpotLoader(RawDataLoader):
    """Downloads and stores Binance spot OHLCV data for fixed timeframe.

    Combines BinanceClient (source) and DuckDbSpotStore (storage) to provide
    complete data pipeline with gap filling and incremental updates.

    Attributes:
        store (DuckDbSpotStore): Storage backend. Default: raw_data.duckdb.
        timeframe (str): Fixed timeframe for all data. Default: "1m".
    """

    store: DuckDbSpotStore = field(default_factory=lambda: DuckDbSpotStore(db_path=Path("raw_data.duckdb")))
    timeframe: str = "1m"

    async def download(
        self,
        pairs: list[str],
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        fill_gaps: bool = True,
    ) -> None:
        """Download historical data with intelligent range detection.

        Automatically determines what to download:
            - If no existing data: download full range
            - If data exists: download before/after existing range
            - If fill_gaps=True: detect and fill gaps in existing range

        Args:
            pairs (list[str]): Trading pairs to download.
            days (int | None): Number of days back from end. Default: 7.
            start (datetime | None): Range start (overrides days).
            end (datetime | None): Range end. Default: now.
            fill_gaps (bool): Detect and fill gaps. Default: True.

        Note:
            Runs async download for all pairs concurrently.
            Logs progress for large downloads.
            Errors logged but don't stop other pairs.
        """

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if end is None:
            end = now
        else:
            end = _ensure_utc_naive(end)

        if start is None:
            start = end - timedelta(days=days if days else 7)
        else:
            start = _ensure_utc_naive(start)

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

        async def download_pair(client: BinanceClient, pair: str) -> None:
            logger.info(f"Processing {pair} from {start} to {end}")

            db_min, db_max = self.store.get_time_bounds(pair)
            ranges_to_download: list[tuple[datetime, datetime]] = []

            if db_min is None:
                ranges_to_download.append((start, end))
            else:
                if start < db_min:
                    ranges_to_download.append((start, db_min - timedelta(minutes=tf_minutes)))
                if end > db_max:
                    ranges_to_download.append((db_max + timedelta(minutes=tf_minutes), end))

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
                        timeframe=self.timeframe,
                        start_time=range_start,
                        end_time=range_end,
                    )
                    self.store.insert_klines(pair, klines)
                except Exception as e:
                    logger.error(f"Error downloading {pair}: {e}")

        async with BinanceClient() as client:
            await asyncio.gather(*[download_pair(client, pair) for pair in pairs])

        self.store.close()

    async def sync(
        self,
        pairs: list[str],
        update_interval_sec: int = 60,
    ) -> None:
        """Real-time sync - continuously update with latest data.

        Runs indefinitely, fetching latest candles at specified interval.
        Useful for live trading or monitoring.

        Args:
            pairs (list[str]): Trading pairs to sync.
            update_interval_sec (int): Update interval in seconds. Default: 60.

        Note:
            Runs forever - use Ctrl+C to stop or run in background task.
            Fetches last 5 candles per update (ensures no gaps).
            Errors logged but sync continues.
        """

        logger.info(f"Starting real-time sync for {pairs}")
        logger.info(f"Update interval: {update_interval_sec}s (timeframe={self.timeframe})")

        async def fetch_and_store(client: BinanceClient, pair: str) -> None:
            try:
                klines = await client.get_klines(pair=pair, timeframe=self.timeframe, limit=5)
                self.store.insert_klines(pair, klines)
            except Exception as e:
                logger.error(f"Error syncing {pair}: {e}")

        async with BinanceClient() as client:
            while True:
                await asyncio.gather(*[fetch_and_store(client, pair) for pair in pairs])
                logger.debug(f"Synced {len(pairs)} pairs")
                await asyncio.sleep(update_interval_sec)
