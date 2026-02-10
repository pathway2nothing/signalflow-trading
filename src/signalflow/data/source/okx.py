"""OKX data source - async REST client and loaders for spot & futures."""

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

# OKX-specific bar interval mapping.
_OKX_BAR_MAP: dict[str, str] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "1D",
}

_QUOTE_CURRENCIES = ("USDT", "USDC", "USDK", "BTC", "ETH", "DAI")


def _to_okx_inst_id(pair: str, suffix: str = "") -> str:
    """Convert a compact pair to an OKX instrument ID.

    Examples::

        _to_okx_inst_id("BTCUSDT")           -> "BTC-USDT"
        _to_okx_inst_id("ETHBTC")            -> "ETH-BTC"
        _to_okx_inst_id("BTCUSDT", "-SWAP")  -> "BTC-USDT-SWAP"
    """
    for quote in _QUOTE_CURRENCIES:
        if pair.upper().endswith(quote):
            base = pair[: -len(quote)]
            return f"{base}-{quote}{suffix}"
    return f"{pair}{suffix}"


@dataclass
@sf_component(name="okx")
class OkxClient(RawDataSource):
    """Async client for OKX v5 REST API.

    Provides methods for fetching OHLCV kline data with retries,
    rate-limit handling, and backward pagination.

    Returned timestamps are candle **close** times (open + 1 tf).
    The ``/api/v5/market/candles`` endpoint serves recent data while
    ``/api/v5/market/history-candles`` covers older periods.

    Attributes:
        base_url: OKX API base URL.
        max_retries: Maximum retry attempts.
        timeout_sec: Request timeout in seconds.
        min_delay_sec: Minimum delay between requests (higher default
            because OKX allows only 100 candles per request).
    """

    base_url: str = "https://www.okx.com"
    max_retries: int = 3
    timeout_sec: int = 30
    min_delay_sec: float = 0.1

    _session: Optional[aiohttp.ClientSession] = field(default=None, init=False)

    async def __aenter__(self) -> "OkxClient":
        timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def get_klines(
        self,
        inst_id: str,
        timeframe: str = "1m",
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        use_history: bool = False,
    ) -> list[dict]:
        """Fetch OHLCV klines from OKX.

        Returned ``timestamp`` is the candle **close** time (open + 1 tf, UTC-naive).

        Args:
            inst_id: OKX instrument ID (e.g. ``"BTC-USDT"``, ``"BTC-USDT-SWAP"``).
            timeframe: Interval in SignalFlow format (1m, 1h, …).
            start_time: Fetch candles **after** (newer than) this time.
            end_time: Fetch candles **before** (older than) this time.
            limit: Max candles per request (OKX max: 100).
            use_history: Use ``/history-candles`` endpoint for older data.

        Returns:
            List of OHLCV dicts sorted ascending by timestamp.

        Raises:
            RuntimeError: If not in async context or API error.
        """
        if self._session is None:
            raise RuntimeError("OkxClient must be used as an async context manager.")

        bar = _OKX_BAR_MAP.get(timeframe)
        if bar is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        params: dict[str, object] = {
            "instId": inst_id,
            "bar": bar,
            "limit": str(int(min(limit, 100))),
        }
        # OKX pagination: `after` returns records NEWER than ts, `before` returns OLDER.
        if start_time is not None:
            params["after"] = str(dt_to_ms_utc(start_time))
        if end_time is not None:
            params["before"] = str(dt_to_ms_utc(end_time))

        endpoint = "/api/v5/market/history-candles" if use_history else "/api/v5/market/candles"
        url = f"{self.base_url}{endpoint}"
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s (inst={inst_id})")
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"OKX API HTTP {resp.status}: {text}")

                    body = await resp.json()

                if body.get("code") != "0":
                    raise RuntimeError(f"OKX API error {body.get('code')}: {body.get('msg')}")

                rows = body.get("data", [])

                tf_ms = TIMEFRAME_MS.get(timeframe, 60_000)
                out: list[dict] = []
                for row in reversed(rows):  # OKX returns descending - reverse
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
                    logger.warning(f"Request failed, retrying in {wait}s (inst={inst_id}): {e}")
                    await asyncio.sleep(wait)
                else:
                    break

        raise last_err or RuntimeError("Unknown error while fetching OKX klines.")

    async def get_klines_range(
        self,
        inst_id: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        *,
        limit: int = 100,
    ) -> list[dict]:
        """Download all klines for a period with backward pagination.

        Uses the ``history-candles`` endpoint for bulk downloads.
        Paginates backward from *end_time* using the ``before`` parameter.

        Args:
            inst_id: OKX instrument ID.
            timeframe: Interval (must be in ``TIMEFRAME_MS``).
            start_time: Range start (inclusive).
            end_time: Range end (inclusive).
            limit: Candles per request (max 100).

        Returns:
            Deduplicated, ascending OHLCV dicts.
        """
        if timeframe not in TIMEFRAME_MS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        start_time = ensure_utc_naive(start_time)
        end_time = ensure_utc_naive(end_time)

        if start_time >= end_time:
            return []

        start_ms = dt_to_ms_utc(start_time)
        cursor_ms = dt_to_ms_utc(end_time) + 1  # exclusive upper bound

        all_klines: list[dict] = []
        max_loops = 2_000_000
        loops = 0

        while cursor_ms > start_ms:
            loops += 1
            if loops > max_loops:
                raise RuntimeError("Pagination guard triggered (too many loops).")

            klines = await self.get_klines(
                inst_id=inst_id,
                timeframe=timeframe,
                end_time=ms_to_dt_utc_naive(cursor_ms),
                limit=limit,
                use_history=True,
            )

            if not klines:
                break

            for k in klines:
                ts = k["timestamp"]
                if start_time <= ts <= end_time:
                    all_klines.append(k)

            oldest_ts = klines[0]["timestamp"]  # close time (open + tf)
            cursor_ms = dt_to_ms_utc(oldest_ts) - TIMEFRAME_MS[timeframe]

            if len(all_klines) and len(all_klines) % 5000 == 0:
                logger.info(f"{inst_id}: loaded {len(all_klines):,} candles...")

            await asyncio.sleep(self.min_delay_sec)

        uniq: dict[datetime, dict] = {}
        for k in all_klines:
            uniq[k["timestamp"]] = k

        out = list(uniq.values())
        out.sort(key=lambda x: x["timestamp"])
        return out


@dataclass
@sf_component(name="okx/spot")
class OkxSpotLoader(RawDataLoader):
    """Downloads and stores OKX spot OHLCV data.

    Pairs are provided in compact format (e.g. ``"BTCUSDT"``) and
    automatically converted to OKX instrument IDs (``"BTC-USDT"``).

    Attributes:
        store: Storage backend.
        timeframe: Fixed timeframe for all data.
    """

    store: DuckDbSpotStore = field(default_factory=lambda: DuckDbSpotStore(db_path=Path("raw_data_okx_spot.duckdb")))
    timeframe: str = "1m"

    async def download(
        self,
        pairs: list[str],
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        fill_gaps: bool = True,
    ) -> None:
        """Download historical OKX spot data.

        Args:
            pairs: Trading pairs (e.g. ``["BTCUSDT"]``).
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

        async def download_pair(client: OkxClient, pair: str) -> None:
            inst_id = _to_okx_inst_id(pair)
            logger.info(f"Processing {pair} -> {inst_id} (okx/spot) from {start} to {end}")
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
                        inst_id=inst_id,
                        timeframe=self.timeframe,
                        start_time=range_start,
                        end_time=range_end,
                    )
                    self.store.insert_klines(pair, klines)
                except Exception as e:
                    logger.error(f"Error downloading {pair}: {e}")

        async with OkxClient() as client:
            await asyncio.gather(*[download_pair(client, pair) for pair in pairs])

        self.store.close()

    async def sync(
        self,
        pairs: list[str],
        update_interval_sec: int = 60,
    ) -> None:
        """Continuously sync latest OKX spot data.

        Args:
            pairs: Trading pairs to sync.
            update_interval_sec: Update interval in seconds.
        """
        logger.info(f"Starting real-time sync (okx/spot) for {pairs}")
        logger.info(f"Update interval: {update_interval_sec}s (timeframe={self.timeframe})")

        async def fetch_and_store(client: OkxClient, pair: str) -> None:
            inst_id = _to_okx_inst_id(pair)
            try:
                klines = await client.get_klines(
                    inst_id=inst_id,
                    timeframe=self.timeframe,
                    limit=5,
                )
                self.store.insert_klines(pair, klines)
            except Exception as e:
                logger.error(f"Error syncing {pair}: {e}")

        async with OkxClient() as client:
            while True:
                await asyncio.gather(*[fetch_and_store(client, pair) for pair in pairs])
                logger.debug(f"Synced {len(pairs)} pairs (okx/spot)")
                await asyncio.sleep(update_interval_sec)


@dataclass
@sf_component(name="okx/futures")
class OkxFuturesLoader(RawDataLoader):
    """Downloads and stores OKX futures/swap OHLCV data.

    Pairs are converted to OKX instrument IDs with a configurable suffix.
    Default suffix ``"-SWAP"`` targets perpetual contracts; use e.g.
    ``"-240329"`` for dated delivery contracts.

    Attributes:
        store: Storage backend.
        timeframe: Fixed timeframe for all data.
        inst_suffix: Suffix appended after the quote currency
            (e.g. ``"-SWAP"``, ``"-240329"``).
    """

    store: DuckDbSpotStore = field(default_factory=lambda: DuckDbSpotStore(db_path=Path("raw_data_okx_futures.duckdb")))
    timeframe: str = "1m"
    inst_suffix: str = "-SWAP"

    async def download(
        self,
        pairs: list[str],
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        fill_gaps: bool = True,
    ) -> None:
        """Download historical OKX futures data.

        Args:
            pairs: Trading pairs (e.g. ``["BTCUSDT"]``).
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

        async def download_pair(client: OkxClient, pair: str) -> None:
            inst_id = _to_okx_inst_id(pair, suffix=self.inst_suffix)
            logger.info(f"Processing {pair} -> {inst_id} (okx/futures) from {start} to {end}")
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
                        inst_id=inst_id,
                        timeframe=self.timeframe,
                        start_time=range_start,
                        end_time=range_end,
                    )
                    self.store.insert_klines(pair, klines)
                except Exception as e:
                    logger.error(f"Error downloading {pair}: {e}")

        async with OkxClient() as client:
            await asyncio.gather(*[download_pair(client, pair) for pair in pairs])

        self.store.close()

    async def sync(
        self,
        pairs: list[str],
        update_interval_sec: int = 60,
    ) -> None:
        """Continuously sync latest OKX futures data.

        Args:
            pairs: Trading pairs to sync.
            update_interval_sec: Update interval in seconds.
        """
        logger.info(f"Starting real-time sync (okx/futures) for {pairs}")
        logger.info(f"Update interval: {update_interval_sec}s (timeframe={self.timeframe})")

        async def fetch_and_store(client: OkxClient, pair: str) -> None:
            inst_id = _to_okx_inst_id(pair, suffix=self.inst_suffix)
            try:
                klines = await client.get_klines(
                    inst_id=inst_id,
                    timeframe=self.timeframe,
                    limit=5,
                )
                self.store.insert_klines(pair, klines)
            except Exception as e:
                logger.error(f"Error syncing {pair}: {e}")

        async with OkxClient() as client:
            while True:
                await asyncio.gather(*[fetch_and_store(client, pair) for pair in pairs])
                logger.debug(f"Synced {len(pairs)} pairs (okx/futures)")
                await asyncio.sleep(update_interval_sec)
