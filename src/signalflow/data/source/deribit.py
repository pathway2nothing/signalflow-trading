"""Deribit data source - async REST client and loaders for derivatives."""

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
    normalize_deribit_pair,
    to_deribit_instrument,
)

# Deribit interval mapping (resolution parameter).
_DERIBIT_INTERVAL_MAP: dict[str, str] = {
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
    "1d": "1D",
}


@dataclass
@sf_component(name="deribit")
class DeribitClient(RawDataSource):
    """Async client for Deribit REST API (JSON-RPC 2.0).

    Provides methods for fetching OHLCV kline data with retries,
    rate-limit handling, and pagination.

    Returned timestamps are candle **close** times (open + 1 tf, UTC-naive).
    Deribit only offers derivatives (no spot trading).

    Attributes:
        base_url: Deribit API base URL.
        max_retries: Maximum retry attempts.
        timeout_sec: Request timeout in seconds.
        min_delay_sec: Minimum delay between requests.
    """

    base_url: str = "https://www.deribit.com/api/v2"
    max_retries: int = 3
    timeout_sec: int = 30
    min_delay_sec: float = 0.1

    _session: Optional[aiohttp.ClientSession] = field(default=None, init=False)

    async def __aenter__(self) -> "DeribitClient":
        timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def get_pairs(
        self,
        currency: str = "BTC",
        kind: str = "future",
        expired: bool = False,
    ) -> list[str]:
        """Get list of available instruments from Deribit.

        Args:
            currency (str): Base currency ("BTC", "ETH", "USDC", etc.).
            kind (str): Instrument kind ("future", "option", "spot", "future_combo").
            expired (bool): Include expired instruments.

        Returns:
            list[str]: List of instrument names (e.g., ["BTC-PERPETUAL", "ETH-PERPETUAL"]).

        Example:
            ```python
            async with DeribitClient() as client:
                # All BTC futures
                btc = await client.get_pairs(currency="BTC", kind="future")

                # All ETH perpetuals
                eth = await client.get_pairs(currency="ETH")
            ```
        """
        if self._session is None:
            raise RuntimeError("DeribitClient must be used as an async context manager.")

        url = f"{self.base_url}/public/get_instruments"
        params = {
            "currency": currency.upper(),
            "kind": kind,
            "expired": str(expired).lower(),
        }

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
                        raise RuntimeError(f"Deribit API HTTP {resp.status}: {text}")

                    body = await resp.json()

                if "error" in body:
                    err = body["error"]
                    raise RuntimeError(f"Deribit API error {err.get('code')}: {err.get('message')}")

                instruments: list[str] = []
                for inst in body.get("result", []):
                    if inst.get("is_active", True):
                        instruments.append(inst.get("instrument_name", ""))

                return sorted(instruments)

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
        instrument: str,
        timeframe: str = "1m",
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10000,
    ) -> list[dict]:
        """Fetch OHLCV klines from Deribit.

        Returned ``timestamp`` is the candle **close** time (open + 1 tf, UTC-naive).

        Args:
            instrument: Deribit instrument name (e.g., "BTC-PERPETUAL").
            timeframe: Interval in SignalFlow format (1m, 1h, ...).
            start_time: Fetch candles after this time.
            end_time: Fetch candles before this time.
            limit: Max candles per request (Deribit max: 10000).

        Returns:
            List of OHLCV dicts sorted ascending by timestamp.

        Raises:
            RuntimeError: If not in async context or API error.
        """
        if self._session is None:
            raise RuntimeError("DeribitClient must be used as an async context manager.")

        resolution = _DERIBIT_INTERVAL_MAP.get(timeframe)
        if resolution is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        params: dict[str, object] = {
            "instrument_name": instrument,
            "resolution": resolution,
        }
        if start_time is not None:
            params["start_timestamp"] = dt_to_ms_utc(start_time)
        if end_time is not None:
            params["end_timestamp"] = dt_to_ms_utc(end_time)

        url = f"{self.base_url}/public/get_tradingview_chart_data"
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s (inst={instrument})")
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Deribit API HTTP {resp.status}: {text}")

                    body = await resp.json()

                if "error" in body:
                    err = body["error"]
                    raise RuntimeError(f"Deribit API error {err.get('code')}: {err.get('message')}")

                result = body.get("result", {})
                if result.get("status") == "no_data":
                    return []

                ticks = result.get("ticks", [])
                opens = result.get("open", [])
                highs = result.get("high", [])
                lows = result.get("low", [])
                closes = result.get("close", [])
                volumes = result.get("volume", [])

                tf_ms = TIMEFRAME_MS.get(timeframe, 60_000)
                out: list[dict] = []
                for i, ts in enumerate(ticks):
                    out.append(
                        {
                            "timestamp": ms_to_dt_utc_naive(int(ts) + tf_ms),
                            "open": float(opens[i]),
                            "high": float(highs[i]),
                            "low": float(lows[i]),
                            "close": float(closes[i]),
                            "volume": float(volumes[i]),
                            "trades": 0,
                        }
                    )
                return out

            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(f"Request failed, retrying in {wait}s (inst={instrument}): {e}")
                    await asyncio.sleep(wait)
                else:
                    break

        raise last_err or RuntimeError("Unknown error while fetching Deribit klines.")

    async def get_klines_range(
        self,
        instrument: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        *,
        limit: int = 10000,
    ) -> list[dict]:
        """Download all klines for a period with automatic pagination.

        Args:
            instrument: Deribit instrument name.
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
                instrument=instrument,
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

            if len(all_klines) and len(all_klines) % 10000 == 0:
                logger.info(f"{instrument}: loaded {len(all_klines):,} candles...")

            await asyncio.sleep(self.min_delay_sec)

        uniq: dict[datetime, dict] = {}
        for k in all_klines:
            uniq[k["timestamp"]] = k

        out = list(uniq.values())
        out.sort(key=lambda x: x["timestamp"])
        return out


@dataclass
@sf_component(name="deribit/futures")
class DeribitFuturesLoader(RawDataLoader):
    """Downloads and stores Deribit futures/perpetual OHLCV data.

    Pairs are provided in compact format (e.g., "BTCUSD") and
    automatically converted to Deribit instrument names (e.g., "BTC-PERPETUAL").

    Attributes:
        store: Storage backend.
        timeframe: Fixed timeframe for all data.
        currency: Base currency for get_pairs() (BTC, ETH, etc.).
    """

    store: DuckDbSpotStore = field(
        default_factory=lambda: DuckDbSpotStore(db_path=Path("raw_data_deribit_futures.duckdb"))
    )
    timeframe: str = "1m"
    currency: str = "BTC"

    async def get_pairs(self, kind: str = "future") -> list[str]:
        """Get list of available Deribit futures instruments.

        Args:
            kind (str): Instrument kind ("future", "option", etc.).

        Returns:
            list[str]: List of instrument names (e.g., ["BTC-PERPETUAL"]).

        Example:
            ```python
            loader = DeribitFuturesLoader(store=store)
            instruments = await loader.get_pairs()
            # ['BTC-PERPETUAL', 'BTC-27DEC24', ...]
            ```
        """
        async with DeribitClient() as client:
            return await client.get_pairs(currency=self.currency, kind=kind)

    async def download(
        self,
        pairs: list[str],
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        fill_gaps: bool = True,
    ) -> None:
        """Download historical Deribit futures data.

        Args:
            pairs: Trading pairs (e.g., ["BTCUSD"]) or instruments (e.g., ["BTC-PERPETUAL"]).
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
            "12h": 720,
            "1d": 1440,
        }.get(self.timeframe, 1)

        async def download_pair(client: DeribitClient, pair: str) -> None:
            # Convert compact pair to Deribit instrument if needed
            if "-" not in pair:
                instrument = to_deribit_instrument(pair)
            else:
                instrument = pair

            store_pair = normalize_deribit_pair(instrument)
            logger.info(f"Processing {pair} -> {instrument} (deribit/futures) from {start} to {end}")

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
                        instrument=instrument,
                        timeframe=self.timeframe,
                        start_time=range_start,
                        end_time=range_end,
                    )
                    self.store.insert_klines(store_pair, klines)
                except Exception as e:
                    logger.error(f"Error downloading {store_pair}: {e}")

        async with DeribitClient() as client:
            await asyncio.gather(*[download_pair(client, pair) for pair in pairs])

        self.store.close()

    async def sync(
        self,
        pairs: list[str],
        update_interval_sec: int = 60,
    ) -> None:
        """Continuously sync latest Deribit futures data.

        Args:
            pairs: Trading pairs or instruments to sync.
            update_interval_sec: Update interval in seconds.
        """
        logger.info(f"Starting real-time sync (deribit/futures) for {pairs}")
        logger.info(f"Update interval: {update_interval_sec}s (timeframe={self.timeframe})")

        async def fetch_and_store(client: DeribitClient, pair: str) -> None:
            if "-" not in pair:
                instrument = to_deribit_instrument(pair)
            else:
                instrument = pair

            store_pair = normalize_deribit_pair(instrument)
            try:
                klines = await client.get_klines(
                    instrument=instrument,
                    timeframe=self.timeframe,
                )
                # Take last 5 candles
                if len(klines) > 5:
                    klines = klines[-5:]
                self.store.insert_klines(store_pair, klines)
            except Exception as e:
                logger.error(f"Error syncing {store_pair}: {e}")

        async with DeribitClient() as client:
            while True:
                await asyncio.gather(*[fetch_and_store(client, pair) for pair in pairs])
                logger.debug(f"Synced {len(pairs)} pairs (deribit/futures)")
                await asyncio.sleep(update_interval_sec)
