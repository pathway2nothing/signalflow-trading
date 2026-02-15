"""Kraken data source - async REST client and loaders for spot & futures."""

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
    sec_to_dt_utc_naive,
    dt_to_sec_utc,
    ensure_utc_naive,
    normalize_kraken_spot_pair,
    to_kraken_spot_symbol,
    normalize_kraken_futures_pair,
    to_kraken_futures_symbol,
)

# Kraken spot interval mapping (minutes).
# Note: Kraken spot has LIMITED interval support.
_KRAKEN_SPOT_INTERVAL_MAP: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    # Not supported: 2h, 3m, 6h, 8h, 12h
}

# Kraken futures interval mapping.
_KRAKEN_FUTURES_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "12h": "12h",
    "1d": "1d",
    "1w": "1w",
}


@dataclass
@sf_component(name="kraken")
class KrakenClient(RawDataSource):
    """Async client for Kraken REST APIs (spot and futures).

    Kraken has separate APIs for spot and futures trading.
    IMPORTANT: Kraken uses timestamps in SECONDS, not milliseconds.

    Returned timestamps are candle **close** times (UTC-naive).

    Attributes:
        spot_url: Kraken Spot API base URL.
        futures_url: Kraken Futures API base URL.
        max_retries: Maximum retry attempts.
        timeout_sec: Request timeout in seconds.
        min_delay_sec: Minimum delay between requests (Kraken has strict rate limits).
    """

    spot_url: str = "https://api.kraken.com/0/public"
    futures_url: str = "https://futures.kraken.com/derivatives/api/v4"
    max_retries: int = 3
    timeout_sec: int = 30
    min_delay_sec: float = 0.2  # Kraken has stricter rate limits

    _session: Optional[aiohttp.ClientSession] = field(default=None, init=False)

    async def __aenter__(self) -> "KrakenClient":
        timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # -------------------------------------------------------------------------
    # Spot API methods
    # -------------------------------------------------------------------------

    async def get_spot_pairs(self, quote: str | None = None) -> list[str]:
        """Get list of available Kraken spot trading pairs.

        Args:
            quote (str | None): Filter by quote currency (e.g., "USD", "EUR").
                If None, returns all pairs.

        Returns:
            list[str]: List of pair names in Kraken format (e.g., ["XXBTZUSD", "XETHZUSD"]).

        Example:
            ```python
            async with KrakenClient() as client:
                # All USD pairs
                usd_pairs = await client.get_spot_pairs(quote="USD")
            ```
        """
        if self._session is None:
            raise RuntimeError("KrakenClient must be used as an async context manager.")

        url = f"{self.spot_url}/AssetPairs"

        for attempt in range(self.max_retries):
            try:
                async with self._session.get(url) as resp:
                    if resp.status == 429:
                        logger.warning("Rate limited, waiting 60s")
                        await asyncio.sleep(60)
                        continue

                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Kraken API HTTP {resp.status}: {text}")

                    body = await resp.json()

                if body.get("error"):
                    raise RuntimeError(f"Kraken API error: {body['error']}")

                pairs: list[str] = []
                for name, info in body.get("result", {}).items():
                    if info.get("status") == "online":
                        pair_quote = info.get("quote", "")
                        if quote is None or quote.upper() in pair_quote.upper():
                            pairs.append(name)

                return sorted(pairs)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(f"Request failed, retrying in {wait}s: {e}")
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(f"Failed to get spot pairs: {e}") from e

        return []

    async def get_spot_klines(
        self,
        pair: str,
        timeframe: str = "1m",
        *,
        since: Optional[datetime] = None,
        limit: int = 720,
    ) -> list[dict]:
        """Fetch OHLCV klines from Kraken Spot API.

        Kraken spot returns max 720 candles, with timestamps in SECONDS.
        Returned ``timestamp`` is the candle **close** time (UTC-naive).

        Args:
            pair: Kraken pair name (e.g., "XXBTZUSD").
            timeframe: Interval in SignalFlow format (1m, 1h, ...).
            since: Fetch candles since this time.
            limit: Max candles (Kraken max: 720).

        Returns:
            List of OHLCV dicts sorted ascending by timestamp.

        Raises:
            RuntimeError: If not in async context or API error.
        """
        if self._session is None:
            raise RuntimeError("KrakenClient must be used as an async context manager.")

        interval = _KRAKEN_SPOT_INTERVAL_MAP.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe for Kraken spot: {timeframe}")

        params: dict[str, object] = {
            "pair": pair,
            "interval": interval,
        }
        if since is not None:
            params["since"] = dt_to_sec_utc(since)

        url = f"{self.spot_url}/OHLC"
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 429:
                        logger.warning(f"Rate limited, waiting 60s (pair={pair})")
                        await asyncio.sleep(60)
                        continue

                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Kraken API HTTP {resp.status}: {text}")

                    body = await resp.json()

                if body.get("error"):
                    raise RuntimeError(f"Kraken API error: {body['error']}")

                result = body.get("result", {})
                # Result contains pair data and "last" timestamp
                pair_data = None
                for key, val in result.items():
                    if key != "last" and isinstance(val, list):
                        pair_data = val
                        break

                if not pair_data:
                    return []

                # Response: [time, open, high, low, close, vwap, volume, count]
                # time is in SECONDS (open time)
                tf_sec = interval * 60
                out: list[dict] = []
                for row in pair_data:
                    open_time_sec = int(row[0])
                    close_time_sec = open_time_sec + tf_sec
                    out.append(
                        {
                            "timestamp": sec_to_dt_utc_naive(close_time_sec),
                            "open": float(row[1]),
                            "high": float(row[2]),
                            "low": float(row[3]),
                            "close": float(row[4]),
                            "volume": float(row[6]),
                            "trades": int(row[7]),
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

        raise last_err or RuntimeError("Unknown error while fetching Kraken spot klines.")

    async def get_spot_klines_range(
        self,
        pair: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        *,
        limit: int = 720,
    ) -> list[dict]:
        """Download all spot klines for a period with pagination.

        Note: Kraken spot only returns 720 candles max per request.
        Cannot fetch data older than what's available.

        Args:
            pair: Kraken pair name.
            timeframe: Interval (must be in ``_KRAKEN_SPOT_INTERVAL_MAP``).
            start_time: Range start (inclusive).
            end_time: Range end (inclusive).
            limit: Candles per request.

        Returns:
            Deduplicated, ascending OHLCV dicts.
        """
        if timeframe not in _KRAKEN_SPOT_INTERVAL_MAP:
            raise ValueError(f"Unsupported timeframe for Kraken spot: {timeframe}")

        start_time = ensure_utc_naive(start_time)
        end_time = ensure_utc_naive(end_time)

        if start_time >= end_time:
            return []

        interval_min = _KRAKEN_SPOT_INTERVAL_MAP[timeframe]
        tf_sec = interval_min * 60
        window = timedelta(seconds=tf_sec * limit)

        all_klines: list[dict] = []
        current_start = start_time

        max_loops = 2_000_000
        loops = 0

        while current_start < end_time:
            loops += 1
            if loops > max_loops:
                raise RuntimeError("Pagination guard triggered (too many loops).")

            klines = await self.get_spot_klines(
                pair=pair,
                timeframe=timeframe,
                since=current_start,
                limit=limit,
            )

            if not klines:
                current_start = current_start + window
                await asyncio.sleep(self.min_delay_sec)
                continue

            for k in klines:
                ts = k["timestamp"]
                if start_time <= ts <= end_time:
                    all_klines.append(k)

            last_ts = klines[-1]["timestamp"]
            next_start = last_ts

            if next_start <= current_start:
                current_start = current_start + timedelta(seconds=1)
            else:
                current_start = next_start

            if len(all_klines) and len(all_klines) % 5000 == 0:
                logger.info(f"{pair}: loaded {len(all_klines):,} candles...")

            await asyncio.sleep(self.min_delay_sec)

        uniq: dict[datetime, dict] = {}
        for k in all_klines:
            uniq[k["timestamp"]] = k

        out = list(uniq.values())
        out.sort(key=lambda x: x["timestamp"])
        return out

    # -------------------------------------------------------------------------
    # Futures API methods
    # -------------------------------------------------------------------------

    async def get_futures_pairs(self, settlement: str | None = None) -> list[str]:
        """Get list of available Kraken futures trading pairs.

        Args:
            settlement (str | None): Filter by settlement currency (e.g., "USD").
                If None, returns all pairs.

        Returns:
            list[str]: List of ticker symbols (e.g., ["PI_XBTUSD", "PI_ETHUSD"]).

        Example:
            ```python
            async with KrakenClient() as client:
                perps = await client.get_futures_pairs()
            ```
        """
        if self._session is None:
            raise RuntimeError("KrakenClient must be used as an async context manager.")

        url = f"{self.futures_url}/tickers"

        for attempt in range(self.max_retries):
            try:
                async with self._session.get(url) as resp:
                    if resp.status == 429:
                        logger.warning("Rate limited, waiting 60s")
                        await asyncio.sleep(60)
                        continue

                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Kraken Futures API HTTP {resp.status}: {text}")

                    body = await resp.json()

                if body.get("result") != "success":
                    raise RuntimeError(f"Kraken Futures API error: {body.get('error')}")

                pairs: list[str] = []
                for ticker in body.get("tickers", []):
                    symbol = ticker.get("symbol", "")
                    # Filter perpetuals (PI_ prefix) and futures (FI_ prefix)
                    if symbol.startswith(("PI_", "PF_", "FI_")):
                        if settlement is None or settlement.upper() in symbol.upper():
                            pairs.append(symbol)

                return sorted(pairs)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(f"Request failed, retrying in {wait}s: {e}")
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(f"Failed to get futures pairs: {e}") from e

        return []

    async def get_futures_klines(
        self,
        symbol: str,
        timeframe: str = "1m",
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 5000,
    ) -> list[dict]:
        """Fetch OHLCV klines from Kraken Futures API.

        Kraken futures uses timestamps in SECONDS.
        Returned ``timestamp`` is the candle **close** time (UTC-naive).

        Args:
            symbol: Kraken futures symbol (e.g., "PI_XBTUSD").
            timeframe: Interval in SignalFlow format (1m, 1h, ...).
            start_time: Fetch candles from this time.
            end_time: Fetch candles until this time.
            limit: Max candles.

        Returns:
            List of OHLCV dicts sorted ascending by timestamp.

        Raises:
            RuntimeError: If not in async context or API error.
        """
        if self._session is None:
            raise RuntimeError("KrakenClient must be used as an async context manager.")

        interval = _KRAKEN_FUTURES_INTERVAL_MAP.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe for Kraken futures: {timeframe}")

        # Kraken futures candles endpoint
        url = f"{self.futures_url}/charts/v1/trade/{symbol}/{interval}"
        params: dict[str, object] = {}
        if start_time is not None:
            params["from"] = dt_to_sec_utc(start_time)
        if end_time is not None:
            params["to"] = dt_to_sec_utc(end_time)

        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 429:
                        logger.warning(f"Rate limited, waiting 60s (symbol={symbol})")
                        await asyncio.sleep(60)
                        continue

                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Kraken Futures API HTTP {resp.status}: {text}")

                    body = await resp.json()

                if "error" in body:
                    raise RuntimeError(f"Kraken Futures API error: {body['error']}")

                candles = body.get("candles", [])
                if not candles:
                    return []

                # Response: {time, open, high, low, close, volume}
                # time is in SECONDS (close time already)
                out: list[dict] = []
                for candle in candles:
                    out.append(
                        {
                            "timestamp": sec_to_dt_utc_naive(int(candle["time"])),
                            "open": float(candle["open"]),
                            "high": float(candle["high"]),
                            "low": float(candle["low"]),
                            "close": float(candle["close"]),
                            "volume": float(candle.get("volume", 0)),
                            "trades": 0,
                        }
                    )

                out.sort(key=lambda x: x["timestamp"])
                return out

            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(f"Request failed, retrying in {wait}s (symbol={symbol}): {e}")
                    await asyncio.sleep(wait)
                else:
                    break

        raise last_err or RuntimeError("Unknown error while fetching Kraken futures klines.")

    async def get_futures_klines_range(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        *,
        limit: int = 5000,
    ) -> list[dict]:
        """Download all futures klines for a period with pagination.

        Args:
            symbol: Kraken futures symbol.
            timeframe: Interval (must be in ``_KRAKEN_FUTURES_INTERVAL_MAP``).
            start_time: Range start (inclusive).
            end_time: Range end (inclusive).
            limit: Candles per request.

        Returns:
            Deduplicated, ascending OHLCV dicts.
        """
        if timeframe not in _KRAKEN_FUTURES_INTERVAL_MAP:
            raise ValueError(f"Unsupported timeframe for Kraken futures: {timeframe}")

        start_time = ensure_utc_naive(start_time)
        end_time = ensure_utc_naive(end_time)

        if start_time >= end_time:
            return []

        tf_ms = TIMEFRAME_MS.get(timeframe, 60_000)
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

            klines = await self.get_futures_klines(
                symbol=symbol,
                timeframe=timeframe,
                start_time=current_start,
                end_time=req_end,
                limit=limit,
            )

            if not klines:
                current_start = req_end + timedelta(seconds=1)
                await asyncio.sleep(self.min_delay_sec)
                continue

            for k in klines:
                ts = k["timestamp"]
                if start_time <= ts <= end_time:
                    all_klines.append(k)

            last_ts = klines[-1]["timestamp"]
            next_start = last_ts

            if next_start <= current_start:
                current_start = current_start + timedelta(seconds=1)
            else:
                current_start = next_start

            if len(all_klines) and len(all_klines) % 10000 == 0:
                logger.info(f"{symbol}: loaded {len(all_klines):,} candles...")

            await asyncio.sleep(self.min_delay_sec)

        uniq: dict[datetime, dict] = {}
        for k in all_klines:
            uniq[k["timestamp"]] = k

        out = list(uniq.values())
        out.sort(key=lambda x: x["timestamp"])
        return out


@dataclass
@sf_component(name="kraken/spot")
class KrakenSpotLoader(RawDataLoader):
    """Downloads and stores Kraken spot OHLCV data.

    Pairs are provided in compact format (e.g., "BTCUSD") and
    automatically converted to Kraken format (e.g., "XXBTZUSD").

    Note: Kraken spot has limited timeframe support (no 2h, 3m, 6h, 8h, 12h).

    Attributes:
        store: Storage backend.
        timeframe: Fixed timeframe for all data.
    """

    store: DuckDbSpotStore = field(default_factory=lambda: DuckDbSpotStore(db_path=Path("raw_data_kraken_spot.duckdb")))
    timeframe: str = "1m"

    async def get_pairs(self, quote: str | None = None) -> list[str]:
        """Get list of available Kraken spot trading pairs.

        Args:
            quote (str | None): Filter by quote currency (e.g., "USD").
                If None, returns all pairs.

        Returns:
            list[str]: List of pair names in Kraken format (e.g., ["XXBTZUSD"]).

        Example:
            ```python
            loader = KrakenSpotLoader(store=store)
            usd_pairs = await loader.get_pairs(quote="USD")
            ```
        """
        async with KrakenClient() as client:
            return await client.get_spot_pairs(quote=quote)

    async def download(
        self,
        pairs: list[str],
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        fill_gaps: bool = True,
    ) -> None:
        """Download historical Kraken spot data.

        Args:
            pairs: Trading pairs (e.g., ["BTCUSD"]) or Kraken names (e.g., ["XXBTZUSD"]).
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

        tf_minutes = _KRAKEN_SPOT_INTERVAL_MAP.get(self.timeframe, 1)

        async def download_pair(client: KrakenClient, pair: str) -> None:
            # Convert compact pair to Kraken format if needed
            kraken_pair = to_kraken_spot_symbol(pair)
            store_pair = normalize_kraken_spot_pair(kraken_pair)

            logger.info(f"Processing {pair} -> {kraken_pair} (kraken/spot) from {start} to {end}")

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
                    klines = await client.get_spot_klines_range(
                        pair=kraken_pair,
                        timeframe=self.timeframe,
                        start_time=range_start,
                        end_time=range_end,
                    )
                    self.store.insert_klines(store_pair, klines)
                except Exception as e:
                    logger.error(f"Error downloading {store_pair}: {e}")

        async with KrakenClient() as client:
            # Sequential downloads due to Kraken's strict rate limits
            for pair in pairs:
                await download_pair(client, pair)

        self.store.close()

    async def sync(
        self,
        pairs: list[str],
        update_interval_sec: int = 60,
    ) -> None:
        """Continuously sync latest Kraken spot data.

        Args:
            pairs: Trading pairs to sync.
            update_interval_sec: Update interval in seconds.
        """
        logger.info(f"Starting real-time sync (kraken/spot) for {pairs}")
        logger.info(f"Update interval: {update_interval_sec}s (timeframe={self.timeframe})")

        async def fetch_and_store(client: KrakenClient, pair: str) -> None:
            kraken_pair = to_kraken_spot_symbol(pair)
            store_pair = normalize_kraken_spot_pair(kraken_pair)
            try:
                klines = await client.get_spot_klines(
                    pair=kraken_pair,
                    timeframe=self.timeframe,
                )
                # Take last 5 candles
                if len(klines) > 5:
                    klines = klines[-5:]
                self.store.insert_klines(store_pair, klines)
            except Exception as e:
                logger.error(f"Error syncing {store_pair}: {e}")

        async with KrakenClient() as client:
            while True:
                # Sequential due to rate limits
                for pair in pairs:
                    await fetch_and_store(client, pair)
                    await asyncio.sleep(client.min_delay_sec)
                logger.debug(f"Synced {len(pairs)} pairs (kraken/spot)")
                await asyncio.sleep(update_interval_sec)


@dataclass
@sf_component(name="kraken/futures")
class KrakenFuturesLoader(RawDataLoader):
    """Downloads and stores Kraken futures OHLCV data.

    Pairs are provided in compact format (e.g., "BTCUSD") and
    automatically converted to Kraken futures format (e.g., "pi_xbtusd").

    Attributes:
        store: Storage backend.
        timeframe: Fixed timeframe for all data.
        prefix: Futures symbol prefix (default "PI_" for perpetuals).
    """

    store: DuckDbSpotStore = field(
        default_factory=lambda: DuckDbSpotStore(db_path=Path("raw_data_kraken_futures.duckdb"))
    )
    timeframe: str = "1m"
    prefix: str = "PI_"

    async def get_pairs(self, settlement: str | None = None) -> list[str]:
        """Get list of available Kraken futures pairs.

        Args:
            settlement (str | None): Filter by settlement currency.
                If None, returns all futures pairs.

        Returns:
            list[str]: List of ticker symbols (e.g., ["PI_XBTUSD"]).

        Example:
            ```python
            loader = KrakenFuturesLoader(store=store)
            perps = await loader.get_pairs()
            ```
        """
        async with KrakenClient() as client:
            return await client.get_futures_pairs(settlement=settlement)

    async def download(
        self,
        pairs: list[str],
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        fill_gaps: bool = True,
    ) -> None:
        """Download historical Kraken futures data.

        Args:
            pairs: Trading pairs (e.g., ["BTCUSD"]) or symbols (e.g., ["PI_XBTUSD"]).
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
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "12h": 720,
            "1d": 1440,
        }.get(self.timeframe, 1)

        async def download_pair(client: KrakenClient, pair: str) -> None:
            # Convert compact pair to Kraken futures symbol if needed
            if pair.upper().startswith(("PI_", "PF_", "FI_")):
                symbol = pair
            else:
                symbol = to_kraken_futures_symbol(pair, prefix=self.prefix)

            store_pair = normalize_kraken_futures_pair(symbol)
            logger.info(f"Processing {pair} -> {symbol} (kraken/futures) from {start} to {end}")

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
                    klines = await client.get_futures_klines_range(
                        symbol=symbol,
                        timeframe=self.timeframe,
                        start_time=range_start,
                        end_time=range_end,
                    )
                    self.store.insert_klines(store_pair, klines)
                except Exception as e:
                    logger.error(f"Error downloading {store_pair}: {e}")

        async with KrakenClient() as client:
            # Sequential downloads due to Kraken's strict rate limits
            for pair in pairs:
                await download_pair(client, pair)

        self.store.close()

    async def sync(
        self,
        pairs: list[str],
        update_interval_sec: int = 60,
    ) -> None:
        """Continuously sync latest Kraken futures data.

        Args:
            pairs: Trading pairs or symbols to sync.
            update_interval_sec: Update interval in seconds.
        """
        logger.info(f"Starting real-time sync (kraken/futures) for {pairs}")
        logger.info(f"Update interval: {update_interval_sec}s (timeframe={self.timeframe})")

        async def fetch_and_store(client: KrakenClient, pair: str) -> None:
            if pair.upper().startswith(("PI_", "PF_", "FI_")):
                symbol = pair
            else:
                symbol = to_kraken_futures_symbol(pair, prefix=self.prefix)

            store_pair = normalize_kraken_futures_pair(symbol)
            try:
                klines = await client.get_futures_klines(
                    symbol=symbol,
                    timeframe=self.timeframe,
                )
                # Take last 5 candles
                if len(klines) > 5:
                    klines = klines[-5:]
                self.store.insert_klines(store_pair, klines)
            except Exception as e:
                logger.error(f"Error syncing {store_pair}: {e}")

        async with KrakenClient() as client:
            while True:
                # Sequential due to rate limits
                for pair in pairs:
                    await fetch_and_store(client, pair)
                    await asyncio.sleep(client.min_delay_sec)
                logger.debug(f"Synced {len(pairs)} pairs (kraken/futures)")
                await asyncio.sleep(update_interval_sec)
