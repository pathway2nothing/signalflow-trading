import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from loguru import logger

from .data_store import SpotStore
from .cex_clients import BinanceClient


@dataclass
class BinanceSpotLoader:
    """Downloads and stores Binance spot OHLCV data for a fixed project timeframe."""

    db_path: Path = field(default_factory=lambda: Path("raw_data.duckdb"))
    timeframe: str = "1m" 

    async def download(
        self,
        pairs: list[str],
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        fill_gaps: bool = True,
    ):
        store = SpotStore(self.db_path, timeframe=self.timeframe)

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if end is None:
            end = now
        if start is None:
            start = end - timedelta(days=days if days else 7)

        tf_minutes = {
            "1m": 1, "5m": 5, "15m": 15,
            "1h": 60, "4h": 240, "1d": 1440,
        }.get(self.timeframe, 1)

        async def download_pair(client: BinanceClient, pair: str):
            logger.info(f"Processing {pair} from {start} to {end}")

            db_min, db_max = store.get_time_bounds(pair)
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
                        gaps = store.find_gaps(pair, overlap_start, overlap_end, tf_minutes)
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
                    store.insert_klines(pair, klines)
                except Exception as e:
                    logger.error(f"Error downloading {pair}: {e}")

        async with BinanceClient() as client:
            import asyncio
            await asyncio.gather(*[download_pair(client, pair) for pair in pairs])
        store.close()

    async def sync(
        self,
        pairs: list[str],
        update_interval_sec: int = 60,
    ):
        from .cex_clients import BinanceClient 

        store = SpotStore(self.db_path, timeframe=self.timeframe)

        logger.info(f"Starting real-time sync for {pairs}")
        logger.info(f"Update interval: {update_interval_sec}s (timeframe={self.timeframe})")

        async def fetch_and_store(client: BinanceClient, pair: str):
            try:
                klines = await client.get_klines(pair=pair, timeframe=self.timeframe, limit=5)
                store.insert_klines(pair, klines)
            except Exception as e:
                logger.error(f"Error syncing {pair}: {e}")

        async with BinanceClient() as client:
            while True:
                await asyncio.gather(*[fetch_and_store(client, pair) for pair in pairs])
                logger.debug(f"Synced {len(pairs)} pairs")
                await asyncio.sleep(update_interval_sec)
