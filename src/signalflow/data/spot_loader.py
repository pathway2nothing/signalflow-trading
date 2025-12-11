import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from loguru import logger

from .spot_store import SpotStore
from .cex_clients import BinanceClient


@dataclass
class BinanceSpotLoader:
    """Downloads and stores Binance spot OHLCV data."""
    
    db_path: Path = field(default_factory=lambda: Path("raw_data.duckdb"))
    
    async def download(
        self,
        pairs: list[str],
        timeframe: str = "1m",
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        fill_gaps: bool = True,
    ):
        """Downloads historical data for a list of pairs.
        
        Args:
            pairs: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT']).
            timeframe: Candle interval.
            days: Number of days to download (alternative to start/end).
            start: Start of the range.
            end: End of the range (defaults to now).
            fill_gaps: Whether to fill gaps in existing data.
        """
        store = SpotStore(self.db_path)
        
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        
        if end is None:
            end = now
        if start is None:
            if days:
                start = end - timedelta(days=days)
            else:
                start = end - timedelta(days=7)
        
        tf_minutes = {
            "1m": 1, "5m": 5, "15m": 15, 
            "1h": 60, "4h": 240, "1d": 1440
        }.get(timeframe, 1)
        
        async def download_pair(client: BinanceClient, pair: str):
            logger.info(f"Processing {pair} from {start} to {end}")
            
            db_min, db_max = store.get_time_bounds(pair, timeframe)
            ranges_to_download = []
            
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
                        gaps = store.find_gaps(pair, timeframe, overlap_start, overlap_end, tf_minutes)
                        ranges_to_download.extend(gaps)
            
            for range_start, range_end in ranges_to_download:
                if range_start >= range_end:
                    continue
                    
                logger.info(f"{pair}: downloading {range_start} -> {range_end}")
                
                try:
                    klines = await client.get_klines_range(
                        symbol=pair,
                        timeframe=timeframe,
                        start_time=range_start,
                        end_time=range_end,
                    )
                    store.insert_klines(pair, timeframe, klines)
                    
                except Exception as e:
                    logger.error(f"Error downloading {pair}: {e}")
        
        async with BinanceClient() as client:
            await asyncio.gather(*[download_pair(client, pair) for pair in pairs])
        
        logger.info("=" * 60)
        print(store.get_stats())
        
        store.close()
    
    async def sync(
        self,
        pairs: list[str],
        timeframe: str = "1m",
        update_interval_sec: int = 60,
    ):
        """Real-time sync that updates latest candles periodically.
        
        Args:
            pairs: List of trading pairs.
            timeframe: Candle interval.
            update_interval_sec: Seconds between updates.
        """
        store = SpotStore(self.db_path)
        
        logger.info(f"Starting real-time sync for {pairs}")
        logger.info(f"Update interval: {update_interval_sec}s")
        
        async def fetch_and_store(client: BinanceClient, pair: str):
            try:
                klines = await client.get_klines(symbol=pair, timeframe=timeframe, limit=5)
                store.insert_klines(pair, timeframe, klines)
            except Exception as e:
                logger.error(f"Error syncing {pair}: {e}")
        
        async with BinanceClient() as client:
            while True:
                await asyncio.gather(*[fetch_and_store(client, pair) for pair in pairs])
                logger.debug(f"Synced {len(pairs)} pairs")
                await asyncio.sleep(update_interval_sec)