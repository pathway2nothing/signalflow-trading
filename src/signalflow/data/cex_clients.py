
import asyncio
import aiohttp

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger


@dataclass
class BinanceClient:
    """Async client for Binance REST API."""
    
    base_url: str = "https://api.binance.com"
    max_retries: int = 3
    _session: Optional[aiohttp.ClientSession] = field(default=None, init=False)
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()
    
    async def get_klines(
        self,
        pair: str,
        timeframe: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> list[dict]:
        """Fetches OHLCV data from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT').
            timeframe: Candle interval ('1m', '5m', '15m', '1h', '4h', '1d').
            start_time: Start of the range.
            end_time: End of the range.
            limit: Max candles per request (max 1000).
        
        Returns:
            List of OHLCV dictionaries.
        """
        params = {
            "symbol": pair,
            "interval": timeframe,
            "limit": limit
        }
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        url = f"{self.base_url}/api/v3/klines"
        
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
                        raise Exception(f"Binance API error {resp.status}: {text}")
                    
                    data = await resp.json()
                    break
                    
            except aiohttp.ClientError as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Request failed, retrying in {wait}s: {e}")
                    await asyncio.sleep(wait)
                else:
                    raise
        
        return [
            {
                "timestamp": datetime.fromtimestamp(int(k[6]) / 1000),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[7]),
                "trades": int(k[8]),
            }
            for k in data
        ]
    
    async def get_klines_range(
        self,
        pair: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict]:
        """Downloads all klines for a specified period with pagination.
        
        Args:
            symbol: Trading pair.
            timeframe: Candle interval.
            start_time: Start of the range.
            end_time: End of the range.
        
        Returns:
            List of all OHLCV dictionaries in the range.
        """
        all_klines = []
        current_start = start_time
        
        timeframe_ms = {
            "1m": 60_000,
            "5m": 300_000,
            "15m": 900_000,
            "1h": 3_600_000,
            "4h": 14_400_000,
            "1d": 86_400_000,
        }
        step = timedelta(milliseconds=timeframe_ms.get(timeframe, 60_000) * 1000)
        
        while current_start < end_time:
            klines = await self.get_klines(
                pair=pair,
                timeframe=timeframe,
                start_time=current_start,
                end_time=min(current_start + step, end_time),
                limit=1000
            )
            
            if not klines:
                break
            
            all_klines.extend(klines)
            current_start = klines[-1]["timestamp"] + timedelta(milliseconds=1)
            await asyncio.sleep(0.05)
            
            if len(all_klines) % 10000 == 0:
                logger.info(f"{pair}: loaded {len(all_klines):,} candles...")
        
        return all_klines

