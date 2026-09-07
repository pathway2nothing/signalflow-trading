"""Binance source - real public klines REST, stdlib-only (no extra deps)."""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import polars as pl
from loguru import logger

from signalflow._time import interval_seconds, to_epoch
from signalflow.data.source.base import Source, validate_frame
from signalflow.decorators import source

_BASE = "https://api.binance.com/api/v3/klines"
_LIMIT = 1000


@source("binance")
@dataclass
class BinanceSource(Source):
    """Fetch spot OHLCV klines from Binance public REST."""

    name: str = "binance"
    base_url: str = _BASE
    timeout: float = 20.0
    max_retries: int = 3

    def fetch(
        self,
        pairs: list[str],
        start: str,
        end: str | None = None,
        interval: str = "1h",
    ) -> pl.DataFrame:
        interval_seconds(interval)  # fail fast on an unsupported interval
        start_ms = to_epoch(start) * 1000
        end_ms = (to_epoch(end) * 1000) if end else int(time.time() * 1000)
        frames = [self._fetch_pair(p, start_ms, end_ms, interval) for p in pairs]
        return validate_frame(pl.concat([f for f in frames if f.height > 0]))

    def _fetch_pair(self, pair: str, start_ms: int, end_ms: int, interval: str) -> pl.DataFrame:
        step = interval_seconds(interval) * 1000
        expected = max(1, (end_ms - start_ms) // (step * _LIMIT) + 1)
        logger.info(f"binance: {pair} {interval}: fetching ~{expected} request(s)")
        rows: list[tuple] = []
        n_req = 0
        cursor = start_ms
        while cursor < end_ms:
            batch = self._request(pair, interval, cursor, end_ms)
            n_req += 1
            if not batch:
                break
            for k in batch:
                rows.append((k[0] + step, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
            logger.debug(
                f"binance: {pair} {interval}: request {n_req}/~{expected}, +{len(batch)} bars ({len(rows)} total)"
            )
            last_open = batch[-1][0]
            cursor = last_open + step
            if len(batch) < _LIMIT:
                break
        logger.info(f"binance: {pair} {interval}: fetched {len(rows)} bars in {n_req} request(s)")
        if not rows:
            logger.warning(f"binance: no klines for {pair}")
            return pl.DataFrame(
                schema={
                    "pair": pl.Utf8,
                    "ts": pl.Datetime("ms"),
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                }
            )
        ts, o, h, lo, c, v = zip(*rows, strict=True)
        return pl.DataFrame(
            {
                "pair": [pair] * len(rows),
                "ts": list(ts),
                "open": list(o),
                "high": list(h),
                "low": list(lo),
                "close": list(c),
                "volume": list(v),
            }
        ).with_columns(pl.col("ts").cast(pl.Datetime("ms")))

    def _request(self, pair: str, interval: str, start_ms: int, end_ms: int) -> list:
        url = f"{self.base_url}?symbol={pair}&interval={interval}&startTime={start_ms}&endTime={end_ms}&limit={_LIMIT}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "signalflow/1.0"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise ConnectionError(f"binance request failed for {pair}: {last_err}")
