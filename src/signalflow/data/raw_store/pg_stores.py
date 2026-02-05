from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import pandas as pd
import polars as pl
from loguru import logger

from signalflow.core import sf_component
from signalflow.data.raw_store.base import RawDataStore

try:
    import psycopg
except ImportError:
    psycopg = None  # type: ignore[assignment]

_PG_MISSING_MSG = "psycopg is required for PostgreSQL stores. Install with: pip install signalflow-trading[postgres]"

_EMPTY_SCHEMA = {
    "pair": pl.Utf8,
    "timestamp": pl.Datetime,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "trades": pl.Int64,
}

_COLUMNS = ["pair", "timestamp", "open", "high", "low", "close", "volume", "trades"]


def _rows_to_df(rows: list[tuple]) -> pl.DataFrame:
    """Convert raw cursor rows to a Polars DataFrame with standard OHLCV schema."""
    if not rows:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)
    data = {col: [row[i] for row in rows] for i, col in enumerate(_COLUMNS)}
    df = pl.DataFrame(data)
    if "timestamp" in df.columns:
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime).dt.replace_time_zone(None))
    return df


def _normalize_ts(ts: datetime) -> datetime:
    """Normalize timestamp: strip tz, zero seconds/microseconds, round up if needed."""
    if ts.second != 0 or ts.microsecond != 0:
        return ts.replace(tzinfo=None, second=0, microsecond=0) + timedelta(minutes=1)
    return ts.replace(tzinfo=None)


@dataclass
@sf_component(name="postgres/spot")
class PgSpotStore(RawDataStore):
    """PostgreSQL storage backend for OHLCV spot data.

    Requires psycopg. Install with: pip install signalflow-trading[postgres]
    Connection string from dsn param or SIGNALFLOW_PG_DSN env var.
    """

    dsn: str = ""
    timeframe: str = "1m"
    _con: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if psycopg is None:
            raise ImportError(_PG_MISSING_MSG)
        resolved_dsn = self.dsn or os.environ.get("SIGNALFLOW_PG_DSN", "")
        if not resolved_dsn:
            raise ValueError("PostgreSQL DSN must be provided via dsn param or SIGNALFLOW_PG_DSN env var")
        self._con = psycopg.connect(resolved_dsn, autocommit=False)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self._con.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    pair TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open DOUBLE PRECISION NOT NULL,
                    high DOUBLE PRECISION NOT NULL,
                    low DOUBLE PRECISION NOT NULL,
                    close DOUBLE PRECISION NOT NULL,
                    volume DOUBLE PRECISION NOT NULL,
                    trades INTEGER,
                    PRIMARY KEY (pair, timestamp)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_pair_ts
                ON ohlcv(pair, timestamp DESC)
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            cur.execute(
                "INSERT INTO meta(key, value) VALUES (%s, %s) ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value",
                (self.timeframe, self.timeframe),
            )
        self._con.commit()
        logger.info(f"PostgreSQL database initialized (timeframe={self.timeframe})")

    def insert_klines(self, pair: str, klines: list[dict]) -> None:
        """Upsert klines via INSERT ON CONFLICT DO UPDATE."""
        if not klines:
            return
        rows = [
            (
                pair,
                _normalize_ts(k["timestamp"]),
                k["open"],
                k["high"],
                k["low"],
                k["close"],
                k["volume"],
                k.get("trades"),
            )
            for k in klines
        ]
        with self._con.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO ohlcv(pair, timestamp, open, high, low, close, volume, trades)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(pair, timestamp) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high,
                    low = EXCLUDED.low, close = EXCLUDED.close,
                    volume = EXCLUDED.volume, trades = EXCLUDED.trades
                """,
                rows,
            )
        self._con.commit()
        logger.debug(f"Inserted {len(klines):,} rows for {pair}")

    def get_time_bounds(self, pair: str) -> tuple[Optional[datetime], Optional[datetime]]:
        with self._con.cursor() as cur:
            cur.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv WHERE pair = %s",
                (pair,),
            )
            result = cur.fetchone()
        return (result[0], result[1]) if result and result[0] else (None, None)

    def find_gaps(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        tf_minutes: int,
    ) -> list[tuple[datetime, datetime]]:
        with self._con.cursor() as cur:
            cur.execute(
                "SELECT timestamp FROM ohlcv WHERE pair = %s AND timestamp BETWEEN %s AND %s ORDER BY timestamp",
                (pair, start, end),
            )
            existing = cur.fetchall()

        if not existing:
            return [(start, end)]

        existing_times = {row[0] for row in existing}
        gaps: list[tuple[datetime, datetime]] = []
        gap_start: Optional[datetime] = None
        current = start

        while current <= end:
            if current not in existing_times:
                if gap_start is None:
                    gap_start = current
            else:
                if gap_start is not None:
                    gaps.append((gap_start, current - timedelta(minutes=tf_minutes)))
                    gap_start = None
            current += timedelta(minutes=tf_minutes)

        if gap_start is not None:
            gaps.append((gap_start, end))

        return gaps

    def load(
        self,
        pair: str,
        hours: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        query = "SELECT pair, timestamp, open, high, low, close, volume, trades FROM ohlcv WHERE pair = %s"
        params: list[object] = [pair]

        if hours is not None:
            cutoff = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
            query += " AND timestamp > %s"
            params.append(cutoff)
        elif start and end:
            query += " AND timestamp BETWEEN %s AND %s"
            params.extend([start, end])
        elif start:
            query += " AND timestamp >= %s"
            params.append(start)
        elif end:
            query += " AND timestamp <= %s"
            params.append(end)

        query += " ORDER BY timestamp"
        with self._con.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return _rows_to_df(rows)

    def load_many(
        self,
        pairs: Iterable[str],
        hours: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        pairs = list(pairs)
        if not pairs:
            return pl.DataFrame(schema=_EMPTY_SCHEMA)

        placeholders = ",".join(["%s"] * len(pairs))
        query = (
            f"SELECT pair, timestamp, open, high, low, close, volume, trades FROM ohlcv WHERE pair IN ({placeholders})"
        )
        params: list[object] = [*pairs]

        if hours is not None:
            cutoff = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
            query += " AND timestamp > %s"
            params.append(cutoff)
        elif start and end:
            query += " AND timestamp BETWEEN %s AND %s"
            params.extend([start, end])
        elif start:
            query += " AND timestamp >= %s"
            params.append(start)
        elif end:
            query += " AND timestamp <= %s"
            params.append(end)

        query += " ORDER BY pair, timestamp"
        with self._con.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return _rows_to_df(rows)

    def load_many_pandas(
        self,
        pairs: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        df_pl = self.load_many(pairs=pairs, start=start, end=end)
        return df_pl.to_pandas()

    def get_stats(self) -> pl.DataFrame:
        with self._con.cursor() as cur:
            cur.execute("""
                SELECT
                    pair,
                    COUNT(*) as rows,
                    MIN(timestamp) as first_candle,
                    MAX(timestamp) as last_candle,
                    ROUND(SUM(volume)::numeric, 2) as total_volume
                FROM ohlcv
                GROUP BY pair
                ORDER BY pair
            """)
            rows = cur.fetchall()
        if not rows:
            return pl.DataFrame(
                schema={
                    "pair": pl.Utf8,
                    "rows": pl.Int64,
                    "first_candle": pl.Datetime,
                    "last_candle": pl.Datetime,
                    "total_volume": pl.Float64,
                }
            )
        cols = ["pair", "rows", "first_candle", "last_candle", "total_volume"]
        data = {col: [row[i] for row in rows] for i, col in enumerate(cols)}
        return pl.DataFrame(data)

    def close(self) -> None:
        self._con.close()
