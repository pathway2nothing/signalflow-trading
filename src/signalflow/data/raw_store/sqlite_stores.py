from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import polars as pl
from loguru import logger

from signalflow.core import sf_component
from signalflow.data.raw_store.base import RawDataStore


def _adapt_datetime(val: datetime) -> str:
    return val.isoformat(" ")


def _convert_datetime(val: bytes) -> datetime:
    return datetime.fromisoformat(val.decode())


sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("TIMESTAMP", _convert_datetime)

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
@sf_component(name="sqlite/spot")
class SqliteSpotStore(RawDataStore):
    """SQLite storage backend for OHLCV spot data.

    Zero extra dependencies — uses stdlib sqlite3.
    Same interface as DuckDbSpotStore.
    """

    db_path: Path
    timeframe: str = "1m"
    _con: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._con = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        self._con.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        self._con.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                pair TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                trades INTEGER,
                PRIMARY KEY (pair, timestamp)
            );
            CREATE INDEX IF NOT EXISTS idx_ohlcv_pair_ts
                ON ohlcv(pair, timestamp DESC);
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        self._con.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('timeframe', ?)",
            (self.timeframe,),
        )
        self._con.commit()
        logger.info(f"Database initialized: {self.db_path} (timeframe={self.timeframe})")

    def insert_klines(self, pair: str, klines: list[dict]) -> None:
        """Upsert klines (INSERT OR REPLACE)."""
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
        self._con.executemany(
            "INSERT OR REPLACE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._con.commit()
        logger.debug(f"Inserted {len(klines):,} rows for {pair}")

    def get_time_bounds(self, pair: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get earliest and latest timestamps for a pair."""
        result = self._con.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv WHERE pair = ?",
            (pair,),
        ).fetchone()
        if not result or result[0] is None:
            return (None, None)
        mn = result[0] if isinstance(result[0], datetime) else datetime.fromisoformat(result[0])
        mx = result[1] if isinstance(result[1], datetime) else datetime.fromisoformat(result[1])
        return (mn, mx)

    def find_gaps(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        tf_minutes: int,
    ) -> list[tuple[datetime, datetime]]:
        """Find gaps in data coverage for a pair."""
        existing = self._con.execute(
            "SELECT timestamp FROM ohlcv WHERE pair = ? AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (pair, start, end),
        ).fetchall()

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
        query = "SELECT pair, timestamp, open, high, low, close, volume, trades FROM ohlcv WHERE pair = ?"
        params: list[object] = [pair]

        if hours is not None:
            cutoff = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
            query += " AND timestamp > ?"
            params.append(cutoff)
        elif start and end:
            query += " AND timestamp BETWEEN ? AND ?"
            params.extend([start, end])
        elif start:
            query += " AND timestamp >= ?"
            params.append(start)
        elif end:
            query += " AND timestamp <= ?"
            params.append(end)

        query += " ORDER BY timestamp"
        rows = self._con.execute(query, params).fetchall()
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

        placeholders = ",".join(["?"] * len(pairs))
        query = (
            f"SELECT pair, timestamp, open, high, low, close, volume, trades FROM ohlcv WHERE pair IN ({placeholders})"
        )
        params: list[object] = [*pairs]

        if hours is not None:
            cutoff = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
            query += " AND timestamp > ?"
            params.append(cutoff)
        elif start and end:
            query += " AND timestamp BETWEEN ? AND ?"
            params.extend([start, end])
        elif start:
            query += " AND timestamp >= ?"
            params.append(start)
        elif end:
            query += " AND timestamp <= ?"
            params.append(end)

        query += " ORDER BY pair, timestamp"
        rows = self._con.execute(query, params).fetchall()
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
        """Get database statistics per pair."""
        rows = self._con.execute("""
            SELECT
                pair,
                COUNT(*) as rows,
                MIN(timestamp) as first_candle,
                MAX(timestamp) as last_candle,
                ROUND(SUM(volume), 2) as total_volume
            FROM ohlcv
            GROUP BY pair
            ORDER BY pair
        """).fetchall()
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
