from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import polars as pl
from loguru import logger

from signalflow.core import sf_component, SfComponentType
from signalflow.core.registry import default_registry
from signalflow.data.raw_store.base import RawDataStore
from signalflow.data.raw_store._schema import (
    CORE_COLUMNS,
    normalize_ts,
    polars_schema,
    resolve_columns,
)


def _adapt_datetime(val: datetime) -> str:
    return val.isoformat(" ")


def _convert_datetime(val: bytes) -> datetime:
    return datetime.fromisoformat(val.decode())


sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("TIMESTAMP", _convert_datetime)

# SQLite type mapping.
_SQL_TYPES: dict[str, str] = {
    "pair": "TEXT NOT NULL",
    "timestamp": "TIMESTAMP NOT NULL",
    "trades": "INTEGER",
}


@dataclass
@sf_component(name="sqlite/spot")
class SqliteRawStore(RawDataStore):
    """SQLite storage backend for raw market data.

    Supports any registered raw data type (spot, futures, perpetual, custom).
    Zero extra dependencies — uses stdlib sqlite3.
    """

    db_path: Path
    data_type: str = "spot"
    timeframe: str = "1m"
    _con: sqlite3.Connection = field(init=False, repr=False)
    _columns: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._columns = resolve_columns(self.data_type)
        self._con = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        self._con.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

    # ── Schema management ─────────────────────────────────────────────

    @property
    def _all_column_names(self) -> list[str]:
        return ["pair", "timestamp"] + self._columns

    def _col_sql_type(self, col: str) -> str:
        if col in _SQL_TYPES:
            return _SQL_TYPES[col]
        return "REAL NOT NULL" if col in CORE_COLUMNS else "REAL"

    def _ensure_tables(self) -> None:
        # Check existing columns
        cur = self._con.execute("PRAGMA table_info(ohlcv)")
        existing_cols = {row[1] for row in cur.fetchall()}

        if not existing_cols:
            col_defs = ", ".join(f"{c} {self._col_sql_type(c)}" for c in self._all_column_names)
            self._con.executescript(f"""
                CREATE TABLE ohlcv (
                    {col_defs},
                    PRIMARY KEY (pair, timestamp)
                );
                CREATE INDEX idx_ohlcv_pair_ts ON ohlcv(pair, timestamp DESC);
            """)
        else:
            for col in self._columns:
                if col not in existing_cols:
                    sql_type = self._col_sql_type(col)
                    self._con.execute(f"ALTER TABLE ohlcv ADD COLUMN {col} {sql_type}")
                    logger.info(f"Added column {col} ({sql_type}) to ohlcv table")

        self._con.executescript("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_pair_ts ON ohlcv(pair, timestamp DESC);
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        self._con.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('timeframe', ?)",
            (self.timeframe,),
        )
        self._con.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('data_type', ?)",
            (self.data_type,),
        )
        self._con.commit()
        logger.info(f"Database initialized: {self.db_path} (data_type={self.data_type}, timeframe={self.timeframe})")

    # ── Insert ────────────────────────────────────────────────────────

    def insert_klines(self, pair: str, klines: list[dict]) -> None:
        if not klines:
            return
        n_cols = len(self._all_column_names)
        placeholders = ", ".join(["?"] * n_cols)
        rows = [(pair, normalize_ts(k["timestamp"])) + tuple(k.get(c) for c in self._columns) for k in klines]
        self._con.executemany(
            f"INSERT OR REPLACE INTO ohlcv VALUES ({placeholders})",
            rows,
        )
        self._con.commit()
        logger.debug(f"Inserted {len(klines):,} rows for {pair}")

    # ── Queries ───────────────────────────────────────────────────────

    def get_time_bounds(self, pair: str) -> tuple[Optional[datetime], Optional[datetime]]:
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

    # ── Load ──────────────────────────────────────────────────────────

    def _rows_to_df(self, rows: list[tuple]) -> pl.DataFrame:
        if not rows:
            return pl.DataFrame(schema=polars_schema(self._columns))
        all_cols = self._all_column_names
        data = {col: [row[i] for row in rows] for i, col in enumerate(all_cols)}
        df = pl.DataFrame(data)
        if "timestamp" in df.columns:
            df = df.with_columns(pl.col("timestamp").cast(pl.Datetime).dt.replace_time_zone(None))
        return df

    def load(
        self,
        pair: str,
        hours: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        col_list = ", ".join(self._all_column_names)
        query = f"SELECT {col_list} FROM ohlcv WHERE pair = ?"
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
        return self._rows_to_df(rows)

    def load_many(
        self,
        pairs: Iterable[str],
        hours: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        pairs = list(pairs)
        if not pairs:
            return pl.DataFrame(schema=polars_schema(self._columns))

        col_list = ", ".join(self._all_column_names)
        placeholders = ",".join(["?"] * len(pairs))
        query = f"SELECT {col_list} FROM ohlcv WHERE pair IN ({placeholders})"
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
        return self._rows_to_df(rows)

    def load_many_pandas(
        self,
        pairs: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        return self.load_many(pairs=pairs, start=start, end=end).to_pandas()

    # ── Stats / lifecycle ─────────────────────────────────────────────

    def get_stats(self) -> pl.DataFrame:
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


# Register for futures and perpetual.
default_registry.register(SfComponentType.RAW_DATA_STORE, "sqlite/futures", SqliteRawStore, override=True)
default_registry.register(SfComponentType.RAW_DATA_STORE, "sqlite/perpetual", SqliteRawStore, override=True)

# Backward-compatible alias.
SqliteSpotStore = SqliteRawStore
