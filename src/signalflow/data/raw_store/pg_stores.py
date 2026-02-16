from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import polars as pl
from loguru import logger

from signalflow.core import SfComponentType, sf_component
from signalflow.core.registry import default_registry
from signalflow.data.raw_store._schema import (
    CORE_COLUMNS,
    normalize_ts,
    polars_schema,
    resolve_columns,
)
from signalflow.data.raw_store.base import RawDataStore

try:
    import psycopg
except ImportError:
    psycopg = None  # type: ignore[assignment]

_PG_MISSING_MSG = "psycopg is required for PostgreSQL stores. Install with: pip install signalflow-trading[postgres]"

# PostgreSQL type mapping.
_SQL_TYPES: dict[str, str] = {
    "pair": "TEXT NOT NULL",
    "timestamp": "TIMESTAMP NOT NULL",
    "trades": "INTEGER",
}


@dataclass
@sf_component(name="postgres/spot")
class PgRawStore(RawDataStore):
    """PostgreSQL storage backend for raw market data.

    Supports any registered raw data type (spot, futures, perpetual, custom).
    Requires psycopg. Install with: pip install signalflow-trading[postgres]
    Connection string from dsn param or SIGNALFLOW_PG_DSN env var.
    """

    dsn: str = ""
    data_type: str = "spot"
    timeframe: str = "1m"
    _con: Any = field(init=False, repr=False)
    _columns: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if psycopg is None:
            raise ImportError(_PG_MISSING_MSG)
        resolved_dsn = self.dsn or os.environ.get("SIGNALFLOW_PG_DSN", "")
        if not resolved_dsn:
            raise ValueError("PostgreSQL DSN must be provided via dsn param or SIGNALFLOW_PG_DSN env var")
        self._columns = resolve_columns(self.data_type)
        self._con = psycopg.connect(resolved_dsn, autocommit=False)
        self._ensure_tables()

    # ── Schema management ─────────────────────────────────────────────

    @property
    def _all_column_names(self) -> list[str]:
        return ["pair", "timestamp"] + self._columns

    def _col_sql_type(self, col: str) -> str:
        if col in _SQL_TYPES:
            return _SQL_TYPES[col]
        return "DOUBLE PRECISION NOT NULL" if col in CORE_COLUMNS else "DOUBLE PRECISION"

    def _ensure_tables(self) -> None:
        with self._con.cursor() as cur:
            # Check existing columns
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'ohlcv'
            """)
            existing_cols = {row[0] for row in cur.fetchall()}

            if not existing_cols:
                col_defs = ", ".join(f"{c} {self._col_sql_type(c)}" for c in self._all_column_names)
                cur.execute(f"""
                    CREATE TABLE ohlcv (
                        {col_defs},
                        PRIMARY KEY (pair, timestamp)
                    )
                """)
            else:
                for col in self._columns:
                    if col not in existing_cols:
                        sql_type = self._col_sql_type(col)
                        cur.execute(f"ALTER TABLE ohlcv ADD COLUMN {col} {sql_type}")
                        logger.info(f"Added column {col} ({sql_type}) to ohlcv table")

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
                ("timeframe", self.timeframe),
            )
            cur.execute(
                "INSERT INTO meta(key, value) VALUES (%s, %s) ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value",
                ("data_type", self.data_type),
            )
        self._con.commit()
        logger.info(f"PostgreSQL database initialized (data_type={self.data_type}, timeframe={self.timeframe})")

    # ── Insert ────────────────────────────────────────────────────────

    def insert_klines(self, pair: str, klines: list[dict]) -> None:
        if not klines:
            return
        col_names = ", ".join(self._all_column_names)
        placeholders = ", ".join(["%s"] * len(self._all_column_names))
        update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in self._columns)

        rows = [(pair, normalize_ts(k["timestamp"])) + tuple(k.get(c) for c in self._columns) for k in klines]
        with self._con.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO ohlcv({col_names})
                VALUES ({placeholders})
                ON CONFLICT(pair, timestamp) DO UPDATE SET
                    {update_set}
                """,
                rows,
            )
        self._con.commit()
        logger.debug(f"Inserted {len(klines):,} rows for {pair}")

    # ── Queries ───────────────────────────────────────────────────────

    def get_time_bounds(self, pair: str) -> tuple[datetime | None, datetime | None]:
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
        gap_start: datetime | None = None
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
        hours: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        col_list = ", ".join(self._all_column_names)
        query = f"SELECT {col_list} FROM ohlcv WHERE pair = %s"
        params: list[object] = [pair]

        if hours is not None:
            cutoff = datetime.now(tz=UTC).replace(tzinfo=None) - timedelta(hours=hours)
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
        return self._rows_to_df(rows)

    def load_many(
        self,
        pairs: Iterable[str],
        hours: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        pairs = list(pairs)
        if not pairs:
            return pl.DataFrame(schema=polars_schema(self._columns))

        col_list = ", ".join(self._all_column_names)
        placeholders = ",".join(["%s"] * len(pairs))
        query = f"SELECT {col_list} FROM ohlcv WHERE pair IN ({placeholders})"
        params: list[object] = [*pairs]

        if hours is not None:
            cutoff = datetime.now(tz=UTC).replace(tzinfo=None) - timedelta(hours=hours)
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


# Register for futures and perpetual.
default_registry.register(SfComponentType.RAW_DATA_STORE, "postgres/futures", PgRawStore, override=True)
default_registry.register(SfComponentType.RAW_DATA_STORE, "postgres/perpetual", PgRawStore, override=True)

# Backward-compatible alias.
PgSpotStore = PgRawStore
