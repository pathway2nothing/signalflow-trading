from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl
from loguru import logger

from signalflow.core import RawData, SfComponentType, sf_component
from signalflow.core.registry import default_registry
from signalflow.data.raw_store._schema import (
    CORE_COLUMNS,
    normalize_ts,
    polars_schema,
    resolve_columns,
)
from signalflow.data.raw_store.base import RawDataStore

# Column name -> DuckDB SQL type.  Anything not listed defaults to DOUBLE.
_SQL_TYPES: dict[str, str] = {
    "pair": "VARCHAR NOT NULL",
    "timestamp": "TIMESTAMP NOT NULL",
    "trades": "INTEGER",
}


@dataclass
@sf_component(name="duckdb/spot")
class DuckDbRawStore(RawDataStore):
    """DuckDB storage backend for raw market data.

    Supports any registered raw data type (spot, futures, perpetual, custom).
    Schema is derived dynamically from the registry based on ``data_type``.

    Key features:
        - Dynamic schema from registry columns
        - Automatic column migration when upgrading data type
        - Legacy schema migration (timeframe / open_time columns)
        - Efficient batch inserts with upsert (INSERT OR REPLACE)
        - Gap detection for data continuity checks
        - Multi-pair batch loading

    Attributes:
        db_path: Path to DuckDB file.
        data_type: Raw data type name (e.g. "spot", "futures", "perpetual").
        timeframe: Fixed timeframe for all data (e.g. "1m", "5m").

    Example:
        ```python
        from signalflow.data.raw_store import DuckDbRawStore

        # Spot store (default)
        spot = DuckDbRawStore(db_path=Path("data/spot.duckdb"))

        # Futures store with open_interest column
        futures = DuckDbRawStore(
            db_path=Path("data/futures.duckdb"),
            data_type="futures",
        )

        # Perpetual store with funding_rate + open_interest
        perp = DuckDbRawStore(
            db_path=Path("data/perp.duckdb"),
            data_type="perpetual",
        )
        ```
    """

    db_path: Path
    data_type: str = "spot"
    timeframe: str = "1m"
    _con: duckdb.DuckDBPyConnection = field(init=False, repr=False)
    _columns: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize database connection and ensure schema."""
        self._columns = resolve_columns(self.data_type)
        self._con = duckdb.connect(str(self.db_path))
        self._ensure_tables()

    # ── Schema management ─────────────────────────────────────────────

    @property
    def _all_column_names(self) -> list[str]:
        """Full column list including pair and timestamp."""
        return ["pair", "timestamp", *self._columns]

    def _col_sql_type(self, col: str) -> str:
        """SQL type string for a column."""
        if col in _SQL_TYPES:
            return _SQL_TYPES[col]
        return "DOUBLE NOT NULL" if col in CORE_COLUMNS else "DOUBLE"

    def _ensure_tables(self) -> None:
        """Create tables, migrate legacy schema, and add missing columns."""
        existing = self._con.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'ohlcv'
        """).fetchall()
        existing_cols = {row[0] for row in existing}

        # Legacy migration: timeframe column or open_time column
        if existing_cols and ("timeframe" in existing_cols or "open_time" in existing_cols):
            self._migrate_legacy(existing_cols)
            # Re-read columns after migration
            existing = self._con.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'ohlcv'
            """).fetchall()
            existing_cols = {row[0] for row in existing}

        if not existing_cols:
            # Fresh database — create table with full schema
            col_defs = ",\n                    ".join(f"{c} {self._col_sql_type(c)}" for c in self._all_column_names)
            self._con.execute(f"""
                CREATE TABLE ohlcv (
                    {col_defs},
                    PRIMARY KEY (pair, timestamp)
                )
            """)
        else:
            # Existing table — add any missing columns (e.g. upgrading spot → futures)
            for col in self._columns:
                if col not in existing_cols:
                    sql_type = self._col_sql_type(col)
                    self._con.execute(f"ALTER TABLE ohlcv ADD COLUMN {col} {sql_type}")
                    logger.info(f"Added column {col} ({sql_type}) to ohlcv table")

        self._con.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_pair_ts
            ON ohlcv(pair, timestamp DESC)
        """)

        self._con.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key VARCHAR PRIMARY KEY,
                value VARCHAR NOT NULL
            )
        """)
        self._con.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('timeframe', ?)",
            [self.timeframe],
        )
        self._con.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('data_type', ?)",
            [self.data_type],
        )

        logger.info(f"Database initialized: {self.db_path} (data_type={self.data_type}, timeframe={self.timeframe})")

    def _migrate_legacy(self, existing_cols: set[str]) -> None:
        """Migrate from legacy schema (timeframe / open_time / quote_volume columns)."""
        logger.info("Migrating legacy schema -> fixed-timeframe table...")

        # Build target table with spot columns (legacy databases are always spot)
        spot_cols = resolve_columns("spot")
        col_defs = ",\n                    ".join(
            f"{c} {self._col_sql_type(c)}" for c in ["pair", "timestamp", *spot_cols]
        )
        self._con.execute(f"""
            CREATE TABLE IF NOT EXISTS ohlcv_new (
                {col_defs},
                PRIMARY KEY (pair, timestamp)
            )
        """)

        ts_col = "open_time" if "open_time" in existing_cols else "timestamp"
        vol_col = "quote_volume" if "quote_volume" in existing_cols else "volume"

        self._con.execute(f"""
            INSERT OR REPLACE INTO ohlcv_new (pair, timestamp, open, high, low, close, volume, trades)
            SELECT pair, {ts_col}, open, high, low, close, {vol_col}, trades
            FROM ohlcv
        """)

        self._con.execute("DROP TABLE ohlcv")
        self._con.execute("ALTER TABLE ohlcv_new RENAME TO ohlcv")
        logger.info("Legacy migration complete")

    # ── Insert ────────────────────────────────────────────────────────

    def insert_klines(self, pair: str, klines: list[dict]) -> None:
        """Upsert klines (INSERT OR REPLACE).

        Args:
            pair: Trading pair (e.g. "BTCUSDT").
            klines: List of kline dicts.  Must contain keys matching the
                store's data_type columns (timestamp, open, high, low, close,
                volume, and any extra like open_interest or funding_rate).
                ``trades`` is always optional.
        """
        if not klines:
            return

        n_cols = len(self._all_column_names)

        if len(klines) <= 10:
            placeholders = ", ".join(["?"] * n_cols)
            self._con.executemany(
                f"INSERT OR REPLACE INTO ohlcv VALUES ({placeholders})",
                [self._kline_to_row(pair, k) for k in klines],
            )
        else:
            data: dict[str, list] = {
                "pair": [pair] * len(klines),
                "timestamp": [normalize_ts(k["timestamp"]) for k in klines],
            }
            for col in self._columns:
                data[col] = [k.get(col) for k in klines]

            df = pl.DataFrame(data)
            self._con.register("temp_klines", df.to_arrow())
            self._con.execute("INSERT OR REPLACE INTO ohlcv SELECT * FROM temp_klines")
            self._con.unregister("temp_klines")

        logger.debug(f"Inserted {len(klines):,} rows for {pair}")

    def _kline_to_row(self, pair: str, k: dict) -> tuple:
        """Convert a kline dict to a positional tuple matching table columns."""
        return (pair, k["timestamp"], *tuple(k.get(c) for c in self._columns))

    # ── Queries ───────────────────────────────────────────────────────

    def get_time_bounds(self, pair: str) -> tuple[datetime | None, datetime | None]:
        """Get earliest and latest timestamps for a pair."""
        result = self._con.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv WHERE pair = ?",
            [pair],
        ).fetchone()
        return (result[0], result[1]) if result and result[0] else (None, None)

    def find_gaps(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        tf_minutes: int,
    ) -> list[tuple[datetime, datetime]]:
        """Find gaps in data coverage for a pair."""
        existing = self._con.execute(
            """
            SELECT timestamp FROM ohlcv
            WHERE pair = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """,
            [pair, start, end],
        ).fetchall()

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

    def load(
        self,
        pair: str,
        hours: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Load data for a single trading pair."""
        col_list = ", ".join(self._columns)
        query = f"""
            SELECT ? AS pair, timestamp, {col_list}
            FROM ohlcv WHERE pair = ?
        """
        params: list[object] = [pair, pair]

        if hours is not None:
            query += f" AND timestamp > NOW() - INTERVAL '{int(hours)}' HOUR"
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
        df = self._con.execute(query, params).pl()

        if "timestamp" in df.columns:
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

        return df

    def load_many(
        self,
        pairs: Iterable[str],
        hours: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Batch load for multiple pairs."""
        pairs = list(pairs)
        if not pairs:
            return pl.DataFrame(schema=polars_schema(self._columns))

        col_list = ", ".join(self._columns)
        placeholders = ", ".join(["?"] * len(pairs))
        query = f"""
            SELECT pair, timestamp, {col_list}
            FROM ohlcv WHERE pair IN ({placeholders})
        """
        params: list[object] = [*pairs]

        if hours is not None:
            query += f" AND timestamp > NOW() - INTERVAL '{int(hours)}' HOUR"
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
        df = self._con.execute(query, params).pl()

        if "timestamp" in df.columns:
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

        return df

    def load_many_pandas(
        self,
        pairs: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Load data for multiple pairs as Pandas DataFrame."""
        return self.load_many(pairs=pairs, start=start, end=end).to_pandas()

    # ── Stats / lifecycle ─────────────────────────────────────────────

    def get_stats(self) -> pl.DataFrame:
        """Get database statistics per pair."""
        return self._con.execute("""
            SELECT
                pair,
                COUNT(*) as rows,
                MIN(timestamp) as first_candle,
                MAX(timestamp) as last_candle,
                ROUND(SUM(volume), 2) as total_volume
            FROM ohlcv
            GROUP BY pair
            ORDER BY pair
        """).pl()

    def close(self) -> None:
        """Close database connection."""
        self._con.close()

    def to_raw_data(
        self,
        pairs: list[str],
        start: datetime,
        end: datetime,
        data_key: str | None = None,
    ) -> RawData:
        """Convert store data to RawData container.

        Loads data for specified pairs and date range, validates schema,
        and packages into immutable RawData container.

        Args:
            pairs: List of trading pairs to load.
            start: Start datetime (inclusive).
            end: End datetime (inclusive).
            data_key: Key for data in RawData.data dict.
                If None, uses store's data_type (e.g., "spot", "futures").

        Returns:
            RawData container with loaded and validated data.

        Raises:
            ValueError: If required columns missing or duplicates detected.

        Example:
            ```python
            store = DuckDbRawStore(db_path="data/spot.duckdb", data_type="spot")

            raw_data = store.to_raw_data(
                pairs=["BTCUSDT", "ETHUSDT"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 12, 31),
            )

            spot_df = raw_data["spot"]
            ```
        """
        key = data_key if data_key is not None else self.data_type

        df = self.load_many(pairs=pairs, start=start, end=end)

        # Validate required columns
        required = {"pair", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

        # Normalize timestamps
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None))

        # Check for duplicates
        dup_count = df.group_by(["pair", "timestamp"]).len().filter(pl.col("len") > 1)
        if dup_count.height > 0:
            dups = (
                df.join(dup_count.select(["pair", "timestamp"]), on=["pair", "timestamp"])
                .select(["pair", "timestamp"])
                .head(10)
            )
            raise ValueError(f"Duplicate (pair, timestamp) detected. Examples:\n{dups}")

        # Sort by (pair, timestamp)
        df = df.sort(["pair", "timestamp"])

        return RawData(
            datetime_start=start,
            datetime_end=end,
            pairs=pairs,
            data={key: df},
        )


# Register the same class for futures and perpetual data types.
# The factory passes data_type as a kwarg, overriding the default "spot".
default_registry.register(SfComponentType.RAW_DATA_STORE, "duckdb/futures", DuckDbRawStore, override=True)
default_registry.register(SfComponentType.RAW_DATA_STORE, "duckdb/perpetual", DuckDbRawStore, override=True)

# Backward-compatible alias.
DuckDbSpotStore = DuckDbRawStore
