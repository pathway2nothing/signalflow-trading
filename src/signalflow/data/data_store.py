import duckdb
import polars as pl

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Iterable
from loguru import logger
import pandas as pd


@dataclass
class SpotStore:
    """
    DuckDB storage for OHLCV data.
    Project timeframe is fixed -> timeframe is NOT stored in table and NOT present in APIs.
    """

    db_path: Path
    timeframe: str = "1m"  
    _con: duckdb.DuckDBPyConnection = field(init=False)

    def __post_init__(self) -> None:
        self._con = duckdb.connect(str(self.db_path))
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        existing = self._con.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'ohlcv'
        """).fetchall()
        existing_cols = {row[0] for row in existing}

        if existing_cols and ("timeframe" in existing_cols or "open_time" in existing_cols):
            logger.info("Migrating schema -> fixed-timeframe table (no timeframe column)...")

            self._con.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_new (
                    pair VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open DOUBLE NOT NULL,
                    high DOUBLE NOT NULL,
                    low DOUBLE NOT NULL,
                    close DOUBLE NOT NULL,
                    volume DOUBLE NOT NULL,
                    trades INTEGER,
                    PRIMARY KEY (pair, timestamp)
                )
            """)

            if "open_time" in existing_cols:

                self._con.execute("""
                    INSERT OR REPLACE INTO ohlcv_new
                    SELECT
                        pair,
                        open_time AS timestamp,
                        open, high, low, close,
                        quote_volume AS volume,
                        trades
                    FROM ohlcv
                """)
            else:
                self._con.execute("""
                    INSERT OR REPLACE INTO ohlcv_new
                    SELECT
                        pair,
                        timestamp,
                        open, high, low, close,
                        volume,
                        trades
                    FROM ohlcv
                """)

            self._con.execute("DROP TABLE ohlcv")
            self._con.execute("ALTER TABLE ohlcv_new RENAME TO ohlcv")
            logger.info("Migration complete")

        self._con.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                pair VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                trades INTEGER,
                PRIMARY KEY (pair, timestamp)
            )
        """)

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
        self._con.execute("""
            INSERT OR REPLACE INTO meta(key, value) VALUES ('timeframe', ?)
        """, [self.timeframe])

        logger.info(f"Database initialized: {self.db_path} (timeframe={self.timeframe})")

    def insert_klines(self, pair: str, klines: list[dict]) -> None:
        """Upsert klines. Kline dict must contain: timestamp, open, high, low, close, volume, trades."""
        if not klines:
            return

        if len(klines) <= 10:
            self._con.executemany(
                "INSERT OR REPLACE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        pair,
                        k["timestamp"],
                        k["open"],
                        k["high"],
                        k["low"],
                        k["close"],
                        k["volume"],
                        k.get("trades"),
                    )
                    for k in klines
                ],
            )
        else:
            df = pl.DataFrame(
                {
                    "pair": [pair] * len(klines),
                    "timestamp": [k["timestamp"] for k in klines],
                    "open": [k["open"] for k in klines],
                    "high": [k["high"] for k in klines],
                    "low": [k["low"] for k in klines],
                    "close": [k["close"] for k in klines],
                    "volume": [k["volume"] for k in klines],
                    "trades": [k.get("trades") for k in klines],
                }
            )
            self._con.register("temp_klines", df.to_arrow())
            self._con.execute("INSERT OR REPLACE INTO ohlcv SELECT * FROM temp_klines")
            self._con.unregister("temp_klines")

        logger.debug(f"Inserted {len(klines):,} rows for {pair}")

    def get_time_bounds(self, pair: str) -> tuple[Optional[datetime], Optional[datetime]]:
        result = self._con.execute("""
            SELECT MIN(timestamp), MAX(timestamp)
            FROM ohlcv
            WHERE pair = ?
        """, [pair]).fetchone()
        return (result[0], result[1]) if result and result[0] else (None, None)

    def find_gaps(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        tf_minutes: int,
    ) -> list[tuple[datetime, datetime]]:
        existing = self._con.execute("""
            SELECT timestamp
            FROM ohlcv
            WHERE pair = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """, [pair, start, end]).fetchall()

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
        """
        Output columns:
        pair, timestamp, open, high, low, close, volume, trades
        """
        query = """
            SELECT
                ? AS pair,
                timestamp, open, high, low, close, volume, trades
            FROM ohlcv
            WHERE pair = ?
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
        return self._con.execute(query, params).pl()
    
    def load_many_pandas(
        self,
        pairs: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        df_pl = self.load_many(pairs=pairs, start=start, end=end)
        return df_pl.to_pandas()

    def load_many(
        self,
        pairs: Iterable[str],
        hours: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """
        Batch load for multiple pairs.
        Output columns:
        pair, timestamp, open, high, low, close, volume, trades
        """
        pairs = list(pairs)
        if not pairs:
            return pl.DataFrame(
                schema={
                    "pair": pl.Utf8,
                    "timestamp": pl.Datetime,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                    "trades": pl.Int64,
                }
            )

        placeholders = ",".join(["?"] * len(pairs))
        query = f"""
            SELECT
                pair,
                timestamp, open, high, low, close, volume, trades
            FROM ohlcv
            WHERE pair IN ({placeholders})
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
        return self._con.execute(query, params).pl()

    def get_stats(self) -> pl.DataFrame:
        """DB contents stats (timeframe not stored, but meta has it)."""
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
        self._con.close()
