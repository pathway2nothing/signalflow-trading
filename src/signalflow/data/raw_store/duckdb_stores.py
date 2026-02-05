# IMPORTANT
import duckdb
import polars as pl

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Iterable
from loguru import logger
import pandas as pd

from signalflow.core import sf_component
from signalflow.data.raw_store.base import RawDataStore


@dataclass
@sf_component(name="duckdb/spot")
class DuckDbSpotStore(RawDataStore):
    """DuckDB storage backend for OHLCV spot data.

    Provides efficient storage and retrieval of candlestick (OHLCV) data
    using DuckDB as the backend. Designed for fixed-timeframe storage
    (timeframe not stored per-row, configured at database level).

    Key features:
        - Automatic schema migration from legacy formats
        - Efficient batch inserts with upsert (INSERT OR REPLACE)
        - Gap detection for data continuity checks
        - Multi-pair batch loading
        - Indexed queries for fast retrieval

    Schema:
        - pair (VARCHAR): Trading pair
        - timestamp (TIMESTAMP): Bar open time (timezone-naive)
        - open, high, low, close (DOUBLE): OHLC prices
        - volume (DOUBLE): Trading volume
        - trades (INTEGER): Number of trades

    Attributes:
        db_path (Path): Path to DuckDB file.
        timeframe (str): Fixed timeframe for all data (e.g., "1m", "5m"). Default: "1m".
        _con (duckdb.DuckDBPyConnection): Database connection (initialized in __post_init__).

    Example:
        ```python
        from signalflow.data.raw_store import DuckDbSpotStore
        from pathlib import Path
        from datetime import datetime

        # Create store
        store = DuckDbSpotStore(
            db_path=Path("data/binance_spot.duckdb"),
            timeframe="1m"
        )

        try:
            # Insert data
            klines = [
                {
                    "timestamp": datetime(2024, 1, 1, 10, 0),
                    "open": 45000.0,
                    "high": 45100.0,
                    "low": 44900.0,
                    "close": 45050.0,
                    "volume": 100.5,
                    "trades": 150
                }
            ]
            store.insert_klines("BTCUSDT", klines)

            # Load data
            df = store.load("BTCUSDT", hours=24)

            # Check data bounds
            min_ts, max_ts = store.get_time_bounds("BTCUSDT")
            print(f"Data range: {min_ts} to {max_ts}")

            # Get statistics
            stats = store.get_stats()
            print(stats)

        finally:
            store.close()
        ```

    Note:
        Timeframe is fixed per database, not per row.
        Automatically migrates from legacy schema (open_time, timeframe columns).
        Always call close() to cleanup database connection.

    See Also:
        RawDataStore: Base class with interface definition.
        RawDataFactory: Factory for creating RawData from stores.
    """

    db_path: Path
    timeframe: str = "1m"
    _con: duckdb.DuckDBPyConnection = field(init=False)

    def __post_init__(self) -> None:
        """Initialize database connection and ensure schema."""
        self._con = duckdb.connect(str(self.db_path))
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create tables and migrate from legacy schema if needed.

        Automatically detects and migrates from:
            - Legacy schema with 'timeframe' column
            - Legacy schema with 'open_time' instead of 'timestamp'
            - Legacy schema with 'quote_volume' instead of 'volume'

        Creates:
            - ohlcv table with PRIMARY KEY (pair, timestamp)
            - Index on (pair, timestamp DESC) for fast queries
            - meta table for storing timeframe configuration
        """
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
        self._con.execute(
            """
            INSERT OR REPLACE INTO meta(key, value) VALUES ('timeframe', ?)
        """,
            [self.timeframe],
        )

        logger.info(f"Database initialized: {self.db_path} (timeframe={self.timeframe})")

    def insert_klines(self, pair: str, klines: list[dict]) -> None:
        """Upsert klines (INSERT OR REPLACE).

        Efficient batch insertion with automatic upsert on (pair, timestamp) conflict.
        Uses Arrow-based bulk insert for >10 rows for better performance.

        Timestamp normalization:
            - Removes timezone info
            - Rounds to minute (removes seconds/microseconds)
            - If second != 0, rounds up to next minute

        Args:
            pair (str): Trading pair (e.g., "BTCUSDT").
            klines (list[dict]): List of kline dictionaries. Each must contain:
                - timestamp (datetime): Bar open time
                - open (float): Open price
                - high (float): High price
                - low (float): Low price
                - close (float): Close price
                - volume (float): Trading volume
                - trades (int, optional): Number of trades

        Example:
            ```python
            from datetime import datetime

            # Insert single kline
            store.insert_klines("BTCUSDT", [
                {
                    "timestamp": datetime(2024, 1, 1, 10, 0),
                    "open": 45000.0,
                    "high": 45100.0,
                    "low": 44900.0,
                    "close": 45050.0,
                    "volume": 100.5,
                    "trades": 150
                }
            ])

            # Batch insert (efficient for >10 rows)
            klines = [
                {
                    "timestamp": datetime(2024, 1, 1, 10, i),
                    "open": 45000.0 + i,
                    "high": 45100.0 + i,
                    "low": 44900.0 + i,
                    "close": 45050.0 + i,
                    "volume": 100.0,
                    "trades": 150
                }
                for i in range(100)
            ]
            store.insert_klines("BTCUSDT", klines)

            # Upsert - updates existing rows
            store.insert_klines("BTCUSDT", [
                {
                    "timestamp": datetime(2024, 1, 1, 10, 0),
                    "open": 45010.0,  # Updated price
                    "high": 45110.0,
                    "low": 44910.0,
                    "close": 45060.0,
                    "volume": 101.0,
                    "trades": 152
                }
            ])
            ```

        Note:
            Empty klines list is silently ignored.
            Uses executemany for ≤10 rows, Arrow bulk insert for >10 rows.
            Automatically logs insert count at debug level.
        """
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
                    "timestamp": [
                        k["timestamp"].replace(tzinfo=None).replace(second=0, microsecond=0) + timedelta(minutes=1)
                        if k["timestamp"].second != 0 or k["timestamp"].microsecond != 0
                        else k["timestamp"].replace(tzinfo=None)
                        for k in klines
                    ],
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
        """Get earliest and latest timestamps for a pair.

        Useful for:
            - Checking data availability
            - Planning data updates
            - Validating date ranges

        Args:
            pair (str): Trading pair (e.g., "BTCUSDT").

        Returns:
            tuple[datetime | None, datetime | None]: (min_timestamp, max_timestamp).
                Both None if no data exists for pair.

        Example:
            ```python
            # Check data availability
            min_ts, max_ts = store.get_time_bounds("BTCUSDT")

            if min_ts and max_ts:
                print(f"Data available: {min_ts} to {max_ts}")
                days = (max_ts - min_ts).days
                print(f"Total days: {days}")
            else:
                print("No data available")

            # Plan incremental update
            _, max_ts = store.get_time_bounds("BTCUSDT")
            if max_ts:
                # Fetch data from max_ts to now
                fetch_data(start=max_ts, end=datetime.now())
            ```
        """
        result = self._con.execute(
            """
            SELECT MIN(timestamp), MAX(timestamp)
            FROM ohlcv
            WHERE pair = ?
        """,
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
        """Find gaps in data coverage for a pair.

        Detects missing bars in expected continuous sequence based on timeframe.
        Useful for data quality checks and incremental backfilling.

        Args:
            pair (str): Trading pair (e.g., "BTCUSDT").
            start (datetime): Start of expected range.
            end (datetime): End of expected range.
            tf_minutes (int): Timeframe in minutes (e.g., 1 for 1m, 5 for 5m).

        Returns:
            list[tuple[datetime, datetime]]: List of (gap_start, gap_end) tuples.
                Empty list if no gaps found.

        Example:
            ```python
            from datetime import datetime

            # Check for gaps in January 2024
            gaps = store.find_gaps(
                pair="BTCUSDT",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31),
                tf_minutes=1
            )

            if gaps:
                print(f"Found {len(gaps)} gaps:")
                for gap_start, gap_end in gaps:
                    duration = gap_end - gap_start
                    print(f"  {gap_start} to {gap_end} ({duration})")

                    # Backfill gaps
                    backfill_data(pair="BTCUSDT", start=gap_start, end=gap_end)
            else:
                print("No gaps found - data is continuous")

            # Data quality report
            gaps = store.find_gaps("BTCUSDT", start, end, tf_minutes=1)
            total_expected = int((end - start).total_seconds() / 60)
            total_missing = sum((g[1] - g[0]).total_seconds() / 60 for g in gaps)
            coverage = (1 - total_missing / total_expected) * 100
            print(f"Data coverage: {coverage:.2f}%")
            ```

        Note:
            Returns full range [(start, end)] if no data exists.
            Computationally expensive for large date ranges - use sparingly.
        """
        existing = self._con.execute(
            """
            SELECT timestamp
            FROM ohlcv
            WHERE pair = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """,
            [pair, start, end],
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
        """Load data for a single trading pair.

        Output columns: pair, timestamp, open, high, low, close, volume, trades

        Args:
            pair (str): Trading pair (e.g., "BTCUSDT").
            hours (int | None): Load last N hours of data. Mutually exclusive with start/end.
            start (datetime | None): Start datetime (inclusive). Requires end parameter.
            end (datetime | None): End datetime (inclusive). Requires start parameter.

        Returns:
            pl.DataFrame: OHLCV data sorted by timestamp. Timezone-naive timestamps.

        Example:
            ```python
            # Load last 24 hours
            df = store.load("BTCUSDT", hours=24)

            # Load specific range
            df = store.load(
                "BTCUSDT",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31)
            )

            # Check loaded data
            print(df.select(["timestamp", "close"]).head())
            ```
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
        """Load data for multiple pairs as Pandas DataFrame.

        Convenience wrapper around load_many() for Pandas compatibility.

        Args:
            pairs (list[str]): List of trading pairs.
            start (datetime | None): Start datetime (inclusive).
            end (datetime | None): End datetime (inclusive).

        Returns:
            pd.DataFrame: Combined OHLCV data as Pandas DataFrame.

        Example:
            ```python
            df = store.load_many_pandas(
                pairs=["BTCUSDT", "ETHUSDT"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31)
            )

            # Use with pandas
            df["returns"] = df.groupby("pair")["close"].pct_change()
            ```
        """
        df_pl = self.load_many(pairs=pairs, start=start, end=end)
        return df_pl.to_pandas()

    def load_many(
        self,
        pairs: Iterable[str],
        hours: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Batch load for multiple pairs.

        Output columns: pair, timestamp, open, high, low, close, volume, trades

        More efficient than multiple load() calls due to single query.

        Args:
            pairs (Iterable[str]): Trading pairs to load.
            hours (int | None): Load last N hours of data.
            start (datetime | None): Start datetime (inclusive).
            end (datetime | None): End datetime (inclusive).

        Returns:
            pl.DataFrame: Combined OHLCV data sorted by (pair, timestamp).
                Empty DataFrame with correct schema if no pairs provided.

        Example:
            ```python
            # Load multiple pairs
            df = store.load_many(
                pairs=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31)
            )

            # Analyze by pair
            for pair in df["pair"].unique():
                pair_df = df.filter(pl.col("pair") == pair)
                print(f"{pair}: {len(pair_df)} bars")
            ```
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

        df = self._con.execute(query, params).pl()

        if "timestamp" in df.columns:
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

        return df

    def get_stats(self) -> pl.DataFrame:
        """Get database statistics per pair.

        Returns summary statistics for all pairs in database.

        Returns:
            pl.DataFrame: Statistics with columns:
                - pair (str): Trading pair
                - rows (int): Number of bars
                - first_candle (datetime): Earliest timestamp
                - last_candle (datetime): Latest timestamp
                - total_volume (float): Sum of volume

        Example:
            ```python
            # Get overview
            stats = store.get_stats()
            print(stats)

            # Check coverage
            for row in stats.iter_rows(named=True):
                pair = row["pair"]
                days = (row["last_candle"] - row["first_candle"]).days
                print(f"{pair}: {row['rows']:,} bars over {days} days")

            # Identify incomplete data
            min_rows = stats["rows"].min()
            incomplete = stats.filter(pl.col("rows") < min_rows * 0.9)
            print(f"Pairs with <90% coverage: {incomplete['pair'].to_list()}")
            ```

        Note:
            Timeframe not included in output (stored in meta table).
            Sorted alphabetically by pair.
        """
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
        """Close database connection and cleanup resources.

        Always call in finally block or use context manager to ensure cleanup.

        Example:
            ```python
            store = DuckDbSpotStore(Path("data/binance.duckdb"))
            try:
                df = store.load("BTCUSDT", hours=24)
            finally:
                store.close()
            ```
        """
        self._con.close()
