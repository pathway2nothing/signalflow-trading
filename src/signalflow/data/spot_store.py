import duckdb
import polars as pl
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from loguru import logger


@dataclass
class SpotStore:
    """DuckDB storage for OHLCV data."""
    
    db_path: Path
    _con: duckdb.DuckDBPyConnection = field(init=False)
    
    def __post_init__(self):
        self._con = duckdb.connect(str(self.db_path))
        self._ensure_tables()
    
    def _ensure_tables(self):
        existing = self._con.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'ohlcv'
        """).fetchall()
        existing_cols = {row[0] for row in existing}
        
        if existing_cols and "open_time" in existing_cols:
            logger.info("Migrating old schema...")
            self._con.execute("""
                CREATE TABLE ohlcv_new (
                    symbol VARCHAR NOT NULL,
                    timeframe VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open DOUBLE NOT NULL,
                    high DOUBLE NOT NULL,
                    low DOUBLE NOT NULL,
                    close DOUBLE NOT NULL,
                    volume DOUBLE NOT NULL,
                    trades INTEGER,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )
            """)
            self._con.execute("""
                INSERT INTO ohlcv_new 
                SELECT symbol, interval, open_time, open, high, low, close, quote_volume, trades
                FROM ohlcv
            """)
            self._con.execute("DROP TABLE ohlcv")
            self._con.execute("ALTER TABLE ohlcv_new RENAME TO ohlcv")
            logger.info("Migration complete")
        
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                trades INTEGER,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        """)
        
        self._con.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time 
            ON ohlcv(symbol, timeframe, timestamp DESC)
        """)
        
        logger.info(f"Database initialized: {self.db_path}")
    
    def insert_klines(self, symbol: str, timeframe: str, klines: list[dict]):
        """Inserts klines with upsert logic.
        
        Args:
            symbol: Trading pair.
            timeframe: Candle interval.
            klines: List of OHLCV dictionaries.
        """
        if not klines:
            return
        
        if len(klines) <= 10:
            self._con.executemany(
                "INSERT OR REPLACE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (symbol, timeframe, k["timestamp"], k["open"], k["high"], 
                     k["low"], k["close"], k["volume"], k["trades"])
                    for k in klines
                ]
            )
        else:
            df = pl.DataFrame({
                "symbol": [symbol] * len(klines),
                "timeframe": [timeframe] * len(klines),
                "timestamp": [k["timestamp"] for k in klines],
                "open": [k["open"] for k in klines],
                "high": [k["high"] for k in klines],
                "low": [k["low"] for k in klines],
                "close": [k["close"] for k in klines],
                "volume": [k["volume"] for k in klines],
                "trades": [k["trades"] for k in klines],
            })
            
            self._con.register("temp_klines", df.to_arrow())
            self._con.execute("INSERT OR REPLACE INTO ohlcv SELECT * FROM temp_klines")
            self._con.unregister("temp_klines")
        
        logger.debug(f"Inserted {len(klines):,} rows for {symbol} ({timeframe})")
    
    def get_time_bounds(self, symbol: str, timeframe: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Returns min and max timestamps for a symbol.
        
        Args:
            symbol: Trading pair.
            timeframe: Candle interval.
        
        Returns:
            Tuple of (min_timestamp, max_timestamp) or (None, None).
        """
        result = self._con.execute("""
            SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv 
            WHERE symbol = ? AND timeframe = ?
        """, [symbol, timeframe]).fetchone()
        return (result[0], result[1]) if result and result[0] else (None, None)
    
    def find_gaps(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        tf_minutes: int
    ) -> list[tuple[datetime, datetime]]:
        """Finds missing data gaps in the specified range.
        
        Args:
            symbol: Trading pair.
            timeframe: Candle interval.
            start: Start of the range.
            end: End of the range.
            tf_minutes: Timeframe duration in minutes.
        
        Returns:
            List of (gap_start, gap_end) tuples.
        """
        existing = self._con.execute("""
            SELECT timestamp FROM ohlcv 
            WHERE symbol = ? AND timeframe = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """, [symbol, timeframe, start, end]).fetchall()
        
        if not existing:
            return [(start, end)]
        
        existing_times = {row[0] for row in existing}
        gaps = []
        gap_start = None
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
        symbol: str,
        timeframe: str = "1m",
        hours: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Loads OHLCV data into a Polars DataFrame.
        
        Args:
            symbol: Trading pair.
            timeframe: Candle interval.
            hours: Load last N hours (alternative to start/end).
            start: Start of the range.
            end: End of the range.
        
        Returns:
            Polars DataFrame with columns: timestamp, open, high, low, close, volume, trades.
        """
        query = """
            SELECT timestamp, open, high, low, close, volume, trades
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]
        
        if hours:
            query += f" AND timestamp > NOW() - INTERVAL '{hours}' HOUR"
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
    
    def get_stats(self) -> pl.DataFrame:
        """Returns statistics for all pairs in database.
        
        Returns:
            DataFrame with row counts and time bounds per symbol.
        """
        return self._con.execute("""
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as rows,
                MIN(timestamp) as first_candle,
                MAX(timestamp) as last_candle,
                ROUND(SUM(volume), 2) as total_volume
            FROM ohlcv
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """).pl()
    
    def close(self):
        """Closes the database connection."""
        self._con.close()

