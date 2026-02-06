from abc import ABC, abstractmethod

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, ClassVar
from signalflow.core import SfComponentType
import polars as pl
import pandas as pd


@dataclass
class RawDataStore(ABC):
    """Abstract base class for raw data storage backends.

    Defines the interface for loading historical market data from storage.
    Implementations provide specific storage backends (DuckDB, Parquet, etc.)
    while maintaining a consistent API.

    Key features:
        - Single and batch loading (load, load_many)
        - Flexible time filtering (hours, start/end)
        - Multi-format output (Polars, Pandas)
        - Resource management (close)

    Supported operations:
        - Load single pair with time filtering
        - Load multiple pairs efficiently (batch query)
        - Convert to Pandas for legacy compatibility
        - Cleanup resources on shutdown

    Attributes:
        component_type (ClassVar[SfComponentType]): Always RAW_DATA_STORE for registry.

    Example:
        ```python
        from signalflow.data.raw_store import DuckDbSpotStore
        from datetime import datetime
        from pathlib import Path

        # Create store instance
        store = DuckDbSpotStore(Path("data/binance_spot.duckdb"))

        try:
            # Load single pair
            btc_df = store.load(
                pair="BTCUSDT",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31)
            )

            # Load multiple pairs
            multi_df = store.load_many(
                pairs=["BTCUSDT", "ETHUSDT"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31)
            )

            # Load as Pandas
            pandas_df = store.load_many_pandas(
                pairs=["BTCUSDT"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31)
            )
        finally:
            store.close()

        # Use with context manager (if implemented)
        with DuckDbSpotStore(Path("data/binance_spot.duckdb")) as store:
            df = store.load("BTCUSDT", hours=24)
        ```

    Note:
        Subclasses must implement all abstract methods.
        Always call close() or use context manager to cleanup resources.
        Time filtering supports both relative (hours) and absolute (start/end).

    See Also:
        DuckDbSpotStore: DuckDB implementation for spot data.
        RawDataFactory: Factory for creating RawData from stores.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.RAW_DATA_STORE

    @abstractmethod
    def load(
        self, pair: str, hours: Optional[int] = None, start: Optional[datetime] = None, end: Optional[datetime] = None
    ) -> pl.DataFrame:
        """Load data for a single trading pair.

        Loads historical market data with flexible time filtering.
        Use either relative (hours) or absolute (start/end) time filtering.

        Args:
            pair (str): Trading pair (e.g., "BTCUSDT").
            hours (int | None): Load last N hours of data. Mutually exclusive with start/end.
            start (datetime | None): Start datetime (inclusive). Requires end parameter.
            end (datetime | None): End datetime (inclusive). Requires start parameter.

        Returns:
            pl.DataFrame: Market data as Polars DataFrame.
                Typically includes columns: pair, timestamp, open, high, low, close, volume.

        Raises:
            ValueError: If both hours and start/end are provided.
            ValueError: If start provided without end or vice versa.
            FileNotFoundError: If storage file/database doesn't exist.

        Example:
            ```python
            from datetime import datetime

            # Load last 24 hours
            recent_df = store.load("BTCUSDT", hours=24)

            # Load specific date range
            historical_df = store.load(
                "BTCUSDT",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31)
            )

            # Check loaded data
            print(f"Loaded {len(historical_df)} bars")
            print(f"Date range: {historical_df['timestamp'].min()} to {historical_df['timestamp'].max()}")
            ```

        Note:
            Implementation should handle timezone normalization.
            Returned DataFrame should be sorted by timestamp.
        """
        pass

    @abstractmethod
    def load_many(
        self,
        pairs: list[str],
        hours: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Load data for multiple trading pairs efficiently.

        Batch loading is more efficient than calling load() repeatedly.
        Returns combined DataFrame with all pairs.

        Args:
            pairs (list[str]): List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"]).
            hours (int | None): Load last N hours of data. Mutually exclusive with start/end.
            start (datetime | None): Start datetime (inclusive). Requires end parameter.
            end (datetime | None): End datetime (inclusive). Requires start parameter.

        Returns:
            pl.DataFrame: Combined market data for all pairs as Polars DataFrame.
                Includes pair column to distinguish between pairs.

        Raises:
            ValueError: If both hours and start/end are provided.
            ValueError: If start provided without end or vice versa.
            ValueError: If pairs list is empty.

        Example:
            ```python
            from datetime import datetime

            # Load multiple pairs
            multi_df = store.load_many(
                pairs=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31)
            )

            # Analyze by pair
            for pair in multi_df["pair"].unique():
                pair_df = multi_df.filter(pl.col("pair") == pair)
                print(f"{pair}: {len(pair_df)} bars")

            # Last 24 hours for monitoring
            recent_multi = store.load_many(
                pairs=["BTCUSDT", "ETHUSDT"],
                hours=24
            )
            ```

        Note:
            Returned DataFrame sorted by (pair, timestamp).
            More efficient than multiple load() calls due to batch query.
        """
        pass

    @abstractmethod
    def load_many_pandas(
        self, pairs: list[str], start: Optional[datetime] = None, end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Load data for multiple pairs as Pandas DataFrame.

        Convenience method for legacy code or libraries requiring Pandas.
        Typically converts from Polars internally.

        Args:
            pairs (list[str]): List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"]).
            start (datetime | None): Start datetime (inclusive). Requires end parameter.
            end (datetime | None): End datetime (inclusive). Requires start parameter.

        Returns:
            pd.DataFrame: Combined market data as Pandas DataFrame.

        Raises:
            ValueError: If start provided without end or vice versa.
            ValueError: If pairs list is empty.

        Example:
            ```python
            from datetime import datetime
            import pandas as pd

            # Load as Pandas
            df = store.load_many_pandas(
                pairs=["BTCUSDT", "ETHUSDT"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31)
            )

            # Use with pandas-ta
            import pandas_ta as ta
            df["rsi"] = ta.rsi(df["close"], length=14)

            # Use with legacy extractors
            class LegacyExtractor:
                def extract(self, df: pd.DataFrame) -> pd.DataFrame:
                    df["sma_20"] = df.groupby("pair")["close"].rolling(20).mean()
                    return df

            extractor = LegacyExtractor()
            df_with_features = extractor.extract(df)
            ```

        Note:
            Prefer load_many() with Polars for better performance.
            Use this only when Pandas is required.
            Timestamps normalized to timezone-naive datetime64[ns].
        """
        pass

    @abstractmethod
    def get_time_bounds(self, pair: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Return the earliest and latest timestamp for a pair.

        Args:
            pair (str): Trading pair (e.g., "BTCUSDT").

        Returns:
            tuple[Optional[datetime], Optional[datetime]]:
                (min_timestamp, max_timestamp) or (None, None) if no data exists.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close storage connection and cleanup resources.

        Releases database connections, file handles, or other resources.
        Should be called when store is no longer needed.

        Always call close() in a finally block or use context manager
        to ensure cleanup even on errors.

        Example:
            ```python
            # Manual cleanup
            store = DuckDbSpotStore(Path("data/binance.duckdb"))
            try:
                df = store.load("BTCUSDT", hours=24)
                # ... process data ...
            finally:
                store.close()

            # With context manager (if implemented)
            with DuckDbSpotStore(Path("data/binance.duckdb")) as store:
                df = store.load("BTCUSDT", hours=24)
                # ... process data ...
            # Automatically closed

            # In RawDataFactory
            store = DuckDbSpotStore(store_path)
            try:
                data = store.load_many(pairs, start, end)
                return RawData(data={"spot": data})
            finally:
                store.close()  # Always cleanup
            ```

        Note:
            Idempotent - safe to call multiple times.
            After close(), store should not be used for loading.
        """
        pass
