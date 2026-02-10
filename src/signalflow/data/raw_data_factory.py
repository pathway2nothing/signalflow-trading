"""Factory for creating RawData containers from storage backends.

Provides methods to load, validate, and package market data from
various storage backends into RawData containers for analysis.

Example:
    ```python
    from signalflow.data import RawDataFactory, StoreFactory
    from datetime import datetime

    # From multiple stores
    spot = StoreFactory.create_raw_store("duckdb", "spot", db_path="spot.duckdb")
    futures = StoreFactory.create_raw_store("duckdb", "futures", db_path="fut.duckdb")

    raw_data = RawDataFactory.from_stores(
        stores=[spot, futures],
        pairs=["BTCUSDT"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
    )

    # Access by type
    spot_df = raw_data["spot"]
    futures_df = raw_data["futures"]
    ```
"""

from datetime import datetime
from pathlib import Path
from typing import Sequence

import polars as pl

from signalflow.core import RawData
from signalflow.data.raw_store import DuckDbSpotStore
from signalflow.data.raw_store.base import RawDataStore


class RawDataFactory:
    """Factory for creating RawData instances from various sources.

    Provides static methods to construct RawData objects from different
    storage backends (DuckDB, Parquet, etc.) with proper validation
    and schema normalization.

    Key features:
        - Automatic schema validation
        - Duplicate detection
        - Timezone normalization
        - Column cleanup (remove unnecessary columns)
        - Proper sorting by (pair, timestamp)

    Example:
        ```python
        from signalflow.data import RawDataFactory
        from pathlib import Path
        from datetime import datetime

        # Load spot data from DuckDB
        raw_data = RawDataFactory.from_duckdb_spot_store(
            spot_store_path=Path("data/binance_spot.duckdb"),
            pairs=["BTCUSDT", "ETHUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
            data_types=["spot"]
        )

        # Access loaded data
        spot_df = raw_data["spot"]
        print(f"Loaded {len(spot_df)} bars")
        print(f"Pairs: {raw_data.pairs}")
        print(f"Date range: {raw_data.datetime_start} to {raw_data.datetime_end}")

        # Use in detector
        from signalflow.detector import SmaCrossSignalDetector

        detector = SmaCrossSignalDetector(fast_window=10, slow_window=20)
        signals = detector.detect(raw_data)
        ```

    See Also:
        RawData: Immutable container for raw market data.
        DuckDbSpotStore: DuckDB storage backend for spot data.
    """

    @staticmethod
    def from_stores(
        stores: Sequence[RawDataStore],
        pairs: list[str],
        start: datetime,
        end: datetime,
    ) -> RawData:
        """Create RawData from multiple stores.

        Loads data from each store and merges into a single RawData container.
        Each store's data_type becomes the key in RawData.data dict.

        Args:
            stores: Sequence of RawDataStore instances.
            pairs: List of trading pairs to load.
            start: Start datetime (inclusive).
            end: End datetime (inclusive).

        Returns:
            RawData: Container with merged data from all stores.

        Raises:
            ValueError: If stores have duplicate data_type keys.

        Example:
            ```python
            from signalflow.data import RawDataFactory, StoreFactory

            # Create stores
            spot_store = StoreFactory.create_raw_store(
                backend="duckdb",
                data_type="spot",
                db_path="data/spot.duckdb",
            )
            futures_store = StoreFactory.create_raw_store(
                backend="duckdb",
                data_type="futures",
                db_path="data/futures.duckdb",
            )

            # Load from multiple stores
            raw_data = RawDataFactory.from_stores(
                stores=[spot_store, futures_store],
                pairs=["BTCUSDT", "ETHUSDT"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 12, 31),
            )

            # Access data by type
            spot_df = raw_data["spot"]
            futures_df = raw_data["futures"]
            ```
        """
        if not stores:
            return RawData(
                datetime_start=start,
                datetime_end=end,
                pairs=pairs,
                data={},
            )

        merged_data: dict[str, pl.DataFrame] = {}

        for store in stores:
            raw = store.to_raw_data(pairs=pairs, start=start, end=end)
            for key, df in raw.data.items():
                if key in merged_data:
                    raise ValueError(f"Duplicate data key '{key}' from multiple stores")
                merged_data[key] = df

        return RawData(
            datetime_start=start,
            datetime_end=end,
            pairs=pairs,
            data=merged_data,
        )

    @staticmethod
    def from_duckdb_spot_store(
        spot_store_path: Path,
        pairs: list[str],
        start: datetime,
        end: datetime,
        data_types: list[str] | None = None,
    ) -> RawData:
        """Create RawData from DuckDB spot store.

        Loads spot trading data from DuckDB storage with validation,
        deduplication checks, and schema normalization.

        Processing steps:
            1. Load data from DuckDB for specified pairs and date range
            2. Validate required columns (pair, timestamp)
            3. Remove unnecessary columns (timeframe)
            4. Normalize timestamps (microseconds, timezone-naive)
            5. Check for duplicates (pair, timestamp)
            6. Sort by (pair, timestamp)
            7. Package into RawData container

        Args:
            spot_store_path (Path): Path to DuckDB file.
            pairs (list[str]): List of trading pairs to load (e.g., ["BTCUSDT", "ETHUSDT"]).
            start (datetime): Start datetime (inclusive).
            end (datetime): End datetime (inclusive).
            data_types (list[str] | None): Data types to load. Default: None.
                Currently supports: ["spot"].

        Returns:
            RawData: Immutable container with loaded and validated data.

        Raises:
            ValueError: If required columns missing (pair, timestamp).
            ValueError: If duplicate (pair, timestamp) combinations detected.

        Example:
            ```python
            from pathlib import Path
            from datetime import datetime
            from signalflow.data import RawDataFactory

            # Load single pair
            raw_data = RawDataFactory.from_duckdb_spot_store(
                spot_store_path=Path("data/binance.duckdb"),
                pairs=["BTCUSDT"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31),
                data_types=["spot"]
            )

            # Load multiple pairs
            raw_data = RawDataFactory.from_duckdb_spot_store(
                spot_store_path=Path("data/binance.duckdb"),
                pairs=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 12, 31),
                data_types=["spot"]
            )

            # Check loaded data
            spot_df = raw_data["spot"]
            print(f"Shape: {spot_df.shape}")
            print(f"Columns: {spot_df.columns}")
            print(f"Pairs: {spot_df['pair'].unique().to_list()}")

            # Verify no duplicates
            dup_check = (
                spot_df.group_by(["pair", "timestamp"])
                .len()
                .filter(pl.col("len") > 1)
            )
            assert dup_check.is_empty()

            # Use in pipeline
            from signalflow.core import RawDataView
            view = RawDataView(raw=raw_data)
            spot_pandas = view.to_pandas("spot")
            ```

        Example:
            ```python
            # Handle missing data gracefully
            try:
                raw_data = RawDataFactory.from_duckdb_spot_store(
                    spot_store_path=Path("data/binance.duckdb"),
                    pairs=["BTCUSDT"],
                    start=datetime(2024, 1, 1),
                    end=datetime(2024, 1, 31),
                    data_types=["spot"]
                )
            except ValueError as e:
                if "missing columns" in str(e):
                    print("Data schema invalid")
                elif "Duplicate" in str(e):
                    print("Data contains duplicates")
                raise

            # Validate date range
            assert raw_data.datetime_start == datetime(2024, 1, 1)
            assert raw_data.datetime_end == datetime(2024, 1, 31)

            # Check data quality
            spot_df = raw_data["spot"]

            # Verify timestamps are sorted
            assert spot_df["timestamp"].is_sorted()

            # Verify timezone-naive
            assert spot_df["timestamp"].dtype == pl.Datetime("us")

            # Verify no nulls in key columns
            assert spot_df["pair"].null_count() == 0
            assert spot_df["timestamp"].null_count() == 0
            ```

        Note:
            Store connection is automatically closed via finally block.
            Timestamps are normalized to timezone-naive microseconds.
            Duplicate detection shows first 10 examples if found.
            All data sorted by (pair, timestamp) for consistent ordering.
        """
        data: dict[str, pl.DataFrame] = {}
        store = DuckDbSpotStore(spot_store_path)
        try:
            if "spot" in data_types:
                spot = store.load_many(pairs=pairs, start=start, end=end)

                required = {"pair", "timestamp"}
                missing = required - set(spot.columns)
                if missing:
                    raise ValueError(f"Spot df missing columns: {sorted(missing)}")

                if "timeframe" in spot.columns:
                    spot = spot.drop("timeframe")

                spot = spot.with_columns(pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None))

                dup_count = spot.group_by(["pair", "timestamp"]).len().filter(pl.col("len") > 1)
                if dup_count.height > 0:
                    dups = (
                        spot.join(
                            dup_count.select(["pair", "timestamp"]),
                            on=["pair", "timestamp"],
                        )
                        .select(["pair", "timestamp"])
                        .head(10)
                    )
                    raise ValueError(f"Duplicate (pair, timestamp) detected. Examples:\n{dups}")

                spot = spot.sort(["pair", "timestamp"])
                data["spot"] = spot

            return RawData(
                datetime_start=start,
                datetime_end=end,
                pairs=pairs,
                data=data,
            )
        finally:
            store.close()
