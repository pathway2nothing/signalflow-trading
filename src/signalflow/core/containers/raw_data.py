import pandas as pd
import polars as pl

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator


@dataclass(frozen=True)
class RawData:
    """Immutable container for raw market data.

    Acts as a unified in-memory bundle for multiple raw datasets
    (e.g. spot prices, funding, trades, orderbook, signals).

    Design principles:
        - Canonical storage is dataset-based (dictionary by name)
        - Datasets accessed via string keys (e.g. raw_data["spot"])
        - No business logic or transformations
        - Immutability ensures reproducibility in pipelines

    Attributes:
        datetime_start (datetime): Start datetime of the data snapshot.
        datetime_end (datetime): End datetime of the data snapshot.
        pairs (list[str]): List of trading pairs in the snapshot.
        data (dict[str, pl.DataFrame]): Dictionary of datasets keyed by name.

    Example:
        ```python
        from signalflow.core import RawData
        import polars as pl
        from datetime import datetime

        # Create RawData with spot data
        raw_data = RawData(
            datetime_start=datetime(2024, 1, 1),
            datetime_end=datetime(2024, 12, 31),
            pairs=["BTCUSDT", "ETHUSDT"],
            data={
                "spot": spot_dataframe,
                "signals": signals_dataframe,
            }
        )

        # Access datasets
        spot_df = raw_data["spot"]
        signals_df = raw_data.get("signals")

        # Check if dataset exists
        if "spot" in raw_data:
            print("Spot data available")
        ```

    Note:
        Dataset schemas are defined by convention, not enforced.
        Views (pandas/polars) should be handled by RawDataView wrapper.
    """

    datetime_start: datetime
    datetime_end: datetime
    pairs: list[str] = field(default_factory=list)
    data: dict[str, pl.DataFrame] = field(default_factory=dict)

    def get(self, key: str) -> pl.DataFrame:
        """Get dataset by key.

        Args:
            key (str): Dataset name (e.g. "spot", "signals").

        Returns:
            pl.DataFrame: Polars DataFrame if exists, empty DataFrame otherwise.

        Raises:
            TypeError: If dataset exists but is not a Polars DataFrame.

        Example:
            ```python
            spot_df = raw_data.get("spot")
            
            # Returns empty DataFrame if key doesn't exist
            missing_df = raw_data.get("nonexistent")
            assert missing_df.is_empty()
            ```
        """
        obj = self.data.get(key)
        if obj is None:
            return pl.DataFrame()
        if not isinstance(obj, pl.DataFrame):
            raise TypeError(
                f"Dataset '{key}' is not a polars.DataFrame: {type(obj)}"
            )
        return obj

    def __getitem__(self, key: str) -> pl.DataFrame:
        """Dictionary-style access to datasets.

        Args:
            key (str): Dataset name.

        Returns:
            pl.DataFrame: Dataset as Polars DataFrame.

        Example:
            ```python
            spot_df = raw_data["spot"]
            ```
        """
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if dataset exists.

        Args:
            key (str): Dataset name to check.

        Returns:
            bool: True if dataset exists, False otherwise.

        Example:
            ```python
            if "spot" in raw_data:
                process_spot_data(raw_data["spot"])
            ```
        """
        return key in self.data

    def keys(self) -> Iterator[str]:
        """Return available dataset keys.

        Returns:
            Iterator[str]: Iterator over dataset names.

        Example:
            ```python
            for key in raw_data.keys():
                print(f"Dataset: {key}")
            ```
        """
        return self.data.keys()

    def items(self):
        """Return (key, dataset) pairs.

        Returns:
            Iterator: Iterator over (key, DataFrame) tuples.

        Example:
            ```python
            for name, df in raw_data.items():
                print(f"{name}: {df.shape}")
            ```
        """
        return self.data.items()

    def values(self):
        """Return dataset values.

        Returns:
            Iterator: Iterator over DataFrames.

        Example:
            ```python
            for df in raw_data.values():
                print(df.columns)
            ```
        """
        return self.data.values()