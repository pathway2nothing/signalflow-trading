import pandas as pd
import polars as pl

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

from signalflow.data.data_store import SpotStore


@dataclass(frozen=True)
class RawData:
    """Immutable container for raw market data.

    RawData acts as a unified in-memory bundle for multiple raw datasets
    (e.g. spot prices, funding, trades, orderbook, signals).

    Design principles:
        - Canonical storage is dataset-based (dictionary by name).
        - Datasets are accessed via string keys (e.g. raw_data["spot"]).
        - RawData itself contains no business logic or transformations.
        - Immutability ensures reproducibility and safe reuse in pipelines.

    Attributes:
        datetime_start: Start datetime of the data snapshot.
        datetime_end: End datetime of the data snapshot.
        pairs: List of trading pairs included in the snapshot.
        data: Dictionary of datasets keyed by dataset name
              (e.g. {"spot": DataFrame}).

    Notes:
        - Dataset schemas are defined by convention, not enforced here.
        - Views (pandas/polars) should live outside RawData.
    """

    datetime_start: datetime
    datetime_end: datetime
    pairs: list[str] = field(default_factory=list)
    data: dict[str, pl.DataFrame] = field(default_factory=dict)

    def get(self, key: str) -> pl.DataFrame:
        """Get dataset by key.

        Args:
            key: Dataset name (e.g. "spot", "signals").

        Returns:
            Polars DataFrame if dataset exists, otherwise empty DataFrame.

        Raises:
            TypeError: If dataset exists but is not a Polars DataFrame.
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

        Example:
            raw_data["spot"]
        """
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if dataset exists."""
        return key in self.data

    def keys(self) -> Iterator[str]:
        """Return available dataset keys."""
        return self.data.keys()

    def items(self):
        """Return (key, dataset) pairs."""
        return self.data.items()

    def values(self):
        """Return dataset values."""
        return self.data.values()


    @staticmethod
    def from_store(
        store_db_path: Path,
        pairs: list[str],
        start: datetime,
        end: datetime,
        data_types: list[str] | None = None,
    ) -> "RawData":
        """Build RawData snapshot from persistent storage.

        This method loads raw datasets from DuckDB storage via SpotStore
        and assembles them into a single RawData container.

        Currently supported datasets:
            - "spot": OHLCV market data

        Args:
            store_db_path: Path to DuckDB database.
            pairs: Trading pairs to load.
            start: Start datetime.
            end: End datetime.
            data_types: Optional list of dataset names to load.
                        If None, default datasets are loaded.

        Returns:
            RawData instance containing requested datasets.

        Raises:
            ValueError: If required columns are missing or duplicates detected.
        """
        store = SpotStore(store_db_path)
        try:
            spot = store.load_many(pairs=pairs, start=start, end=end)

            required = {"pair", "timestamp"}
            missing = required - set(spot.columns)
            if missing:
                raise ValueError(f"Spot df missing columns: {sorted(missing)}")

            if "timeframe" in spot.columns:
                spot = spot.drop("timeframe")

            spot = spot.with_columns(
                pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
            )

            dup_count = spot.group_by(["pair", "timestamp"]).len().filter(
                pl.col("len") > 1
            )
            if dup_count.height > 0:
                dups = (
                    spot.join(dup_count.select(["pair", "timestamp"]), on=["pair", "timestamp"])
                    .select(["pair", "timestamp"])
                    .head(10)
                )
                raise ValueError(
                    f"Duplicate (pair, timestamp) detected. Examples:\n{dups}"
                )

            spot = spot.sort(["pair", "timestamp"])

            return RawData(
                datetime_start=start,
                datetime_end=end,
                pairs=pairs,
                data={
                    "spot": spot,
                },
            )
        finally:
            store.close()
