import pandas as pd
import polars as pl
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from signalflow.data.data_store import SpotStore
from typing import Iterator


@dataclass(frozen=True)
class RawData:
    datetime_start: datetime
    datetime_end: datetime
    pairs: list[str] = field(default_factory=list)
    data: dict[str, pl.DataFrame] = field(default_factory=dict)

    def get(self, key: str) -> pl.DataFrame:
        obj = self.data.get(key)
        if obj is None:
            return pl.DataFrame()
        if not isinstance(obj, pl.DataFrame):
            raise TypeError(f"Dataset '{key}' is not a polars.DataFrame: {type(obj)}")
        return obj
     
    def __getitem__(self, key: str) -> pl.DataFrame:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def keys(self) -> Iterator[str]:
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    @staticmethod
    def from_store(
        store_db_path: Path,
        pairs: list[str],
        start: datetime,
        end: datetime,
        data_types: list[str]|None = None, #TODO: RawData should support more than spot
    ) -> "RawData":

        store = SpotStore(store_db_path)
        try:
            spot = store.load_many(pairs=pairs, start=start, end=end)
            required = {"pair", "timestamp"}
            missing = required - set(spot.columns)
            if missing:
                raise ValueError(f"Spot df missing columns: {sorted(missing)}")

            if "timeframe" in spot.columns:
                spot = spot.drop(columns=["timeframe"])

            spot["timestamp"] = pd.to_datetime(spot["timestamp"], utc=True, errors="raise")

            if spot.duplicated(subset=["pair", "timestamp"]).any():
                dups = spot[spot.duplicated(subset=["pair", "timestamp"], keep=False)][
                    ["pair", "timestamp"]
                ].head(10)
                raise ValueError(f"Duplicate (pair,timestamp) detected. Examples:\n{dups}")

            spot = (
                spot
                .set_index(["pair", "timestamp"])
                .sort_index()
            )

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
    