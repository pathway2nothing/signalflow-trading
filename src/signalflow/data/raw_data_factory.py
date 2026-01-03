from datetime import datetime
from pathlib import Path

import polars as pl

from signalflow.core import RawData
from signalflow.data.raw_store import DuckDbSpotStore


class RawDataFactory:
    @staticmethod
    def from_duckdb_spot_store(
        spot_store_path: Path,
        pairs: list[str],
        start: datetime,
        end: datetime,
        data_types: list[str] | None = None,
    ) -> RawData:
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

                spot = spot.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None)
                )

                dup_count = (
                    spot.group_by(["pair", "timestamp"]).len()
                    .filter(pl.col("len") > 1)
                )
                if dup_count.height > 0:
                    dups = (
                        spot.join(
                            dup_count.select(["pair", "timestamp"]),
                            on=["pair", "timestamp"],
                        )
                        .select(["pair", "timestamp"])
                        .head(10)
                    )
                    raise ValueError(
                        f"Duplicate (pair, timestamp) detected. Examples:\n{dups}"
                    )

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