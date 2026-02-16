from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pandas as pd
import polars as pl

from signalflow.core import SfComponentType, sf_component
from signalflow.core.containers.raw_data import RawData
from signalflow.core.registry import default_registry
from signalflow.data.raw_store._schema import polars_schema, resolve_columns
from signalflow.data.raw_store.base import RawDataStore


@dataclass
@sf_component(name="memory/spot")
class InMemoryRawStore(RawDataStore):
    """In-memory storage backend for raw market data.

    Supports any registered raw data type (spot, futures, perpetual, custom).
    Stores all data in a single Polars DataFrame. Useful for tests,
    notebooks, and short-lived pipelines where persistence is not needed.
    """

    data_type: str = "spot"
    timeframe: str = "1m"
    _columns: list[str] = field(init=False, repr=False)
    _df: pl.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._columns = resolve_columns(self.data_type)
        self._df = pl.DataFrame(schema=polars_schema(self._columns))

    # ── Insert ────────────────────────────────────────────────────────

    def insert_klines(self, pair: str, klines: list[dict]) -> None:
        if not klines:
            return
        rows = []
        for k in klines:
            ts = k["timestamp"]
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            row: dict = {
                "pair": pair,
                "timestamp": ts,
            }
            for col in self._columns:
                if col == "trades":
                    row[col] = k.get(col)
                else:
                    val = k.get(col)
                    row[col] = float(val) if val is not None else None
            rows.append(row)
        new = pl.DataFrame(rows, schema=polars_schema(self._columns))
        # upsert: drop existing rows with same (pair, timestamp), then append
        self._df = (
            self._df.join(new.select("pair", "timestamp"), on=["pair", "timestamp"], how="anti")
            .vstack(new)
            .sort("pair", "timestamp")
        )

    # ── Queries ───────────────────────────────────────────────────────

    def get_time_bounds(self, pair: str) -> tuple[datetime | None, datetime | None]:
        subset = self._df.filter(pl.col("pair") == pair)
        if subset.is_empty():
            return (None, None)
        return (subset["timestamp"].min(), subset["timestamp"].max())

    def find_gaps(self, pair: str, start: datetime, end: datetime, tf_minutes: int) -> list[tuple[datetime, datetime]]:
        subset = self._df.filter(
            (pl.col("pair") == pair) & (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
        )
        existing_times = set(subset["timestamp"].to_list())
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

    def _apply_time_filter(
        self, df: pl.DataFrame, hours: int | None, start: datetime | None, end: datetime | None
    ) -> pl.DataFrame:
        if hours is not None:
            cutoff = datetime.now(tz=UTC).replace(tzinfo=None) - timedelta(hours=hours)
            df = df.filter(pl.col("timestamp") > cutoff)
        elif start and end:
            df = df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
        elif start:
            df = df.filter(pl.col("timestamp") >= start)
        elif end:
            df = df.filter(pl.col("timestamp") <= end)
        return df

    def load(
        self,
        pair: str,
        hours: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        df = self._df.filter(pl.col("pair") == pair)
        df = self._apply_time_filter(df, hours, start, end)
        return df.sort("timestamp")

    def load_many(
        self,
        pairs: Iterable[str],
        hours: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        pairs = list(pairs)
        if not pairs:
            return pl.DataFrame(schema=polars_schema(self._columns))
        df = self._df.filter(pl.col("pair").is_in(pairs))
        df = self._apply_time_filter(df, hours, start, end)
        return df.sort("pair", "timestamp")

    def load_many_pandas(
        self,
        pairs: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        return self.load_many(pairs=pairs, start=start, end=end).to_pandas()

    # ── Stats / lifecycle ─────────────────────────────────────────────

    def get_stats(self) -> pl.DataFrame:
        if self._df.is_empty():
            return pl.DataFrame(
                schema={
                    "pair": pl.Utf8,
                    "rows": pl.UInt32,
                    "first_candle": pl.Datetime,
                    "last_candle": pl.Datetime,
                    "total_volume": pl.Float64,
                }
            )
        return (
            self._df.group_by("pair")
            .agg(
                pl.len().alias("rows"),
                pl.col("timestamp").min().alias("first_candle"),
                pl.col("timestamp").max().alias("last_candle"),
                pl.col("volume").sum().round(2).alias("total_volume"),
            )
            .sort("pair")
        )

    def close(self) -> None:
        self._df = pl.DataFrame(schema=polars_schema(self._columns))

    def to_raw_data(
        self,
        pairs: list[str],
        start: datetime,
        end: datetime,
        data_key: str | None = None,
    ) -> RawData:
        """Convert store data to RawData container.

        Args:
            pairs: List of trading pairs to load.
            start: Start datetime (inclusive).
            end: End datetime (inclusive).
            data_key: Key for data in RawData.data dict.
                If None, uses store's data_type.

        Returns:
            RawData container with loaded and validated data.

        Raises:
            ValueError: If required columns missing or duplicates detected.
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


# Register for futures and perpetual.
default_registry.register(SfComponentType.RAW_DATA_STORE, "memory/futures", InMemoryRawStore, override=True)
default_registry.register(SfComponentType.RAW_DATA_STORE, "memory/perpetual", InMemoryRawStore, override=True)

# Backward-compatible alias.
InMemorySpotStore = InMemoryRawStore
