---
title: Data Stores
description: Storage backends for market data and how to use them
---

# Data Stores

SignalFlow provides multiple storage backends for persisting historical market data with support for different data types (spot, futures, perpetual).

---

## Available Backends

| Backend | Use Case | Dependencies |
|---------|----------|--------------|
| `DuckDbRawStore` | High-performance local storage | `duckdb` |
| `SqliteRawStore` | Lightweight, zero-config | built-in |
| `InMemoryRawStore` | Testing and notebooks | none |
| `PgRawStore` | Production PostgreSQL | `psycopg2` (optional) |

---

## Quick Start

### DuckDB Store (Recommended)

```python
from signalflow.data.raw_store import DuckDbRawStore
from datetime import datetime
from pathlib import Path

# Create store
store = DuckDbRawStore(
    db_path=Path("data/spot.duckdb"),
    data_type="spot",  # or "futures", "perpetual"
)

# Insert data
klines = [
    {"timestamp": datetime(2024, 1, 1, 0, 0), "open": 42000.0, "high": 42100.0,
     "low": 41900.0, "close": 42050.0, "volume": 100.0, "trades": 500},
    {"timestamp": datetime(2024, 1, 1, 0, 1), "open": 42050.0, "high": 42150.0,
     "low": 41950.0, "close": 42100.0, "volume": 120.0, "trades": 600},
]
store.insert_klines("BTCUSDT", klines)

# Load single pair
df = store.load("BTCUSDT", hours=24)

# Load date range
df = store.load(
    "BTCUSDT",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 31),
)

# Load multiple pairs
df = store.load_many(
    pairs=["BTCUSDT", "ETHUSDT"],
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 31),
)

# Cleanup
store.close()
```

### SQLite Store

```python
from signalflow.data.raw_store import SqliteRawStore
from pathlib import Path

store = SqliteRawStore(
    db_path=Path("data/spot.sqlite"),
    data_type="spot",
)

# Same API as DuckDB
store.insert_klines("BTCUSDT", klines)
df = store.load("BTCUSDT", hours=24)
store.close()
```

### In-Memory Store

```python
from signalflow.data.raw_store import InMemoryRawStore

store = InMemoryRawStore(data_type="spot")

store.insert_klines("BTCUSDT", klines)
df = store.load("BTCUSDT")

# Data is lost when store is closed
store.close()
```

---

## Converting to RawData

The `to_raw_data()` method converts store data to an immutable `RawData` container:

```python
from datetime import datetime

# Direct from store
raw_data = store.to_raw_data(
    pairs=["BTCUSDT", "ETHUSDT"],
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
)

# Access data
spot_df = raw_data["spot"]  # Polars DataFrame
print(raw_data.pairs)       # ["BTCUSDT", "ETHUSDT"]
print(raw_data.datetime_start)  # datetime(2024, 1, 1)
```

### Custom Data Key

```python
# Use custom key instead of data_type
raw_data = store.to_raw_data(
    pairs=["BTCUSDT"],
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
    data_key="binance_spot",
)

df = raw_data["binance_spot"]
```

---

## RawDataFactory

For combining data from multiple stores:

```python
from signalflow.data import RawDataFactory
from signalflow.data.raw_store import DuckDbRawStore
from pathlib import Path
from datetime import datetime

# Create stores
spot_store = DuckDbRawStore(db_path=Path("data/spot.duckdb"), data_type="spot")
futures_store = DuckDbRawStore(db_path=Path("data/futures.duckdb"), data_type="futures")

# Combine into single RawData
raw_data = RawDataFactory.from_stores(
    stores=[spot_store, futures_store],
    pairs=["BTCUSDT", "ETHUSDT"],
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
)

# Access both data types
spot_df = raw_data["spot"]
futures_df = raw_data["futures"]
```

### Legacy Factory Methods

```python
# Single DuckDB store
raw_data = RawDataFactory.from_duckdb_spot_store(
    spot_store_path=Path("data/spot.duckdb"),
    pairs=["BTCUSDT"],
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
)

# From Polars DataFrames
raw_data = RawDataFactory.from_polars(
    spot_df=my_polars_df,
    pairs=["BTCUSDT"],
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
)
```

---

## Data Types and Schema

### Built-in Data Types

| Type | Columns |
|------|---------|
| `spot` | pair, timestamp, open, high, low, close, volume, trades |
| `futures` | spot + open_interest |
| `perpetual` | spot + open_interest, funding_rate |

### Using Different Data Types

```python
# Spot data
spot_store = DuckDbRawStore(db_path=Path("spot.duckdb"), data_type="spot")

# Futures data (includes open_interest)
futures_store = DuckDbRawStore(db_path=Path("futures.duckdb"), data_type="futures")
futures_klines = [
    {"timestamp": datetime(2024, 1, 1), "open": 42000.0, "high": 42100.0,
     "low": 41900.0, "close": 42050.0, "volume": 100.0, "open_interest": 50000.0},
]
futures_store.insert_klines("BTCUSDT", futures_klines)

# Perpetual data (includes open_interest + funding_rate)
perp_store = DuckDbRawStore(db_path=Path("perp.duckdb"), data_type="perpetual")
perp_klines = [
    {"timestamp": datetime(2024, 1, 1), "open": 42000.0, "high": 42100.0,
     "low": 41900.0, "close": 42050.0, "volume": 100.0,
     "open_interest": 50000.0, "funding_rate": 0.0001},
]
perp_store.insert_klines("BTCUSDT", perp_klines)
```

---

## Store Operations

### Upsert Semantics

All stores use upsert (INSERT OR REPLACE) based on `(pair, timestamp)` key:

```python
# Insert initial data
store.insert_klines("BTCUSDT", [
    {"timestamp": datetime(2024, 1, 1, 0, 0), "close": 42000.0, ...}
])

# Re-insert with updated values (replaces existing)
store.insert_klines("BTCUSDT", [
    {"timestamp": datetime(2024, 1, 1, 0, 0), "close": 42100.0, ...}
])

# Result: only one row with close=42100.0
```

### Time Bounds

```python
# Get first and last timestamp for a pair
first_ts, last_ts = store.get_time_bounds("BTCUSDT")
print(f"Data from {first_ts} to {last_ts}")
```

### Gap Detection

```python
from datetime import datetime

# Find gaps in data
gaps = store.find_gaps(
    pair="BTCUSDT",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 31),
    tf_minutes=1,  # 1-minute candles
)

for gap_start, gap_end in gaps:
    print(f"Missing data: {gap_start} to {gap_end}")
```

### Statistics

```python
# Get per-pair statistics
stats = store.get_stats()
print(stats)
# shape: (2, 5)
# ┌──────────┬───────┬─────────────────────┬─────────────────────┬──────────────┐
# │ pair     ┆ rows  ┆ first_candle        ┆ last_candle         ┆ total_volume │
# │ ---      ┆ ---   ┆ ---                 ┆ ---                 ┆ ---          │
# │ str      ┆ u32   ┆ datetime[μs]        ┆ datetime[μs]        ┆ f64          │
# ╞══════════╪═══════╪═════════════════════╪═════════════════════╪══════════════╡
# │ BTCUSDT  ┆ 10000 ┆ 2024-01-01 00:00:00 ┆ 2024-01-07 22:39:00 ┆ 1500000.0    │
# │ ETHUSDT  ┆ 10000 ┆ 2024-01-01 00:00:00 ┆ 2024-01-07 22:39:00 ┆ 2100000.0    │
# └──────────┴───────┴─────────────────────┴─────────────────────┴──────────────┘
```

---

## Creating a Custom Store

Extend `RawDataStore` to implement your own backend:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Iterable
import polars as pl

from signalflow.core import sf_component, SfComponentType
from signalflow.core.containers.raw_data import RawData
from signalflow.data.raw_store.base import RawDataStore


@dataclass
@sf_component(name="parquet/spot")
class ParquetRawStore(RawDataStore):
    """Parquet file storage backend."""

    data_dir: Path
    data_type: str = "spot"
    _cache: dict[str, pl.DataFrame] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, pair: str) -> Path:
        return self.data_dir / f"{pair}.parquet"

    def insert_klines(self, pair: str, klines: list[dict]) -> None:
        if not klines:
            return

        new_df = pl.DataFrame(klines)
        path = self._file_path(pair)

        if path.exists():
            existing = pl.read_parquet(path)
            # Upsert: remove existing timestamps, add new
            combined = pl.concat([
                existing.filter(~pl.col("timestamp").is_in(new_df["timestamp"])),
                new_df,
            ]).sort("timestamp")
        else:
            combined = new_df

        combined.write_parquet(path)

    def get_time_bounds(self, pair: str) -> tuple[Optional[datetime], Optional[datetime]]:
        path = self._file_path(pair)
        if not path.exists():
            return (None, None)

        df = pl.read_parquet(path, columns=["timestamp"])
        if df.is_empty():
            return (None, None)

        return (df["timestamp"].min(), df["timestamp"].max())

    def find_gaps(
        self, pair: str, start: datetime, end: datetime, tf_minutes: int
    ) -> list[tuple[datetime, datetime]]:
        # Implementation similar to other stores
        pass

    def load(
        self,
        pair: str,
        hours: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        path = self._file_path(pair)
        if not path.exists():
            return pl.DataFrame()

        df = pl.read_parquet(path)

        # Apply time filters
        if hours is not None:
            cutoff = datetime.now() - timedelta(hours=hours)
            df = df.filter(pl.col("timestamp") > cutoff)
        elif start and end:
            df = df.filter(
                (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
            )

        return df.sort("timestamp")

    def load_many(
        self,
        pairs: Iterable[str],
        hours: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        dfs = []
        for pair in pairs:
            df = self.load(pair, hours=hours, start=start, end=end)
            if not df.is_empty():
                dfs.append(df.with_columns(pl.lit(pair).alias("pair")))

        if not dfs:
            return pl.DataFrame()

        return pl.concat(dfs).sort(["pair", "timestamp"])

    def load_many_pandas(self, pairs: list[str], start=None, end=None):
        return self.load_many(pairs, start=start, end=end).to_pandas()

    def get_stats(self) -> pl.DataFrame:
        # Implementation
        pass

    def close(self) -> None:
        self._cache.clear()

    def to_raw_data(
        self,
        pairs: list[str],
        start: datetime,
        end: datetime,
        data_key: Optional[str] = None,
    ) -> RawData:
        key = data_key if data_key is not None else self.data_type
        df = self.load_many(pairs=pairs, start=start, end=end)

        return RawData(
            datetime_start=start,
            datetime_end=end,
            pairs=pairs,
            data={key: df},
        )
```

---

## StoreFactory

Create stores dynamically using the factory:

```python
from signalflow.data import StoreFactory

# Create DuckDB store
store = StoreFactory.create_raw_store(
    backend="duckdb",
    data_type="spot",
    db_path="data/spot.duckdb",
)

# Create SQLite store
store = StoreFactory.create_raw_store(
    backend="sqlite",
    data_type="futures",
    db_path="data/futures.sqlite",
)

# Create in-memory store
store = StoreFactory.create_raw_store(
    backend="memory",
    data_type="spot",
)
```

---

## PostgreSQL Store (Optional)

For production deployments:

```python
# Requires: pip install signalflow[postgres]
from signalflow.data.raw_store import PgRawStore

store = PgRawStore(
    host="localhost",
    port=5432,
    database="signalflow",
    user="postgres",
    password="secret",
    data_type="spot",
)

store.insert_klines("BTCUSDT", klines)
df = store.load("BTCUSDT", hours=24)
store.close()
```

---

## Best Practices

### Connection Management

```python
# Use try/finally
store = DuckDbRawStore(db_path=Path("data.duckdb"))
try:
    df = store.load("BTCUSDT")
finally:
    store.close()

# Or context manager (if implemented)
with DuckDbRawStore(db_path=Path("data.duckdb")) as store:
    df = store.load("BTCUSDT")
```

### Batch Inserts

```python
# Prefer batch insert over single klines
all_klines = fetch_many_klines()  # e.g., 10,000 klines
store.insert_klines("BTCUSDT", all_klines)  # Single batch

# Avoid
for kline in all_klines:
    store.insert_klines("BTCUSDT", [kline])  # 10,000 separate inserts
```

### Data Validation

```python
# to_raw_data() validates data automatically:
# - Checks required columns (pair, timestamp)
# - Normalizes timestamps to UTC-naive
# - Detects duplicate (pair, timestamp) combinations

try:
    raw_data = store.to_raw_data(pairs=["BTCUSDT"], start=start, end=end)
except ValueError as e:
    print(f"Data validation failed: {e}")
```

---

## API Reference

See the full API documentation:

- [`RawDataStore`](../api/data.md) - Base class for all stores
- [`DuckDbRawStore`](../api/data.md) - DuckDB implementation
- [`SqliteRawStore`](../api/data.md) - SQLite implementation
- [`InMemoryRawStore`](../api/data.md) - In-memory implementation
- [`RawDataFactory`](../api/data.md) - Factory for creating RawData
- [`StoreFactory`](../api/data.md) - Factory for creating stores
