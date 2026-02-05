---
title: Custom Data Types
description: How to register and use custom raw data types in SignalFlow
---

# Custom Data Types

SignalFlow ships with three built-in raw data types: **SPOT**, **FUTURES**, and **PERPETUAL**.
Each type defines a set of required columns that the framework uses for validation, feature
computation, and visualization.

You can register your own data types to work with non-standard data sources such as
limit order books, tick data, or any custom schema.

---

## Built-in Types

| Type | Columns |
|------|---------|
| `SPOT` | pair, timestamp, open, high, low, close, volume |
| `FUTURES` | SPOT + open_interest |
| `PERPETUAL` | SPOT + funding_rate, open_interest |

Access columns programmatically:

```python
from signalflow.core.enums import RawDataType

print(RawDataType.SPOT.columns)
# {'pair', 'timestamp', 'open', 'high', 'low', 'close', 'volume'}
```

---

## Registering a Custom Type

Use `default_registry.register_raw_data_type()` to add a new data type:

```python
from signalflow.core.registry import default_registry

default_registry.register_raw_data_type(
    name="lob",
    columns=["pair", "timestamp", "bid", "ask", "bid_size", "ask_size", "depth"],
)
```

After registration, the type is available everywhere in the framework:

```python
# Look up columns
cols = default_registry.get_raw_data_columns("lob")
# {'pair', 'timestamp', 'bid', 'ask', 'bid_size', 'ask_size', 'depth'}

# List all registered types
print(default_registry.list_raw_data_types())
# ['futures', 'lob', 'perpetual', 'spot']
```

---

## Using Custom Types with Components

### FeaturePipeline

Pass the custom type name as a string to `raw_data_type`:

```python
from signalflow.feature import FeaturePipeline

pipeline = FeaturePipeline(
    features=[MyLobFeature(depth_levels=5)],
    raw_data_type="lob",  # custom type
)
```

The pipeline validates that each feature's required columns are satisfied by the registered
column set. If a feature requires a column not present in your type, you get a clear error
at construction time.

### SignalDetector

```python
from signalflow.detector import SignalDetector

class LobImbalanceDetector(SignalDetector):
    raw_data_type = "lob"

    def detect(self, df, context=None):
        # df is guaranteed to have lob columns
        imbalance = df["bid_size"] - df["ask_size"]
        ...
```

### RawDataView

```python
from signalflow.core import RawData, RawDataView
from signalflow.core.enums import DataFrameType

raw = RawData(
    datetime_start=start,
    datetime_end=end,
    pairs=["BTCUSDT"],
    data={"lob": lob_dataframe},
)

view = RawDataView(raw=raw)

# Access by custom type name
lob_pl = view.get_data("lob", DataFrameType.POLARS)
lob_pd = view.get_data("lob", DataFrameType.PANDAS)
```

---

## Overriding Built-in Types

If you need to extend or redefine columns for a built-in type, pass `override=True`:

```python
default_registry.register_raw_data_type(
    name="spot",
    columns=["pair", "timestamp", "open", "high", "low", "close", "volume", "trades"],
    override=True,
)
```

!!! warning
    Overriding built-in types affects all components that reference them.
    Use with caution.

---

## Full Example: Tick Data Pipeline

```python
from signalflow.core.registry import default_registry
from signalflow.core import RawData, RawDataView
from signalflow.feature import Feature, FeaturePipeline
from signalflow.core.enums import DataFrameType
from dataclasses import dataclass
from typing import ClassVar
from datetime import datetime
import polars as pl

# 1. Register custom type
default_registry.register_raw_data_type(
    name="tick",
    columns=["pair", "timestamp", "price", "qty", "is_buyer_maker"],
)

# 2. Define a feature for tick data
@dataclass
class TickVwapFeature(Feature):
    window: int = 100

    requires: ClassVar[list[str]] = ["price", "qty"]
    outputs: ClassVar[list[str]] = ["vwap_{window}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        pq = (pl.col("price") * pl.col("qty")).rolling_sum(self.window)
        q = pl.col("qty").rolling_sum(self.window)
        return df.with_columns((pq / q).alias(f"vwap_{self.window}"))

# 3. Build pipeline
pipeline = FeaturePipeline(
    features=[TickVwapFeature(window=100)],
    raw_data_type="tick",
)

# 4. Create data and compute
tick_df = pl.DataFrame({
    "pair": ["BTCUSDT"] * 200,
    "timestamp": [datetime(2024, 1, 1, 10, 0, i) for i in range(200)],
    "price": [45000.0 + i * 0.5 for i in range(200)],
    "qty": [0.1] * 200,
    "is_buyer_maker": [True, False] * 100,
})

raw = RawData(
    datetime_start=datetime(2024, 1, 1),
    datetime_end=datetime(2024, 1, 2),
    pairs=["BTCUSDT"],
    data={"tick": tick_df},
)

view = RawDataView(raw=raw)
result = pipeline.run(view)
print(result.select("pair", "timestamp", "price", "vwap_100").tail(5))
```

---

## API Reference

See the full API documentation for the registry methods:

- [`register_raw_data_type()`](../api/core.md) - Register a new data type
- [`get_raw_data_columns()`](../api/core.md) - Look up columns for any type
- [`list_raw_data_types()`](../api/core.md) - List all registered types
