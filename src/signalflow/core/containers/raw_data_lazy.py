"""Lazy-loading version of RawData with configurable caching.

Provides identical interface to RawData but loads data on-demand
from stores or cached parquet files.

Example:
    ```python
    from signalflow.core import RawDataLazy
    from signalflow.data import DuckDbRawStore

    # Create lazy container
    raw = RawDataLazy.from_stores(
        stores={
            "binance": DuckDbRawStore(db_path="binance.duckdb", data_type="perpetual"),
            "okx": DuckDbRawStore(db_path="okx.duckdb", data_type="perpetual"),
        },
        pairs=["BTCUSDT", "ETHUSDT"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
        cache_mode="memory",  # or "disk", "none"
    )

    # Data loaded only when accessed
    df = raw.perpetual.binance  # <- loads here
    df = raw.perpetual.okx      # <- loads here

    # Second access uses cache (if cache_mode != "none")
    df = raw.perpetual.binance  # <- from cache
    ```
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from signalflow.core.containers.raw_data import RawData
    from signalflow.data.raw_store.base import RawDataStore


CacheMode = Literal["memory", "disk", "none"]


class LazyDataTypeAccessor:
    """Lazy accessor for hierarchical data access pattern.

    Similar to DataTypeAccessor but loads data on first access.

    Example:
        ```python
        # Data loaded lazily
        df = raw.perpetual.binance   # loads from store
        df = raw.perpetual.okx       # loads from store

        # List sources (no loading)
        print(raw.perpetual.sources)  # ["binance", "okx"]
        ```
    """

    __slots__ = ("_data_type", "_lazy_raw")

    def __init__(self, lazy_raw: RawDataLazy, data_type: str):
        object.__setattr__(self, "_lazy_raw", lazy_raw)
        object.__setattr__(self, "_data_type", data_type)

    def __getattr__(self, source: str) -> pl.DataFrame:
        """Access source by attribute: raw.perpetual.binance."""
        lazy_raw = object.__getattribute__(self, "_lazy_raw")
        data_type = object.__getattribute__(self, "_data_type")

        if source not in lazy_raw._stores.get(data_type, {}):
            available = list(lazy_raw._stores.get(data_type, {}).keys())
            raise AttributeError(f"No source '{source}' for data type '{data_type}'. Available: {available}")

        return lazy_raw._load(data_type, source)

    @property
    def sources(self) -> list[str]:
        """List available source names (no data loading)."""
        lazy_raw = object.__getattribute__(self, "_lazy_raw")
        data_type = object.__getattribute__(self, "_data_type")
        return list(lazy_raw._stores.get(data_type, {}).keys())

    def to_polars(self) -> pl.DataFrame:
        """Return default source DataFrame with warning."""
        lazy_raw = object.__getattribute__(self, "_lazy_raw")
        data_type = object.__getattribute__(self, "_data_type")

        sources = self.sources
        if not sources:
            raise ValueError(f"No sources available for '{data_type}'")

        source = lazy_raw.default_source
        if source is None or source not in sources:
            source = sources[0]

        warnings.warn(
            f"Using default source '{source}' for '{data_type}'. Specify explicitly: raw.{data_type}.{source}",
            UserWarning,
            stacklevel=2,
        )
        return lazy_raw._load(data_type, source)

    def __iter__(self):
        """Iterate over (source, DataFrame) pairs. Loads all sources."""
        lazy_raw = object.__getattribute__(self, "_lazy_raw")
        data_type = object.__getattribute__(self, "_data_type")

        for source in self.sources:
            yield source, lazy_raw._load(data_type, source)

    def __len__(self) -> int:
        """Return number of sources (no data loading)."""
        return len(self.sources)

    def __contains__(self, source: str) -> bool:
        """Check if source exists (no data loading)."""
        return source in self.sources

    def __repr__(self) -> str:
        data_type = object.__getattribute__(self, "_data_type")
        return f"LazyDataTypeAccessor(data_type='{data_type}', sources={self.sources})"


@dataclass
class RawDataLazy:
    """Lazy-loading container for raw market data.

    Identical interface to RawData but loads data on-demand from stores.
    Supports configurable caching: memory, disk (parquet), or none.

    Attributes:
        datetime_start: Start datetime of the data snapshot.
        datetime_end: End datetime of the data snapshot.
        pairs: List of trading pairs.
        default_source: Default source for implicit access.
        cache_mode: Caching strategy ("memory", "disk", "none").
        cache_dir: Directory for disk cache (auto-generated if None).

    Example:
        ```python
        raw = RawDataLazy.from_stores(
            stores={"binance": store1, "okx": store2},
            pairs=["BTCUSDT"],
            start=start, end=end,
            cache_mode="memory",
        )

        # Hierarchical access (lazy load)
        df = raw.perpetual.binance
        print(raw.perpetual.sources)

        # Dict-style access (lazy load)
        df = raw["perpetual", "binance"]
        df = raw.get("perpetual", source="binance")
        ```
    """

    datetime_start: datetime
    datetime_end: datetime
    pairs: list[str] = field(default_factory=list)
    default_source: str | None = None
    cache_mode: CacheMode = "memory"
    cache_dir: Path | None = None

    # Internal: stores[data_type][source] -> RawDataStore
    _stores: dict[str, dict[str, RawDataStore]] = field(default_factory=dict, repr=False)
    # Internal: memory cache[data_type][source] -> DataFrame
    _memory_cache: dict[str, dict[str, pl.DataFrame]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Initialize cache directory for disk mode."""
        if self.cache_mode == "disk" and self.cache_dir is None:
            import tempfile

            self.cache_dir = Path(tempfile.mkdtemp(prefix="signalflow_cache_"))

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def from_stores(
        cls,
        stores: Mapping[str, RawDataStore],
        pairs: list[str],
        start: datetime,
        end: datetime,
        default_source: str | None = None,
        cache_mode: CacheMode = "memory",
        cache_dir: Path | None = None,
    ) -> RawDataLazy:
        """Create lazy RawData from dict of stores.

        Args:
            stores: Dict mapping source names to stores.
            pairs: List of trading pairs to load.
            start: Start datetime (inclusive).
            end: End datetime (inclusive).
            default_source: Default source for implicit access.
            cache_mode: Caching strategy ("memory", "disk", "none").
            cache_dir: Directory for disk cache.

        Returns:
            RawDataLazy: Lazy-loading container.

        Example:
            ```python
            raw = RawDataLazy.from_stores(
                stores={
                    "binance": DuckDbRawStore(db_path="binance.duckdb", data_type="perpetual"),
                    "okx": DuckDbRawStore(db_path="okx.duckdb", data_type="perpetual"),
                },
                pairs=["BTCUSDT", "ETHUSDT"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 12, 31),
                cache_mode="disk",
            )

            # Data only loaded when accessed
            df = raw.perpetual.binance
            ```
        """
        # Group stores by data_type
        nested_stores: dict[str, dict[str, RawDataStore]] = {}

        for source_name, store in stores.items():
            data_type = getattr(store, "data_type", "unknown")
            if data_type not in nested_stores:
                nested_stores[data_type] = {}
            nested_stores[data_type][source_name] = store

        if default_source is None and stores:
            default_source = next(iter(stores.keys()))

        return cls(
            datetime_start=start,
            datetime_end=end,
            pairs=pairs,
            default_source=default_source,
            cache_mode=cache_mode,
            cache_dir=cache_dir,
            _stores=nested_stores,
        )

    # ── Lazy Loading ──────────────────────────────────────────────────

    def _cache_key(self, data_type: str, source: str) -> str:
        """Generate cache key for disk storage."""
        key_parts = [
            data_type,
            source,
            ",".join(sorted(self.pairs)),
            self.datetime_start.isoformat(),
            self.datetime_end.isoformat(),
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _cache_path(self, data_type: str, source: str) -> Path:
        """Get parquet cache file path."""
        if self.cache_dir is None:
            raise ValueError("cache_dir not set for disk cache mode")
        key = self._cache_key(data_type, source)
        return self.cache_dir / f"{data_type}_{source}_{key}.parquet"

    def _load(self, data_type: str, source: str) -> pl.DataFrame:
        """Load data with caching based on cache_mode."""
        # Check memory cache first
        if self.cache_mode == "memory" and data_type in self._memory_cache and source in self._memory_cache[data_type]:
            return self._memory_cache[data_type][source]

        # Check disk cache
        if self.cache_mode == "disk":
            cache_path = self._cache_path(data_type, source)
            if cache_path.exists():
                return pl.read_parquet(cache_path)

        # Load from store
        store = self._stores[data_type][source]
        df = store.load_many(pairs=self.pairs, start=self.datetime_start, end=self.datetime_end)

        # Normalize timestamps
        if "timestamp" in df.columns:
            df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None))

        # Sort by (pair, timestamp)
        if {"pair", "timestamp"}.issubset(df.columns):
            df = df.sort(["pair", "timestamp"])

        # Cache result
        if self.cache_mode == "memory":
            if data_type not in self._memory_cache:
                self._memory_cache[data_type] = {}
            self._memory_cache[data_type][source] = df

        elif self.cache_mode == "disk":
            cache_path = self._cache_path(data_type, source)
            df.write_parquet(cache_path)

        return df

    def preload(self, data_types: list[str] | None = None, sources: list[str] | None = None) -> RawDataLazy:
        """Preload specified data into cache.

        Useful to trigger loading before intensive computations.

        Args:
            data_types: Data types to preload. None = all.
            sources: Sources to preload. None = all.

        Returns:
            Self for chaining.

        Example:
            ```python
            raw.preload(data_types=["perpetual"], sources=["binance", "okx"])
            ```
        """
        types_to_load = data_types or list(self._stores.keys())

        for data_type in types_to_load:
            if data_type not in self._stores:
                continue
            sources_to_load = sources or list(self._stores[data_type].keys())
            for source in sources_to_load:
                if source in self._stores[data_type]:
                    self._load(data_type, source)

        return self

    def clear_cache(self) -> RawDataLazy:
        """Clear all cached data.

        Returns:
            Self for chaining.
        """
        self._memory_cache.clear()

        if self.cache_mode == "disk" and self.cache_dir and self.cache_dir.exists():
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()

        return self

    # ── RawData Interface ─────────────────────────────────────────────

    def __getattr__(self, name: str) -> LazyDataTypeAccessor:
        """Attribute access: raw.perpetual -> LazyDataTypeAccessor."""
        # Avoid recursion
        if name.startswith("_") or name in (
            "datetime_start",
            "datetime_end",
            "pairs",
            "default_source",
            "cache_mode",
            "cache_dir",
        ):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        if name not in self._stores:
            raise AttributeError(f"No data type '{name}'. Available: {list(self._stores.keys())}")

        return LazyDataTypeAccessor(self, name)

    def get(self, key: str, source: str | None = None) -> pl.DataFrame:
        """Get dataset by key with optional source.

        Args:
            key: Data type name (e.g., "perpetual").
            source: Source name. If None, uses default with warning.

        Returns:
            pl.DataFrame: Loaded data.
        """
        if key not in self._stores:
            return pl.DataFrame()

        if source is not None:
            if source not in self._stores[key]:
                raise KeyError(f"Source '{source}' not found for '{key}'. Available: {list(self._stores[key].keys())}")
            return self._load(key, source)

        # No source specified - use default with warning
        sources = list(self._stores[key].keys())
        default = self.default_source
        if default is None or default not in sources:
            default = sources[0] if sources else None

        if default is None:
            return pl.DataFrame()

        warnings.warn(
            f"Using default source '{default}' for '{key}'. Specify explicitly: raw.get('{key}', source='{default}')",
            UserWarning,
            stacklevel=2,
        )
        return self._load(key, default)

    def __getitem__(self, key: str | tuple[str, str]) -> pl.DataFrame:
        """Dict-style access: raw["perpetual", "binance"] or raw["perpetual"]."""
        if isinstance(key, tuple):
            data_type, source = key
            return self.get(data_type, source=source)
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if data type exists."""
        return key in self._stores

    def keys(self) -> Iterator[str]:
        """Return available data type keys."""
        return iter(self._stores.keys())

    def sources(self, data_type: str) -> list[str]:
        """Return available sources for a data type."""
        if data_type not in self._stores:
            raise KeyError(f"No data type '{data_type}'. Available: {list(self._stores.keys())}")
        return list(self._stores[data_type].keys())

    # ── Conversion ────────────────────────────────────────────────────

    def to_raw_data(self) -> RawData:
        """Convert to eager RawData by loading all data.

        Returns:
            RawData: Fully loaded container.

        Example:
            ```python
            # Load all data at once
            raw_eager = raw_lazy.to_raw_data()
            ```
        """
        from signalflow.core.containers.raw_data import RawData

        nested_data: dict[str, dict[str, pl.DataFrame]] = {}

        for data_type in self._stores:
            nested_data[data_type] = {}
            for source in self._stores[data_type]:
                nested_data[data_type][source] = self._load(data_type, source)

        return RawData(
            datetime_start=self.datetime_start,
            datetime_end=self.datetime_end,
            pairs=self.pairs,
            data=nested_data,
            default_source=self.default_source,
        )

    @property
    def is_loaded(self) -> dict[str, dict[str, bool]]:
        """Check which data is currently loaded/cached.

        Returns:
            Dict showing load status per data_type/source.
        """
        result: dict[str, dict[str, bool]] = {}

        for data_type in self._stores:
            result[data_type] = {}
            for source in self._stores[data_type]:
                if self.cache_mode == "memory":
                    loaded = data_type in self._memory_cache and source in self._memory_cache[data_type]
                elif self.cache_mode == "disk":
                    loaded = self._cache_path(data_type, source).exists()
                else:
                    loaded = False
                result[data_type][source] = loaded

        return result
