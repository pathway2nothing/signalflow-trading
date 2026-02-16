import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl


class DataTypeAccessor:
    """Accessor for hierarchical data access pattern.

    Enables clean attribute-based access to exchange data:
        raw.perpetual.binance  -> DataFrame for binance
        raw.perpetual.sources  -> ["binance", "okx", "bybit"]
        raw.perpetual.to_polars() -> default source with warning

    Attributes:
        _sources: Dict mapping source names to DataFrames.
        _default: Default source name (first registered if not set).
        _data_type: Data type name for warning messages.

    Example:
        ```python
        # Access specific source
        df = raw.perpetual.binance

        # List available sources
        print(raw.perpetual.sources)  # ["binance", "okx", "bybit"]

        # Get default with warning
        df = raw.perpetual.to_polars()  # warns about implicit source

        # Iterate over sources
        for source, df in raw.perpetual:
            print(f"{source}: {df.shape}")
        ```
    """

    __slots__ = ("_data_type", "_default", "_sources")

    def __init__(
        self,
        sources: dict[str, pl.DataFrame],
        default: str | None,
        data_type: str,
    ):
        object.__setattr__(self, "_sources", sources)
        object.__setattr__(self, "_default", default)
        object.__setattr__(self, "_data_type", data_type)

    def __getattr__(self, source: str) -> pl.DataFrame:
        """Access source by attribute: raw.perpetual.binance."""
        sources = object.__getattribute__(self, "_sources")
        if source in sources:
            return sources[source]
        data_type = object.__getattribute__(self, "_data_type")
        available = list(sources.keys())
        raise AttributeError(f"No source '{source}' for data type '{data_type}'. Available: {available}")

    @property
    def sources(self) -> list[str]:
        """List available source names."""
        return list(self._sources.keys())

    def to_polars(self) -> pl.DataFrame:
        """Return default source DataFrame with warning.

        Emits UserWarning to encourage explicit source selection.

        Returns:
            pl.DataFrame: Default source DataFrame.

        Raises:
            ValueError: If no sources available.
        """
        if not self._sources:
            raise ValueError(f"No sources available for '{self._data_type}'")

        source = self._default
        if source is None or source not in self._sources:
            source = next(iter(self._sources.keys()))

        warnings.warn(
            f"Using default source '{source}' for '{self._data_type}'. "
            f"Specify explicitly: raw.{self._data_type}.{source}",
            UserWarning,
            stacklevel=2,
        )
        return self._sources[source]

    def __iter__(self):
        """Iterate over (source, DataFrame) pairs."""
        return iter(self._sources.items())

    def __len__(self) -> int:
        """Return number of sources."""
        return len(self._sources)

    def __contains__(self, source: str) -> bool:
        """Check if source exists."""
        return source in self._sources

    def __repr__(self) -> str:
        return f"DataTypeAccessor(data_type='{self._data_type}', sources={self.sources})"


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

    Supports two data structures:
        - Flat: dict[str, pl.DataFrame] - single source per data type
        - Nested: dict[str, dict[str, pl.DataFrame]] - multi-source per data type

    Attributes:
        datetime_start (datetime): Start datetime of the data snapshot.
        datetime_end (datetime): End datetime of the data snapshot.
        pairs (list[str]): List of trading pairs in the snapshot.
        data (dict): Dictionary of datasets. Can be flat or nested.
        default_source (str | None): Default source for nested data.

    Example:
        ```python
        from signalflow.core import RawData
        import polars as pl
        from datetime import datetime

        # Flat structure (single source)
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

        # Nested structure (multi-source)
        raw_data = RawData(
            datetime_start=datetime(2024, 1, 1),
            datetime_end=datetime(2024, 12, 31),
            pairs=["BTCUSDT", "ETHUSDT"],
            data={
                "perpetual": {
                    "binance": binance_df,
                    "okx": okx_df,
                    "bybit": bybit_df,
                }
            },
            default_source="binance",
        )

        # Hierarchical access
        df = raw_data.perpetual.binance      # specific source
        df = raw_data.perpetual.to_polars()  # default with warning
        print(raw_data.perpetual.sources)    # ["binance", "okx", "bybit"]

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
    data: dict[str, pl.DataFrame | dict[str, pl.DataFrame]] = field(default_factory=dict)
    default_source: str | None = None

    def _is_nested(self, key: str) -> bool:
        """Check if data type has nested (multi-source) structure."""
        value = self.data.get(key)
        if value is None:
            return False
        return isinstance(value, dict) and not isinstance(value, pl.DataFrame)

    def __getattr__(self, name: str) -> DataTypeAccessor:
        """Attribute access for hierarchical pattern: raw.perpetual.binance.

        Args:
            name: Data type name.

        Returns:
            DataTypeAccessor: Accessor for the data type.

        Raises:
            AttributeError: If data type doesn't exist.
        """
        # Avoid recursion for special attributes
        if name.startswith("_") or name in ("data", "default_source", "datetime_start", "datetime_end", "pairs"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        data = object.__getattribute__(self, "data")
        default_source = object.__getattribute__(self, "default_source")

        if name not in data:
            raise AttributeError(f"No data type '{name}'. Available: {list(data.keys())}")

        value = data[name]

        # Nested structure: return accessor
        if isinstance(value, dict) and not isinstance(value, pl.DataFrame):
            return DataTypeAccessor(value, default_source, name)

        # Flat structure: wrap single DataFrame as accessor
        source_name = default_source or "default"
        return DataTypeAccessor({source_name: value}, source_name, name)

    def get(self, key: str, source: str | None = None) -> pl.DataFrame:
        """Get dataset by key.

        For nested (multi-source) data, returns default source with warning
        unless source is explicitly specified.

        Args:
            key (str): Dataset name (e.g. "spot", "perpetual").
            source (str | None): Source name for nested data.
                If None, uses default_source with warning.

        Returns:
            pl.DataFrame: Polars DataFrame if exists, empty DataFrame otherwise.

        Raises:
            TypeError: If dataset exists but is not a valid structure.
            KeyError: If source specified but not found.

        Example:
            ```python
            # Flat structure
            spot_df = raw_data.get("spot")

            # Nested structure - explicit source
            df = raw_data.get("perpetual", source="binance")

            # Nested structure - default source (warns)
            df = raw_data.get("perpetual")

            # Returns empty DataFrame if key doesn't exist
            missing_df = raw_data.get("nonexistent")
            assert missing_df.is_empty()
            ```
        """
        obj = self.data.get(key)
        if obj is None:
            return pl.DataFrame()

        # Flat structure: single DataFrame
        if isinstance(obj, pl.DataFrame):
            return obj

        # Nested structure: dict of DataFrames
        if isinstance(obj, dict):
            if source is not None:
                if source not in obj:
                    raise KeyError(f"Source '{source}' not found for '{key}'. Available: {list(obj.keys())}")
                return obj[source]

            # No source specified - use default with warning
            default = self.default_source
            if default is None or default not in obj:
                default = next(iter(obj.keys()), None)
            if default is None:
                return pl.DataFrame()

            warnings.warn(
                f"Using default source '{default}' for '{key}'. "
                f"Specify explicitly: raw.get('{key}', source='{default}')",
                UserWarning,
                stacklevel=2,
            )
            return obj[default]

        raise TypeError(f"Dataset '{key}' has invalid type: {type(obj)}")

    def __getitem__(self, key: str | tuple[str, str]) -> pl.DataFrame:
        """Dictionary-style access to datasets.

        Supports both simple key and tuple (data_type, source) indexing.

        Args:
            key: Dataset name or (data_type, source) tuple.

        Returns:
            pl.DataFrame: Dataset as Polars DataFrame.

        Example:
            ```python
            # Flat structure
            spot_df = raw_data["spot"]

            # Nested structure - explicit source
            df = raw_data["perpetual", "binance"]

            # Nested structure - default source (warns)
            df = raw_data["perpetual"]
            ```
        """
        if isinstance(key, tuple):
            data_type, source = key
            return self.get(data_type, source=source)
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

    def sources(self, data_type: str) -> list[str]:
        """Return available sources for a data type.

        Args:
            data_type: Data type name (e.g. "perpetual", "spot").

        Returns:
            list[str]: List of source names. Returns ["default"] for flat data.

        Raises:
            KeyError: If data type doesn't exist.

        Example:
            ```python
            # Nested structure
            print(raw_data.sources("perpetual"))  # ["binance", "okx", "bybit"]

            # Flat structure
            print(raw_data.sources("spot"))  # ["default"]
            ```
        """
        if data_type not in self.data:
            raise KeyError(f"No data type '{data_type}'. Available: {list(self.data.keys())}")

        value = self.data[data_type]
        if isinstance(value, pl.DataFrame):
            return [self.default_source or "default"]
        if isinstance(value, dict):
            return list(value.keys())
        raise TypeError(f"Invalid data structure for '{data_type}'")

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
