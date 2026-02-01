from dataclasses import dataclass, field
import pandas as pd
import polars as pl
from .raw_data import RawData
from signalflow.core.enums import DataFrameType

# TODO raw_data_type -> RawDataType

@dataclass
class RawDataView:
    """Adapter for accessing RawData in different DataFrame formats.

    Provides unified interface for converting between Polars and Pandas formats,
    with optional caching for Pandas conversions.

    Key features:
        - Lazy conversion: Polars → Pandas only when needed
        - Optional caching for repeated Pandas access
        - Automatic timestamp normalization
        - Automatic sorting by (pair, timestamp)

    Attributes:
        raw (RawData): Underlying raw data container.
        cache_pandas (bool): Enable caching for Pandas conversions. Default: False.
        _pandas_cache (dict[str, pd.DataFrame]): Internal cache for Pandas DataFrames.

    Example:
        ```python
        from signalflow.core import RawData, RawDataView

        # Create view
        view = RawDataView(raw=raw_data, cache_pandas=True)

        # Access as Polars (zero-copy)
        spot_pl = view.to_polars("spot")

        # Access as Pandas (cached)
        spot_pd = view.to_pandas("spot")

        # Unified interface
        from signalflow.core.enums import DataFrameType
        data = view.get_data("spot", DataFrameType.POLARS)
        ```

    Note:
        Polars access is zero-copy. Pandas conversion creates a copy.
        Enable cache_pandas for repeated Pandas access to same dataset.
    """

    raw: RawData
    cache_pandas: bool = False
    _pandas_cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize internal cache if needed."""
        if self._pandas_cache is None:
            self._pandas_cache = {}

    def to_polars(self, key: str) -> pl.DataFrame:
        """Get dataset as Polars DataFrame.

        Zero-copy access to underlying Polars DataFrame in RawData.

        Args:
            key (str): Dataset name (e.g. "spot", "signals").

        Returns:
            pl.DataFrame: Dataset as Polars DataFrame.

        Example:
            ```python
            spot_df = view.to_polars("spot")
            print(f"Shape: {spot_df.shape}")
            ```
        """
        return self.raw[key]

    def to_pandas(self, key: str) -> pd.DataFrame:
        """Get dataset as Pandas DataFrame.

        Converts Polars DataFrame to Pandas with:
            - Timestamp normalization (UTC-aware → naive)
            - Automatic sorting by (pair, timestamp)
            - Optional caching for repeated access

        Args:
            key (str): Dataset name (e.g. "spot", "signals").

        Returns:
            pd.DataFrame: Dataset as Pandas DataFrame, sorted and normalized.

        Example:
            ```python
            # First access: converts and caches (if enabled)
            spot_df = view.to_pandas("spot")

            # Second access: returns cached version (if cache enabled)
            spot_df_again = view.to_pandas("spot")

            # Check timestamp type
            print(spot_df["timestamp"].dtype)  # datetime64[ns]
            ```

        Note:
            Returns empty DataFrame if dataset doesn't exist.
            Timestamp column is converted to timezone-naive datetime64[ns].
        """
        df_pl = self.to_polars(key)
        if df_pl.is_empty():
            return pd.DataFrame()

        if self.cache_pandas and key in self._pandas_cache:
            df = self._pandas_cache[key]
        else:
            df = df_pl.to_pandas()
            if self.cache_pandas:
                self._pandas_cache[key] = df

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="raise")

        if {"pair", "timestamp"}.issubset(df.columns):
            df = df.sort_values(["pair", "timestamp"], kind="stable").reset_index(drop=True)

        return df

    def get_data(
        self, 
        raw_data_type: str, 
        df_type: DataFrameType
    ) -> pl.DataFrame | pd.DataFrame:
        """Get raw data in specified format.
        
        Unified interface for accessing data in required DataFrame format.
        Used by FeaturePipeline to get data in format expected by extractors.
        
        Args:
            raw_data_type (str): Type of data ('spot', 'futures', 'perpetual').
            df_type (DataFrameType): Target DataFrame type (POLARS or PANDAS).
            
        Returns:
            pl.DataFrame | pd.DataFrame: Dataset in requested format.

        Raises:
            ValueError: If df_type is not POLARS or PANDAS.
            
        Example:
            ```python
            from signalflow.core.enums import DataFrameType

            # Get as Polars
            spot_pl = view.get_data('spot', DataFrameType.POLARS)

            # Get as Pandas
            spot_pd = view.get_data('spot', DataFrameType.PANDAS)

            # Used by FeatureExtractor
            class MyExtractor(FeatureExtractor):
                def extract(self, df):
                    # df will be in format specified by df_type
                    pass
            ```
        """
        if df_type == DataFrameType.POLARS:
            return self.to_polars(raw_data_type)   
        elif df_type == DataFrameType.PANDAS:
            return self.to_pandas(raw_data_type)
        else:
            raise ValueError(f"Unsupported df_type: {df_type}")