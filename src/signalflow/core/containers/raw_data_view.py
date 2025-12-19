from dataclasses import dataclass, field
import pandas as pd
import polars as pl
from .raw_data import RawData
from signalflow.core.enums import DataFrameType


@dataclass
class RawDataView:
    raw: RawData
    cache_pandas: bool = False
    _pandas_cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self):
        if self._pandas_cache is None:
            self._pandas_cache = {}

    def to_polars(self, key: str) -> pl.DataFrame:
        return self.raw[key]

    def to_pandas(self, key: str) -> pd.DataFrame:
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
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")

        if {"pair", "timestamp"}.issubset(df.columns):
            df = df.sort_values(["pair", "timestamp"], kind="stable").reset_index(drop=True)

        return df

    def get_data(
        self, 
        raw_data_type: str, 
        df_type: DataFrameType
    ) -> pl.DataFrame | pd.DataFrame:
        """Get raw data in specified format.
        
        Unified interface for FeatureSet to access data in required format.
        
        Args:
            raw_data_type: Type of data ('spot', 'futures', 'perpetual')
            df_type: Target DataFrame type (POLARS or PANDAS)
            
        Returns:
            Raw data DataFrame in requested format
            
        Example:
            >>> view.get_data('spot', DataFrameType.POLARS)
            >>> view.get_data('futures', DataFrameType.PANDAS)
        """
        if df_type == DataFrameType.POLARS:
            return self.to_polars(raw_data_type)   
        elif df_type == DataFrameType.PANDAS:
            return self.to_pandas(raw_data_type)
        else:
            raise ValueError(f"Unsupported df_type: {df_type}")