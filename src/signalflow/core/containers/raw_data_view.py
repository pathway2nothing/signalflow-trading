from dataclasses import dataclass, field
import pandas as pd
import polars as pl
from .raw_data import RawData
from signalflow.core.enums import DataFrameType

@dataclass
class RawDataView:
    raw: RawData
    cache_pandas: bool = False
    _pd_cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self):
        if self._pd_cache is None:
            self._pd_cache = {}

    def pl(self, key: str) -> pl.DataFrame:
        return self.raw[key]

    def pd(self, key: str) -> pd.DataFrame:
        df_pl = self.pl(key)
        if df_pl.is_empty():
            return pd.DataFrame()

        if self.cache_pandas and key in self._pd_cache:
            df = self._pd_cache[key]
        else:
            df = df_pl.to_pandas()
            if self.cache_pandas:
                self._pd_cache[key] = df

        index_cols = ["pair", "timestamp"] if {"pair", "timestamp"}.issubset(df.columns) else None
        if index_cols is None:
            raise ValueError(f"Cannot infer index columns for '{key}'. Provide index_cols explicitly.")

        if "timestamp" in index_cols and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
        return df.set_index(index_cols).sort_index()
    
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
            return self.pl(raw_data_type)
        elif df_type == DataFrameType.PANDAS:
            return self.pd(raw_data_type)
        else:
            raise ValueError(f"Unsupported df_type: {df_type}")