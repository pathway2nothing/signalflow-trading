from dataclasses import dataclass
import pandas as pd
import polars as pl
from .raw_data import RawData

@dataclass
class RawDataView:
    raw: RawData
    cache_pandas: bool = False
    _pd_cache: dict[str, pd.DataFrame] = None

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
