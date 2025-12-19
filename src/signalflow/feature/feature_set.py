from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
import pandas as pd

from signalflow.feature.base_extractor import FeatureExtractor
from signalflow.core import RawDataView, RawDataType


@dataclass
class FeatureSet:
    """
    Collection of independent extractors for parallel execution (future).

    Current behavior:
      - for each extractor: fetch proper raw data backend (pd/pl)
      - run extractor.extract(...)
      - normalize to Polars
      - join all results on (pair, timestamp)
    """

    extractors: list[FeatureExtractor]
    parallel: bool = False

    pair_col: str = "pair"
    ts_col: str = "timestamp"

    def __post_init__(self) -> None:
        if not self.extractors:
            raise ValueError("At least one extractor must be provided")

        for ex in self.extractors:
            if getattr(ex, "pair_col", self.pair_col) != self.pair_col:
                raise ValueError(
                    f"All extractors must use pair_col='{self.pair_col}'. "
                    f"{ex.__class__.__name__} uses '{getattr(ex, 'pair_col', None)}'"
                )
            if getattr(ex, "ts_col", self.ts_col) != self.ts_col:
                raise ValueError(
                    f"All extractors must use ts_col='{self.ts_col}'. "
                    f"{ex.__class__.__name__} uses '{getattr(ex, 'ts_col', None)}'"
                )

    def extract(self, raw_data: RawDataView, context: dict[str, Any] | None = None) -> pl.DataFrame:
        feature_dfs: list[pl.DataFrame] = []

        for extractor in self.extractors:
            input_df = self._get_input_df(raw_data, extractor)
            result_df = extractor.extract(input_df, context) 

            pl_df = self._to_polars(result_df)
            pl_df = self._normalize_index(pl_df)

            if self.pair_col not in pl_df.columns or self.ts_col not in pl_df.columns:
                raise ValueError(
                    f"{extractor.__class__.__name__} returned no index columns "
                    f"('{self.pair_col}', '{self.ts_col}'). "
                    f"FeatureSet requires index columns to combine features."
                )

            feature_dfs.append(pl_df)

        return self._combine_features(feature_dfs)

    def _get_input_df(self, raw_data: RawDataView, extractor: FeatureExtractor) -> pl.DataFrame | pd.DataFrame:
        """
        Backward compatible data fetch.

        Tries:
          - extractor.raw_data_type / extractor.df_type (old API)
        Fallback:
          - assume "spot" + "polars" if attrs absent (fails fast if raw_data doesn't support)
        """
        raw_data_type = getattr(extractor, "raw_data_type", RawDataType.SPOT)
        df_type = getattr(extractor, "df_type", None)


        df_type_norm: Any = df_type
        if hasattr(df_type, "name"):
            df_type_norm = df_type.name.lower()
        elif hasattr(df_type, "value"):
            df_type_norm = str(df_type.value).lower()

        if df_type_norm in ("pandas", "pd"):
            return raw_data.get_data(raw_data_type, "pandas")
        if df_type_norm in ("polars", "pl"):
            return raw_data.get_data(raw_data_type, "polars")

        return raw_data.get_data(raw_data_type, "polars")

    def _to_polars(self, df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
        if isinstance(df, pl.DataFrame):
            return df
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
        raise TypeError(f"Unsupported df type: {type(df)}")

    def _normalize_index(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.ts_col in df.columns:
            ts_dtype = df.schema.get(self.ts_col)
            if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
                df = df.with_columns(pl.col(self.ts_col).dt.replace_time_zone(None))
        return df

    def _combine_features(self, feature_dfs: list[pl.DataFrame]) -> pl.DataFrame:
        if not feature_dfs:
            raise ValueError("No feature DataFrames to combine")

        combined = feature_dfs[0]

        for right in feature_dfs[1:]:
            right_cols = [c for c in right.columns if c not in (self.pair_col, self.ts_col)]
            dup = set(right_cols).intersection(set(combined.columns))
            if dup:
                raise ValueError(
                    f"Duplicate feature columns during FeatureSet combine: {sorted(dup)}. "
                    f"Rename features or set unique prefixes."
                )

            combined = combined.join(right, on=[self.pair_col, self.ts_col], how="left")

        return combined
