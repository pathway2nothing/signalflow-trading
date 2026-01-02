# IMPORTANT
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from signalflow.feature.base_extractor import FeatureExtractor
from signalflow.core import RawDataView, RawDataType, DataFrameType


@dataclass
class FeatureSet:
    """
    Polars-first collection of independent extractors.

    Behavior:
      - for each extractor: fetch raw data as Polars (canonical)
      - run extractor.extract(pl_df, context)
      - normalize index (timestamp tz -> naive)
      - join all results on (pair, timestamp)

    Notes:
      - FeatureExtractor is polars-first, so any pandas-only logic must be wrapped by adapters
        (e.g. PandasFeatureExtractor) that still expose pl.DataFrame output.
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

            result_df = extractor.extract(input_df, data_context=context)
            if not isinstance(result_df, pl.DataFrame):
                raise TypeError(
                    f"{extractor.__class__.__name__}.extract must return pl.DataFrame, got {type(result_df)}"
                )

            result_df = self._normalize_index(result_df)

            if self.pair_col not in result_df.columns or self.ts_col not in result_df.columns:
                raise ValueError(
                    f"{extractor.__class__.__name__} returned no index columns "
                    f"('{self.pair_col}', '{self.ts_col}'). "
                    f"FeatureSet requires index columns to combine features."
                )

            feature_dfs.append(result_df)

        return self._combine_features(feature_dfs)

    def _get_input_df(self, raw_data: RawDataView, extractor: FeatureExtractor) -> pl.DataFrame:
        """
        Polars-first data fetch.

        Uses:
          - extractor.raw_data_type if present
        Always returns:
          - pl.DataFrame (canonical)
        """
        raw_data_type = getattr(extractor, "raw_data_type", RawDataType.SPOT)

        try:
            return raw_data.get_data(raw_data_type, DataFrameType.POLARS)
        except TypeError:
            return raw_data.get_data(raw_data_type, "polars")

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
            right_feature_cols = [c for c in right.columns if c not in (self.pair_col, self.ts_col)]
            dup = set(right_feature_cols).intersection(set(combined.columns))
            if dup:
                raise ValueError(
                    f"Duplicate feature columns during FeatureSet combine: {sorted(dup)}. "
                    f"Rename features or set unique prefixes."
                )

            combined = combined.join(right, on=[self.pair_col, self.ts_col], how="outer")

        return combined
