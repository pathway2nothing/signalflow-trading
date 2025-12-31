from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from signalflow.core import RawDataType, RollingAggregator, SfComponentType
from typing import ClassVar


@dataclass
class FeatureExtractor(ABC):
    """
    Polars-first base FeatureExtractor.

    Contract:
      - Input:  pl.DataFrame only
      - Output: pl.DataFrame only
      - Any other DF type must be handled by an adapter (e.g. PandasFeatureExtractor).

    Steps:
      1) sort (pair, timestamp)
      2) ensure resample_offset
      3) optional sliding resample
      4) optional last-offset filter
      5) group (pair, resample_offset) and compute features
      6) sort output
      7) optional projection:
           - keep_input_columns=True  -> return everything
           - keep_input_columns=False -> return only [pair, timestamp] + new feature cols
    """

    offset_window: int = 1
    compute_last_offset: bool = False

    pair_col: str = "pair"
    ts_col: str = "timestamp"
    offset_col: str = "resample_offset"

    use_resample: bool = False
    resample_mode: Literal["add", "replace"] = "add"
    resample_prefix: str | None = None
    raw_data_type: RawDataType = RawDataType.SPOT
    component_type: ClassVar[SfComponentType] = SfComponentType.FEATURE_EXTRACTOR
    keep_input_columns: bool = False

    def __post_init__(self) -> None:
        if self.offset_window <= 0:
            raise ValueError(f"offset_window must be > 0, got {self.offset_window}")

        if self.resample_mode not in ("add", "replace"):
            raise ValueError(f"Invalid resample_mode: {self.resample_mode}")

        if self.offset_col != RollingAggregator.OFFSET_COL:
            raise ValueError(
                f"offset_col must be '{RollingAggregator.OFFSET_COL}', got '{self.offset_col}'"
            )

        if not isinstance(self.pair_col, str) or not isinstance(self.ts_col, str) or not isinstance(self.offset_col, str):
            raise TypeError("pair_col/ts_col/offset_col must be str")

    @property
    def _resampler(self) -> RollingAggregator:
        return RollingAggregator(
            offset_window=self.offset_window,
            ts_col=self.ts_col,
            pair_col=self.pair_col,
            mode=self.resample_mode,
            prefix=self.resample_prefix,
            raw_data_type=self.raw_data_type,
        )

    def extract(self, df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        if not isinstance(df, pl.DataFrame):
            raise TypeError(
                f"{self.__class__.__name__} is polars-first and accepts only pl.DataFrame. "
                f"Got: {type(df)}. Use an adapter for other dataframe types."
            )
        self._validate_input(df)

        df0 = df.sort([self.pair_col, self.ts_col])

        if self.offset_col not in df0.columns:
            df0 = self._resampler.add_offset_column(df0)

        if self.use_resample:
            df0 = self._resampler.resample(df0)

        if self.compute_last_offset:
            last_off = self._resampler.get_last_offset(df0)
            df0 = df0.filter(pl.col(self.offset_col) == last_off)

        prepared_cols = set(df0.columns)
        inferred_features: set[str] = set()

        def _wrapped(g: pl.DataFrame) -> pl.DataFrame:
            nonlocal inferred_features

            in_cols = set(g.columns)
            out = self.compute_group(g, data_context=data_context)

            if not isinstance(out, pl.DataFrame):
                raise TypeError(f"{self.__class__.__name__}.compute_pl_group must return pl.DataFrame")

            if out.height != g.height:
                raise ValueError(
                    f"{self.__class__.__name__}: len(output_group)={out.height} != len(input_group)={g.height}"
                )

            if not inferred_features:
                inferred_features = set(out.columns) - in_cols

            return out

        out = (
            df0.group_by(self.pair_col, self.offset_col, maintain_order=True)
            .map_groups(_wrapped)
            .sort([self.pair_col, self.ts_col])
        )

        if self.keep_input_columns:
            return out

        feature_cols = sorted(set(out.columns) - prepared_cols)
        keep_cols = [self.pair_col, self.ts_col] + feature_cols

        missing = [c for c in keep_cols if c not in out.columns]
        if missing:
            raise ValueError(f"Projection error, missing columns: {missing}")

        return out.select(keep_cols)

    def compute_group(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None,
    ) -> pl.DataFrame:
        """
        Compute features for a single (pair, resample_offset) group.

        Requirements:
          - Must return pl.DataFrame
          - Must preserve row count (out.height == group_df.height)
          - Should preserve ordering inside the group
        """
        raise NotImplementedError

    def _validate_input(self, df: pl.DataFrame) -> None:
        missing = [c for c in (self.pair_col, self.ts_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
