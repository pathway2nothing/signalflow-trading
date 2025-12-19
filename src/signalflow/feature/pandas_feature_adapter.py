from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl
import pandas as pd

from signalflow.feature.base_extractor import FeatureExtractor


PandasGroupFn = Callable[[pd.DataFrame, dict[str, Any] | None], pd.DataFrame | pd.Series]


@dataclass
class PandasFeatureExtractor(FeatureExtractor):
    """
    Adapter: run pandas-based group feature function inside Polars-first pipeline.

    - Input to compute_pl_group: pl.DataFrame group (already sorted and prepared by FeatureExtractor)
    - Converts group to pandas
    - Runs pandas function that MUST preserve row count/order
    - Writes produced columns back to the group and returns pl.DataFrame
    """

    pandas_group_fn: PandasGroupFn

    out_cols: list[str] | None = None

    series_name: str = "feature"

    rename_outputs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not callable(self.pandas_group_fn):
            raise TypeError("pandas_group_fn must be callable")

    def compute_pl_group(self, group_df: pl.DataFrame, data_context: dict | None) -> pl.DataFrame:
        pdf = group_df.to_pandas()
        result = self.pandas_group_fn(pdf, data_context)

        if isinstance(result, pd.Series):
            if result.name is None:
                result = result.rename(self.series_name)
            result = result.to_frame()

        if not isinstance(result, pd.DataFrame):
            raise TypeError("pandas_group_fn must return pd.DataFrame or pd.Series")

        if len(result) != len(pdf):
            raise ValueError(
                f"pandas_group_fn must preserve row count: got {len(result)} != {len(pdf)}"
            )
        if self.rename_outputs:
            result = result.rename(columns=self.rename_outputs)
        if self.out_cols is not None:
            missing = set(self.out_cols) - set(result.columns)
            if missing:
                raise ValueError(f"pandas_group_fn output missing columns: {sorted(missing)}")

        out = group_df
        for col in result.columns:
            out = out.with_columns(pl.Series(col, result[col].to_numpy()))

        return out
