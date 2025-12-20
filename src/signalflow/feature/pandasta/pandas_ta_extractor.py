from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from signalflow.feature.adapter.pandas_feature_extractor import PandasFeatureExtractor


@dataclass
class PandasTaExtractor(PandasFeatureExtractor):
    """
    Polars-first Pandas-TA adapter.

    This extractor runs pandas-ta inside `pandas_group_fn` per (pair, resample_offset) group,
    then merges produced feature columns back into the Polars pipeline.

    Key guarantees:
      - pandas-ta output is normalized to pd.DataFrame
      - output length matches input group length
      - output columns are namespaced to avoid collisions across extractors
    """

    indicator: str = "rsi"
    params: dict[str, Any] = field(default_factory=dict)

    input_column: str = "close"
    additional_inputs: dict[str, str] = field(default_factory=dict)

    feature_prefix: str | None = None

    def __post_init__(self) -> None:
        try:
            import pandas_ta as _  
        except ImportError as e:
            raise ImportError("pandas-ta is required. Install with: pip install pandas-ta") from e

        if not isinstance(self.indicator, str) or not self.indicator.strip():
            raise ValueError("indicator name must be a non-empty string")

        if not isinstance(self.input_column, str) or not self.input_column.strip():
            raise ValueError("input_column must be a non-empty string")

        if not isinstance(self.params, dict):
            raise TypeError(f"params must be dict[str, Any], got {type(self.params)}")

        if not isinstance(self.additional_inputs, dict):
            raise TypeError(f"additional_inputs must be dict[str, str], got {type(self.additional_inputs)}")

        for k, v in self.additional_inputs.items():
            if not isinstance(k, str) or not k.strip():
                raise TypeError(f"additional_inputs keys must be non-empty str, got {k!r}")
            if not isinstance(v, str) or not v.strip():
                raise TypeError(f"additional_inputs values must be non-empty str column names, got {v!r}")

        self.pandas_group_fn = self._pandas_ta_group_fn

        super().__post_init__()

    def _pandas_ta_group_fn(self, group: pd.DataFrame, ctx: dict[str, Any] | None) -> pd.DataFrame:
        import pandas_ta as ta

        self._validate_required_columns(group)

        try:
            indicator_func = getattr(ta, self.indicator)
        except AttributeError as e:
            raise AttributeError(f"Indicator '{self.indicator}' not found in pandas-ta.") from e

        kwargs = dict(self.params)

        primary_input = group[self.input_column]
        for param_name, column_name in self.additional_inputs.items():
            kwargs[param_name] = group[column_name]

        out = indicator_func(primary_input, **kwargs)

        out_df = self._normalize_output(out, group_len=len(group))
        out_df = self._namespace_columns(out_df)

        return out_df

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        required = [self.input_column, *self.additional_inputs.values()]
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(f"Missing required columns for pandas-ta: {missing}")

    def _normalize_output(self, out: Any, group_len: int) -> pd.DataFrame:
        """
        Normalize pandas-ta output to pd.DataFrame and ensure length matches group.
        """
        if isinstance(out, pd.Series):
            out_df = out.to_frame()
            col = out_df.columns[0]
            if col is None or (isinstance(col, str) and not col.strip()):
                out_df.columns = [self.indicator]
        elif isinstance(out, pd.DataFrame):
            out_df = out
            if out_df.columns.isnull().any():
                out_df = out_df.copy()
                out_df.columns = [
                    c if (c is not None and (not isinstance(c, str) or c.strip())) else f"{self.indicator}_{i}"
                    for i, c in enumerate(out_df.columns)
                ]
        else:
            raise TypeError(
                f"pandas-ta '{self.indicator}' returned unsupported type: {type(out)}. "
                f"Expected pd.Series or pd.DataFrame."
            )

        if len(out_df) != group_len:
            raise ValueError(
                f"{self.__class__.__name__}: len(output_group)={len(out_df)} != len(input_group)={group_len}"
            )

        return out_df

    def _namespace_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prefix output columns to avoid collisions across different indicators/extractors.
        """
        prefix = self.feature_prefix or self.indicator
        prefix = str(prefix).strip()

        df = df.copy()
        new_cols: list[str] = []
        for i, c in enumerate(df.columns):
            name = str(c) if c is not None else f"{self.indicator}_{i}"
            name = name.strip() or f"{self.indicator}_{i}"

            if name == prefix or name.startswith(prefix + "_"):
                new_cols.append(name)
            else:
                new_cols.append(f"{prefix}_{name}")

        df.columns = new_cols
        return df