from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd

from signalflow.core import DataFrameType, SfComponentType
from signalflow.feature.extractor.base_extractor import FeatureExtractor


@dataclass
class PandasTaExtractor(FeatureExtractor):
    """
    Adapter for pandas-ta indicators integrated into FeatureExtractor pipeline.

    - Uses FeatureExtractor.extract() to:
        * ensure resample_offset
        * optionally resample
        * group by (pair, resample_offset)
    - Implements compute_pd_group() to run pandas-ta for one group.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.FEATURE_EXTRACTOR

    indicator: str = "rsi"
    params: dict[str, Any] = field(default_factory=dict)

    input_column: str = "close"
    additional_inputs: dict[str, str] = field(default_factory=dict)

    rename_outputs: dict[str, str] = field(default_factory=dict)

    df_type: DataFrameType = DataFrameType.PANDAS
    phased_computation: bool = True 
    include_index_columns: bool = True 

    def __post_init__(self) -> None:
        super().__post_init__()

        try:
            import pandas_ta as ta  
        except ImportError:
            raise ImportError(
                "pandas-ta is required for PandasTaExtractor. Install with: pip install pandas-ta"
            )

        if not self.indicator:
            raise ValueError("indicator name must be specified")

        if self.include_index_columns not in (True, False):
            raise ValueError("include_index_columns must be bool")

    def compute_pd_group(self, group_df: pd.DataFrame, data_context: dict | None) -> pd.DataFrame:
        """
        group_df is one (pair, resample_offset) group, already sorted by base extractor.

        MUST:
          - preserve row count and order
        """
        self._validate_required_columns(group_df)

        result = self._calculate_indicator(group_df)

        if isinstance(result, pd.Series):
            result = result.to_frame()

        result = self._normalize_column_names(result)

        if self.rename_outputs:
            result = result.rename(columns=self.rename_outputs)

        out = group_df.copy()
        for col in result.columns:
            out[col] = result[col].to_numpy()

        return out

    def _calculate_indicator(self, group: pd.DataFrame) -> pd.DataFrame | pd.Series:
        import pandas_ta as ta

        try:
            indicator_func = getattr(ta, self.indicator)
        except AttributeError:
            raise AttributeError(
                f"Indicator '{self.indicator}' not found in pandas-ta."
            )

        kwargs = dict(self.params)

        primary_input = group[self.input_column]
        for param_name, column_name in self.additional_inputs.items():
            kwargs[param_name] = group[column_name]

        return indicator_func(primary_input, **kwargs)

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map: dict[str, str] = {}
        for col in df.columns:
            normalized = str(col).lower().replace(".", "_")
            rename_map[col] = normalized
        return df.rename(columns=rename_map)

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        required = [self.pair_col, self.ts_col, self.input_column]
        required.extend(self.additional_inputs.values())
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def postprocess_output(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.include_index_columns:
            drop_cols = [c for c in (self.pair_col, self.ts_col) if c in df.columns]
            return df.drop(columns=drop_cols)
        return df

@dataclass
class PandasTaRsiExtractor(PandasTaExtractor):
    length: int = 14

    def __post_init__(self) -> None:
        self.indicator = "rsi"
        self.params = {"length": int(self.length)}
        self.input_column = "close"
        super().__post_init__()


@dataclass
class PandasTaBbandsExtractor(PandasTaExtractor):
    length: int = 20
    std: float = 2.0

    def __post_init__(self) -> None:
        self.indicator = "bbands"
        self.params = {"length": int(self.length), "std": float(self.std)}
        self.input_column = "close"
        super().__post_init__()


@dataclass
class PandasTaMacdExtractor(PandasTaExtractor):
    fast: int = 12
    slow: int = 26
    signal: int = 9

    def __post_init__(self) -> None:
        self.indicator = "macd"
        self.params = {"fast": int(self.fast), "slow": int(self.slow), "signal": int(self.signal)}
        self.input_column = "close"
        super().__post_init__()


@dataclass
class PandasTaAtrExtractor(PandasTaExtractor):
    length: int = 14

    def __post_init__(self) -> None:
        self.indicator = "atr"
        self.params = {"length": int(self.length)}
        self.input_column = "high"
        self.additional_inputs = {"low": "low", "close": "close"}
        super().__post_init__()
