from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl
import pandas as pd

from signalflow.core import DataFrameType, SfComponentType, SignalType, Signals



@dataclass
class Labeler(ABC):
    """
    FeatureExtractor-like base for labeling (NO offset/resample logic).

    Contract:
      - input: df with at least [pair_col, ts_col]
      - processing is per-pair (group_by pair_col)
      - compute_*_group MUST preserve row count (no filtering inside compute)
      - optional filtering by Signals BEFORE grouping
      - output projection:
          - keep_input_columns=True  -> return full output
          - keep_input_columns=False -> return [pair, ts] + output_columns
            (or auto-detect newly added cols if output_columns is None)
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.DETECTOR
    df_type: DataFrameType = DataFrameType.POLARS
    raw_data_type: str = "spot"

    pair_col: str = "pair"
    ts_col: str = "timestamp"

    keep_input_columns: bool = False
    output_columns: list[str] | None = None 
    filter_signal_type: SignalType | None = None

    def extract(
        self,
        df: pl.DataFrame | pd.DataFrame,
        signals: Signals | None = None,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame | pd.DataFrame:
        if isinstance(df, pl.DataFrame):
            return self._extract_pl(df, signals=signals, data_context=data_context)
        if isinstance(df, pd.DataFrame):
            return self._extract_pd(df, signals=signals, data_context=data_context)
        raise TypeError(f"Unsupported df type: {type(df)}")

    def _extract_pl(
        self,
        df: pl.DataFrame,
        signals: Signals | None,
        data_context: dict[str, Any] | None,
    ) -> pl.DataFrame:
        self._validate_input_pl(df)

        df0 = df.sort([self.pair_col, self.ts_col])

        if signals is not None and self.filter_signal_type is not None:
            df0 = self._filter_by_signals_pl(df0, signals, self.filter_signal_type)

        input_cols = set(df0.columns)

        def _wrapped(g: pl.DataFrame) -> pl.DataFrame:
            out = self.compute_pl_group(g, data_context=data_context)

            if not isinstance(out, pl.DataFrame):
                raise TypeError(f"{self.__class__.__name__}.compute_pl_group must return pl.DataFrame")
            if out.height != g.height:
                raise ValueError(
                    f"{self.__class__.__name__}: len(output_group)={out.height} != len(input_group)={g.height}"
                )
            return out

        out = (
            df0.group_by(self.pair_col, maintain_order=True)
            .map_groups(_wrapped)
            .sort([self.pair_col, self.ts_col])
        )

        if self.keep_input_columns:
            return out

        if self.output_columns is None:
            label_cols = sorted(set(out.columns) - input_cols)
        else:
            label_cols = list(self.output_columns)

        keep_cols = [self.pair_col, self.ts_col] + label_cols

        missing = [c for c in keep_cols if c not in out.columns]
        if missing:
            raise ValueError(f"Projection error, missing columns: {missing}")

        return out.select(keep_cols)

    def _filter_by_signals_pl(self, df: pl.DataFrame, signals: Signals, signal_type: SignalType) -> pl.DataFrame:
        s = signals.value  
        required = {self.pair_col, self.ts_col, "signal_type"}
        missing = required - set(s.columns)
        if missing:
            raise ValueError(f"Signals missing columns: {sorted(missing)}")

        s_f = (
            s.filter(pl.col("signal_type") == signal_type.value)
            .select([self.pair_col, self.ts_col])
            .unique(subset=[self.pair_col, self.ts_col])
        )
        return df.join(s_f, on=[self.pair_col, self.ts_col], how="inner")

    @abstractmethod
    def compute_pl_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
        """
        Per-pair labeling.

        MUST:
          - preserve row order
          - preserve row count (no filtering)
        """
        raise NotImplementedError

    def _validate_input_pl(self, df: pl.DataFrame) -> None:
        missing = [c for c in (self.pair_col, self.ts_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _extract_pd(
        self,
        df: pd.DataFrame,
        signals: Signals | None,
        data_context: dict[str, Any] | None,
    ) -> pd.DataFrame:
        self._validate_input_pd(df)

        df0 = df.sort_values([self.pair_col, self.ts_col], kind="stable").copy()

        if signals is not None and self.filter_signal_type is not None:
            df0 = self._filter_by_signals_pd(df0, signals, self.filter_signal_type)

        input_cols = set(df0.columns)

        parts: list[pd.DataFrame] = []
        for _, g in df0.groupby(self.pair_col, sort=False, dropna=False):
            out_g = self.compute_pd_group(g, data_context=data_context)
            if not isinstance(out_g, pd.DataFrame):
                raise TypeError(f"{self.__class__.__name__}.compute_pd_group must return pd.DataFrame")
            if len(out_g) != len(g):
                raise ValueError(
                    f"{self.__class__.__name__}: len(output_group)={len(out_g)} != len(input_group)={len(g)}"
                )
            parts.append(out_g)

        out = pd.concat(parts, axis=0, ignore_index=True) if parts else df0.iloc[:0].copy()
        out = out.sort_values([self.pair_col, self.ts_col], kind="stable").reset_index(drop=True)

        if self.keep_input_columns:
            return out

        if self.output_columns is None:
            label_cols = [c for c in out.columns if c not in input_cols]
        else:
            label_cols = list(self.output_columns)

        keep_cols = [self.pair_col, self.ts_col] + label_cols

        missing = [c for c in keep_cols if c not in out.columns]
        if missing:
            raise ValueError(f"Projection error, missing columns: {missing}")

        return out.loc[:, keep_cols]

    def _filter_by_signals_pd(self, df: pd.DataFrame, signals: Signals, signal_type: SignalType) -> pd.DataFrame:
        s = signals.value
        if isinstance(s, pl.DataFrame):
            s = s.to_pandas()

        required = {self.pair_col, self.ts_col, "signal_type"}
        missing = required - set(s.columns)
        if missing:
            raise ValueError(f"Signals missing columns: {sorted(missing)}")

        s_f = s.loc[s["signal_type"] == signal_type.value, [self.pair_col, self.ts_col]].drop_duplicates()
        return df.merge(s_f, on=[self.pair_col, self.ts_col], how="inner")

    def compute_pd_group(self, group_df: pd.DataFrame, data_context: dict[str, Any] | None) -> pd.DataFrame:
        """
        Default pandas path: convert -> polars compute -> back.
        Override in concrete labelers for speed.
        """
        pl_in = pl.from_pandas(group_df)
        pl_out = self.compute_pl_group(pl_in, data_context=data_context)
        if pl_out.height != pl_in.height:
            raise ValueError(
                f"{self.__class__.__name__}: len(output_group)={pl_out.height} != len(input_group)={pl_in.height}"
            )
        return pl_out.to_pandas()

    def _validate_input_pd(self, df: pd.DataFrame) -> None:
        missing = [c for c in (self.pair_col, self.ts_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
