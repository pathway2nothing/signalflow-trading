from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.core import RawDataType, SfComponentType, SignalType, Signals


# =========================
# Polars-only base (public interface)
# =========================

@dataclass
class Labeler(ABC):
    """
    Polars-only base for labeling (NO offset/resample logic).

    Public contract:
      - input: pl.DataFrame
      - output: pl.DataFrame
      - per-pair (group_by pair_col)
      - compute_group MUST preserve row count (no filtering inside compute)
      - optional filtering by Signals BEFORE grouping
      - projection logic identical across implementations
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.LABELER
    raw_data_type: RawDataType = RawDataType.SPOT

    pair_col: str = "pair"
    ts_col: str = "timestamp"

    keep_input_columns: bool = False
    output_columns: list[str] | None = None
    filter_signal_type: SignalType | None = None

    def extract(
        self,
        df: pl.DataFrame,
        signals: Signals | None = None,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.extract expects pl.DataFrame, got {type(df)}")
        return self._extract_pl(df=df, signals=signals, data_context=data_context)

    def _extract_pl(
        self,
        df: pl.DataFrame,
        signals: Signals | None,
        data_context: dict[str, Any] | None,
    ) -> pl.DataFrame:
        self._validate_input_pl(df)
        df0 = df.sort([self.pair_col, self.ts_col])

        if signals is not None and self.filter_signal_type is not None:
            s_pl = self._signals_to_pl(signals)
            df0 = self._filter_by_signals_pl(df0, s_pl, self.filter_signal_type)

        input_cols = set(df0.columns)

        def _wrapped(g: pl.DataFrame) -> pl.DataFrame:
            out = self.compute_group(g, data_context=data_context)
            if not isinstance(out, pl.DataFrame):
                raise TypeError(f"{self.__class__.__name__}.compute_group must return pl.DataFrame")
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

        label_cols = (
            sorted(set(out.columns) - input_cols)
            if self.output_columns is None
            else list(self.output_columns)
        )

        keep_cols = [self.pair_col, self.ts_col] + label_cols
        missing = [c for c in keep_cols if c not in out.columns]
        if missing:
            raise ValueError(f"Projection error, missing columns: {missing}")

        return out.select(keep_cols)

    def _signals_to_pl(self, signals: Signals) -> pl.DataFrame:
        s = signals.value
        if isinstance(s, pl.DataFrame):
            return s
        else:
            raise TypeError(f"Unsupported Signals.value type: {type(s)}")

    def _filter_by_signals_pl(self, df: pl.DataFrame, s: pl.DataFrame, signal_type: SignalType) -> pl.DataFrame:
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
    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
        """Polars implementation per pair."""
        raise NotImplementedError

    def _validate_input_pl(self, df: pl.DataFrame) -> None:
        missing = [c for c in (self.pair_col, self.ts_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")