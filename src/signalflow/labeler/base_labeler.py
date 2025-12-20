from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.core import RawDataType, SfComponentType, SignalType, Signals


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

    # Common params for signal masking
    mask_to_signals: bool = True
    out_col: str = "label"
    include_meta: bool = False
    meta_columns: tuple[str, ...] = ("t_hit", "ret")

    def compute(
        self,
        df: pl.DataFrame,
        signals: Signals | None = None,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.compute expects pl.DataFrame, got {type(df)}")
        return self._compute_pl(df=df, signals=signals, data_context=data_context)

    def _compute_pl(
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
        raise TypeError(f"Unsupported Signals.value type: {type(s)}")

    def _filter_by_signals_pl(
        self, df: pl.DataFrame, s: pl.DataFrame, signal_type: SignalType
    ) -> pl.DataFrame:
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
    def compute_group(
        self, group_df: pl.DataFrame, data_context: dict[str, Any] | None
    ) -> pl.DataFrame:
        """Polars implementation per pair."""
        raise NotImplementedError

    def _validate_input_pl(self, df: pl.DataFrame) -> None:
        missing = [c for c in (self.pair_col, self.ts_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _apply_signal_mask(
        self,
        df: pl.DataFrame,
        data_context: dict[str, Any],
        group_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Mask labels to signal timestamps only.

        Labels are computed for all rows, but only signal timestamps
        get actual labels; others are set to SignalType.NONE.

        Uses self.out_col, self.include_meta, and self.meta_columns.
        """
        signal_keys: pl.DataFrame = data_context["signal_keys"]
        pair_value = group_df.get_column(self.pair_col)[0]

        signal_ts = (
            signal_keys.filter(pl.col(self.pair_col) == pair_value)
            .select(self.ts_col)
            .unique()
        )

        if signal_ts.height == 0:
            df = df.with_columns(pl.lit(SignalType.NONE.value).alias(self.out_col))
            if self.include_meta:
                df = df.with_columns(
                    [pl.lit(None).alias(col) for col in self.meta_columns]
                )
        else:
            is_signal = pl.col("_is_signal").fill_null(False)
            mask_exprs = [
                pl.when(is_signal)
                .then(pl.col(self.out_col))
                .otherwise(pl.lit(SignalType.NONE.value))
                .alias(self.out_col),
            ]
            if self.include_meta:
                mask_exprs += [
                    pl.when(is_signal)
                    .then(pl.col(col))
                    .otherwise(pl.lit(None))
                    .alias(col)
                    for col in self.meta_columns
                ]

            df = (
                df.join(
                    signal_ts.with_columns(pl.lit(True).alias("_is_signal")),
                    on=self.ts_col,
                    how="left",
                )
                .with_columns(mask_exprs)
                .drop("_is_signal")
            )

        return df