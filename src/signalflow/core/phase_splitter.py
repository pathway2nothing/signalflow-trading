from __future__ import annotations

from dataclasses import dataclass
from typing import List

import polars as pl
import pandas as pd


@dataclass(frozen=True)
class PhaseSplitter:
    """Split time-indexed data into phase subsets.

    Phase is defined as:
        phase = (timestamp_in_minutes_since_epoch % tf)

    Example:
        tf=15 -> phases:
            [0,15,30,45], [1,16,31,46], ..., [14,29,44,59]
    """

    tf: int
    ts_col: str = "timestamp"
    keep_phase_col: bool = False
    phase_col: str = "__phase"

    def split_pl(self, df: pl.DataFrame) -> List[pl.DataFrame]:
        """Split Polars DataFrame into tf phases."""
        self._validate_tf()
        self._validate_pl(df)

        if self.tf <= 1:
            return [df]

        phase_expr = (
            (pl.col(self.ts_col).dt.timestamp("ms") // 60_000) % self.tf
        ).cast(pl.Int64)

        df2 = df.with_columns(phase_expr.alias(self.phase_col))
        out: List[pl.DataFrame] = []

        for k in range(self.tf):
            part = df2.filter(pl.col(self.phase_col) == k)
            if not self.keep_phase_col:
                part = part.drop(self.phase_col)
            out.append(part)

        return out

    def split_pd(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split pandas DataFrame into tf phases."""
        self._validate_tf()
        self._validate_pd(df)

        if self.tf <= 1:
            return [df]

        ts = pd.to_datetime(df[self.ts_col], utc=True, errors="raise")
        phase = (ts.view("int64") // 60_000_000_000) % self.tf  # ns -> minutes

        if self.keep_phase_col:
            df2 = df.copy()
            df2[self.phase_col] = phase.astype("int64")
        else:
            df2 = df

        out: List[pd.DataFrame] = []
        for k in range(self.tf):
            part = df2.loc[phase == k]
            if not self.keep_phase_col:
                part = part.copy()
            out.append(part)

        return out

    def _validate_tf(self) -> None:
        if not isinstance(self.tf, int) or self.tf <= 0:
            raise ValueError(f"tf must be positive int, got {self.tf!r}")

    def _validate_pl(self, df: pl.DataFrame) -> None:
        if self.ts_col not in df.columns:
            raise ValueError(f"Missing '{self.ts_col}' column")
        if not isinstance(df.schema[self.ts_col], pl.Datetime):
            raise TypeError(f"'{self.ts_col}' must be polars Datetime")

    def _validate_pd(self, df: pd.DataFrame) -> None:
        if self.ts_col not in df.columns:
            raise ValueError(f"Missing '{self.ts_col}' column")
