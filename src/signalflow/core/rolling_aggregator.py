from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl
from signalflow.core.enums import RawDataType


@dataclass
class RollingAggregator:
    """
    Offset (sliding) resampler for RawData (Polars-only).

    For each row t computes aggregates over last `offset_window` rows per pair:
      [t-(k-1), ..., t]

    Invariants:
      - len(out) == len(in)
      - (pair, timestamp) preserved
      - first (k-1) rows per pair -> nulls for resampled fields (min_periods=k)
    """

    offset_window: int = 1
    ts_col: str = "timestamp"
    pair_col: str = "pair"
    mode: Literal["add", "replace"] = "replace"
    prefix: str | None = None
    raw_data_type: RawDataType = RawDataType.SPOT 

    OFFSET_COL: str = "resample_offset"

    @property
    def out_prefix(self) -> str:
        return self.prefix if self.prefix is not None else f"rs_{self.offset_window}m_"

    def _validate_base(self, df: pl.DataFrame) -> None:
        if self.offset_window <= 0:
            raise ValueError(f"offset_window must be > 0, got {self.offset_window}")
        if self.ts_col not in df.columns:
            raise ValueError(f"Missing '{self.ts_col}' column")
        if self.pair_col not in df.columns:
            raise ValueError(f"Missing '{self.pair_col}' column")

    def add_offset_column(self, df: pl.DataFrame) -> pl.DataFrame:
        self._validate_base(df)

        return df.with_columns(
            (pl.col(self.ts_col).dt.minute() % pl.lit(self.offset_window)).cast(pl.Int64).alias(self.OFFSET_COL)
        )

    def get_last_offset(self, df: pl.DataFrame) -> int:
        self._validate_base(df)
        if df.is_empty():
            raise ValueError("Empty dataframe")

        last_ts = df.select(pl.col(self.ts_col).max()).item()
        return int(last_ts.minute % self.offset_window)

    def _spot_validate(self, cols: list[str]) -> None:
        required = {"open", "high", "low", "close"}
        missing = required - set(cols)
        if missing:
            raise ValueError(f"spot resample requires columns {sorted(required)}; missing {sorted(missing)}")

    def resample(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.raw_data_type != RawDataType.SPOT:
            raise NotImplementedError("Currently resample() implemented for data_type='spot' only")

        self._validate_base(df)
        self._spot_validate(df.columns)

        df0 = df.sort([self.pair_col, self.ts_col])

        if self.OFFSET_COL not in df0.columns:
            df0 = self.add_offset_column(df0)

        k = int(self.offset_window)
        pfx = self.out_prefix
        over = [self.pair_col]

        rs_open = pl.col("open").shift(k - 1).over(over)
        rs_high = pl.col("high").rolling_max(window_size=k, min_periods=k).over(over)
        rs_low = pl.col("low").rolling_min(window_size=k, min_periods=k).over(over)
        rs_close = pl.col("close")

        has_volume = "volume" in df0.columns
        has_trades = "trades" in df0.columns

        if self.mode == "add":
            exprs: list[pl.Expr] = [
                rs_open.alias(f"{pfx}open"),
                rs_high.alias(f"{pfx}high"),
                rs_low.alias(f"{pfx}low"),
                rs_close.alias(f"{pfx}close"),
            ]
            if has_volume:
                exprs.append(
                    pl.col("volume")
                    .rolling_sum(window_size=k, min_periods=k)
                    .over(over)
                    .alias(f"{pfx}volume")
                )
            if has_trades:
                exprs.append(
                    pl.col("trades")
                    .rolling_sum(window_size=k, min_periods=k)
                    .over(over)
                    .alias(f"{pfx}trades")
                )
            out = df0.with_columns(exprs)

        elif self.mode == "replace":
            exprs2: list[pl.Expr] = [
                rs_open.alias("open"),
                rs_high.alias("high"),
                rs_low.alias("low"),
                rs_close.alias("close"),
            ]
            if has_volume:
                exprs2.append(
                    pl.col("volume").rolling_sum(window_size=k, min_periods=k).over(over).alias("volume")
                )
            if has_trades:
                exprs2.append(
                    pl.col("trades").rolling_sum(window_size=k, min_periods=k).over(over).alias("trades")
                )
            out = df0.with_columns(exprs2)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if out.height != df.height:
            raise ValueError(f"resample(pl): len(out)={out.height} != len(in)={df.height}")

        return out