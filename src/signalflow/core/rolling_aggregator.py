from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl
import pandas as pd


@dataclass
class RollingAggregator:
    """
    Offset (sliding) resampler for RawData.

    For each row t computes aggregates over last `offset_window` rows per pair:
      [t-(k-1), ..., t]

    Invariants:
      - len(out) == len(in)
      - (pair, timestamp) preserved
      - first (k-1) rows per pair -> null/NaN for resampled fields (min_periods=k)
    """

    offset_window: int = 1
    ts_col: str = "timestamp"
    pair_col: str = "pair"
    mode: Literal["add", "replace"] = "replace"
    prefix: str | None = None
    data_type: str = "spot"

    OFFSET_COL: str = "resample_offset"

    @property
    def out_prefix(self) -> str:
        return self.prefix if self.prefix is not None else f"rs_{self.offset_window}m_"

    def add_offset_column(self, df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame | pd.DataFrame:
        if self.offset_window <= 0:
            raise ValueError(f"offset_window must be > 0, got {self.offset_window}")

        if isinstance(df, pl.DataFrame):
            if self.ts_col not in df.columns:
                raise ValueError(f"Missing '{self.ts_col}' column")
            return df.with_columns(
                (pl.col(self.ts_col).dt.minute() % self.offset_window).alias(self.OFFSET_COL)
            )

        if isinstance(df, pd.DataFrame):
            if self.ts_col not in df.columns:
                raise ValueError(f"Missing '{self.ts_col}' column")
            out = df.copy()
            ts = pd.to_datetime(out[self.ts_col], utc=False, errors="raise")
            out[self.OFFSET_COL] = (ts.dt.minute % self.offset_window).astype("int64")
            return out

        raise TypeError(f"Unsupported df type: {type(df)}")

    def get_last_offset(self, df: pl.DataFrame | pd.DataFrame) -> int:
        if isinstance(df, pl.DataFrame):
            if df.is_empty():
                raise ValueError("Empty dataframe")
            last_ts = df.select(pl.col(self.ts_col).max()).item()
            return int(last_ts.minute % self.offset_window)

        if isinstance(df, pd.DataFrame):
            if df.empty:
                raise ValueError("Empty dataframe")
            last_ts = pd.to_datetime(df[self.ts_col], utc=False, errors="raise").max()
            return int(last_ts.minute % self.offset_window)

        raise TypeError(f"Unsupported df type: {type(df)}")

    def _spot_validate(self, cols: list[str]) -> None:
        required = {"open", "high", "low", "close"}
        missing = required - set(cols)
        if missing:
            raise ValueError(f"spot resample requires columns {sorted(required)}; missing {sorted(missing)}")

    def resample(self, df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame | pd.DataFrame:
        if self.data_type != "spot":
            raise NotImplementedError("Currently resample() implemented for data_type='spot' only")

        if isinstance(df, pl.DataFrame):
            return self._resample_pl(df)
        if isinstance(df, pd.DataFrame):
            return self._resample_pd(df)
        raise TypeError(f"Unsupported df type: {type(df)}")

    def _resample_pl(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.offset_window <= 0:
            raise ValueError(f"offset_window must be > 0, got {self.offset_window}")
        if self.pair_col not in df.columns or self.ts_col not in df.columns:
            raise ValueError(f"Input must contain '{self.pair_col}' and '{self.ts_col}'")

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
                    pl.col("volume").rolling_sum(window_size=k, min_periods=k).over(over).alias(f"{pfx}volume")
                )
            if has_trades:
                exprs.append(
                    pl.col("trades").rolling_sum(window_size=k, min_periods=k).over(over).alias(f"{pfx}trades")
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

    def _resample_pd(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.offset_window <= 0:
            raise ValueError(f"offset_window must be > 0, got {self.offset_window}")
        if self.pair_col not in df.columns or self.ts_col not in df.columns:
            raise ValueError(f"Input must contain '{self.pair_col}' and '{self.ts_col}'")

        self._spot_validate(list(df.columns))

        df0 = df.sort_values([self.pair_col, self.ts_col], kind="stable").copy()

        if self.OFFSET_COL not in df0.columns:
            df0 = self.add_offset_column(df0)  

        k = int(self.offset_window)
        pfx = self.out_prefix

        g = df0.groupby(self.pair_col, sort=False)

        rs_open = g["open"].shift(k - 1)
        rs_high = g["high"].rolling(window=k, min_periods=k).max().reset_index(level=0, drop=True)
        rs_low = g["low"].rolling(window=k, min_periods=k).min().reset_index(level=0, drop=True)
        rs_close = df0["close"]

        has_volume = "volume" in df0.columns
        has_trades = "trades" in df0.columns

        if has_volume:
            rs_volume = g["volume"].rolling(window=k, min_periods=k).sum().reset_index(level=0, drop=True)
        if has_trades:
            rs_trades = g["trades"].rolling(window=k, min_periods=k).sum().reset_index(level=0, drop=True)

        if self.mode == "add":
            out = df0.copy()
            out[f"{pfx}open"] = rs_open
            out[f"{pfx}high"] = rs_high
            out[f"{pfx}low"] = rs_low
            out[f"{pfx}close"] = rs_close
            if has_volume:
                out[f"{pfx}volume"] = rs_volume 
            if has_trades:
                out[f"{pfx}trades"] = rs_trades 

        elif self.mode == "replace":
            out = df0.copy()
            out["open"] = rs_open
            out["high"] = rs_high
            out["low"] = rs_low
            out["close"] = rs_close
            if has_volume:
                out["volume"] = rs_volume 
            if has_trades:
                out["trades"] = rs_trades 

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if len(out) != len(df):
            raise ValueError(f"resample(pd): len(out)={len(out)} != len(in)={len(df)}")

        return out.reset_index(drop=True)
