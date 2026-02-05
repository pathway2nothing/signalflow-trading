from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import polars as pl
from numba import njit, prange

from signalflow.core import sf_component, SignalType
from signalflow.target.base import Labeler


@njit(parallel=True, cache=True)
def _find_first_hit(
    prices: np.ndarray,
    pt: np.ndarray,
    sl: np.ndarray,
    lookforward: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(prices)
    up_off = np.zeros(n, dtype=np.int32)
    dn_off = np.zeros(n, dtype=np.int32)

    for i in prange(n):
        pt_i = pt[i]
        sl_i = sl[i]

        if np.isnan(pt_i) or np.isnan(sl_i):
            continue

        max_j = min(i + lookforward, n - 1)

        for k in range(1, max_j - i + 1):
            j = i + k
            p = prices[j]
            if up_off[i] == 0 and p >= pt_i:
                up_off[i] = k
            if dn_off[i] == 0 and p <= sl_i:
                dn_off[i] = k
            if up_off[i] > 0 and dn_off[i] > 0:
                break

    return up_off, dn_off


@dataclass
@sf_component(name="triple_barrier")
class TripleBarrierLabeler(Labeler):
    """
    Triple-Barrier Labeling (De Prado), Numba-accelerated.

    Volatility-based barriers:
      - pt = close * exp(vol * profit_multiplier)
      - sl = close * exp(-vol * stop_loss_multiplier)
    """

    price_col: str = "close"

    vol_window: int = 60
    lookforward_window: int = 1440
    profit_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.vol_window <= 1:
            raise ValueError("vol_window must be > 1")
        if self.lookforward_window <= 0:
            raise ValueError("lookforward_window must be > 0")
        if self.profit_multiplier <= 0 or self.stop_loss_multiplier <= 0:
            raise ValueError("profit_multiplier/stop_loss_multiplier must be > 0")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        if group_df.height == 0:
            return group_df

        lf = int(self.lookforward_window)
        vw = int(self.vol_window)

        df = group_df.with_columns(
            (pl.col(self.price_col) / pl.col(self.price_col).shift(1))
            .log()
            .rolling_std(window_size=vw, ddof=1)
            .alias("_vol")
        ).with_columns(
            [
                (pl.col(self.price_col) * (pl.col("_vol") * self.profit_multiplier).exp()).alias("_pt"),
                (pl.col(self.price_col) * (-pl.col("_vol") * self.stop_loss_multiplier).exp()).alias("_sl"),
            ]
        )

        prices = df.get_column(self.price_col).to_numpy().astype(np.float64)
        pt = df.get_column("_pt").fill_null(np.nan).to_numpy().astype(np.float64)
        sl = df.get_column("_sl").fill_null(np.nan).to_numpy().astype(np.float64)

        up_off, dn_off = _find_first_hit(prices, pt, sl, lf)

        up_off_series = pl.Series("_up_off", up_off).replace(0, None).cast(pl.Int32)
        dn_off_series = pl.Series("_dn_off", dn_off).replace(0, None).cast(pl.Int32)

        df = df.with_columns([up_off_series, dn_off_series])

        df = self._apply_labels(df)

        if self.include_meta:
            df = self._compute_meta(df, prices, up_off_series, dn_off_series, lf)

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        drop_cols = ["_vol", "_pt", "_sl", "_up_off", "_dn_off"]
        df = df.drop([c for c in drop_cols if c in df.columns])

        return df

    def _apply_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply RISE/FALL/NONE labels based on barrier hits."""
        choose_up = pl.col("_up_off").is_not_null() & (
            pl.col("_dn_off").is_null() | (pl.col("_up_off") <= pl.col("_dn_off"))
        )
        choose_dn = pl.col("_dn_off").is_not_null() & (
            pl.col("_up_off").is_null() | (pl.col("_dn_off") < pl.col("_up_off"))
        )

        return df.with_columns(
            pl.when(choose_up)
            .then(pl.lit(SignalType.RISE.value))
            .when(choose_dn)
            .then(pl.lit(SignalType.FALL.value))
            .otherwise(pl.lit(SignalType.NONE.value))
            .alias(self.out_col)
        )

    def _compute_meta(
        self,
        df: pl.DataFrame,
        prices: np.ndarray,
        up_off_series: pl.Series,
        dn_off_series: pl.Series,
        lf: int,
    ) -> pl.DataFrame:
        """Compute t_hit and ret meta columns."""
        n = df.height
        ts_arr = df.get_column(self.ts_col).to_numpy()

        idx = np.arange(n)
        up_np = up_off_series.fill_null(0).to_numpy()
        dn_np = dn_off_series.fill_null(0).to_numpy()

        hit_off = np.where(
            (up_np > 0) & ((dn_np == 0) | (up_np <= dn_np)),
            up_np,
            np.where(dn_np > 0, dn_np, 0),
        )

        hit_idx = np.clip(idx + hit_off, 0, n - 1)
        vert_idx = np.clip(idx + lf, 0, n - 1)
        final_idx = np.where(hit_off > 0, hit_idx, vert_idx)

        t_hit = ts_arr[final_idx]
        ret = np.where(prices > 0, np.log(prices[final_idx] / prices), np.nan)

        return df.with_columns(
            [
                pl.Series("t_hit", t_hit),
                pl.Series("ret", ret),
            ]
        )
