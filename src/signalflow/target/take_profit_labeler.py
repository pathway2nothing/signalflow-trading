from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numba import njit, prange

from signalflow.core import SignalType, sf_component
from signalflow.target.base import Labeler


@njit(parallel=True, cache=True)
def _find_first_hit_static(
    prices: np.ndarray,
    pt: np.ndarray,
    sl: np.ndarray,
    lookforward: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the first hit for static barriers.

    Returns:
        up_off: offset of the first PT hit (0 = no hit)
        dn_off: offset of the first SL hit (0 = no hit)
    """
    n = len(prices)
    up_off = np.zeros(n, dtype=np.int32)
    dn_off = np.zeros(n, dtype=np.int32)

    for i in prange(n):
        pt_i = pt[i]
        sl_i = sl[i]

        max_j = min(i + lookforward, n - 1)

        for k in range(1, max_j - i + 1):
            p = prices[i + k]

            if up_off[i] == 0 and p >= pt_i:
                up_off[i] = k

            if dn_off[i] == 0 and p <= sl_i:
                dn_off[i] = k

            if up_off[i] > 0 and dn_off[i] > 0:
                break

    return up_off, dn_off


@dataclass
@sf_component(name="take_profit")
class TakeProfitLabeler(Labeler):
    """First-touch labeling with symmetric fixed-percentage barriers.

    Barriers:
      - TP = close[t0] * (1 + barrier_pct)
      - SL = close[t0] * (1 - barrier_pct)
      - Vertical barrier at t0 + horizon

    Label by first touch:
      - RISE if TP touched first (ties -> TP)
      - FALL if SL touched first
      - NONE if neither touched within horizon
    """

    price_col: str = "close"

    horizon: int = 1440
    barrier_pct: float = 0.01

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.barrier_pct <= 0:
            raise ValueError("barrier_pct must be > 0")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        if group_df.height == 0:
            return group_df

        lf = int(self.horizon)
        n = group_df.height

        prices = group_df.get_column(self.price_col).to_numpy().astype(np.float64)
        pt = prices * (1.0 + self.barrier_pct)
        sl = prices * (1.0 - self.barrier_pct)

        up_off, dn_off = _find_first_hit_static(prices, pt, sl, lf)

        up_off_series = pl.Series("_up_off", up_off).replace(0, None).cast(pl.Int32)
        dn_off_series = pl.Series("_dn_off", dn_off).replace(0, None).cast(pl.Int32)

        df = group_df.with_columns([up_off_series, dn_off_series])

        choose_up = pl.col("_up_off").is_not_null() & (
            pl.col("_dn_off").is_null() | (pl.col("_up_off") <= pl.col("_dn_off"))
        )
        choose_dn = pl.col("_dn_off").is_not_null() & (
            pl.col("_up_off").is_null() | (pl.col("_dn_off") < pl.col("_up_off"))
        )

        df = df.with_columns(
            pl.when(choose_up)
            .then(pl.lit(SignalType.RISE.value))
            .when(choose_dn)
            .then(pl.lit(SignalType.FALL.value))
            .otherwise(pl.lit(SignalType.NONE.value))
            .alias(self.out_col)
        )

        if self.include_meta:
            ts_arr = group_df.get_column(self.ts_col).to_numpy()

            up_np = up_off_series.fill_null(0).to_numpy()
            dn_np = dn_off_series.fill_null(0).to_numpy()
            idx = np.arange(n)

            hit_off = np.where(
                (up_np > 0) & ((dn_np == 0) | (up_np <= dn_np)),
                up_np,
                np.where(dn_np > 0, dn_np, 0),
            )

            hit_idx = np.clip(idx + hit_off, 0, n - 1)
            vert_idx = np.clip(idx + lf, 0, n - 1)
            final_idx = np.where(hit_off > 0, hit_idx, vert_idx)

            t_hit = ts_arr[final_idx]
            ret = np.log(prices[final_idx] / prices)

            df = df.with_columns(
                [
                    pl.Series("t_hit", t_hit),
                    pl.Series("ret", ret),
                ]
            )

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        df = df.drop(["_up_off", "_dn_off"])

        return df
