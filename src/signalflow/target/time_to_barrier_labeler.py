"""
Survival-style triple-barrier labeler.

Where :class:`TripleBarrierLabeler` emits a categorical RISE/FALL/NONE label
per bar, this labeler exposes the *timing* of the first barrier touch as a
regression / survival target. For each bar we record:

* ``hit_time``        - normalised time to first barrier ∈ ``(0, 1]``,
                            with ``1.0`` reserved for the vertical-barrier
                            (no horizontal touch within horizon).
    * ``hit_event``       - categorical first-touch outcome (``"pt"``, ``"sl"``,
                            ``"vertical"``).
    * ``hit_ret``         - log return realised at the first-touch bar.
    * ``censored``        - boolean: ``True`` when the vertical barrier was
                            hit first (right-censored observation).

The continuous ``hit_time`` lets downstream models fit hazard / survival
formulations (e.g. parametric ``log(1 - hit_time)`` or DeepSurv-style heads),
addressing the iter-32 finding that fixed-horizon binary labels discard the
*speed* of mean reversion - a signal worth more than the direction alone for
sizing and stop placement.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow.enums import SignalCategory
from signalflow.target._numba import njit, prange
from signalflow.target._soft_helpers import sigmoid_expr
from signalflow.target.base import register_target
from signalflow.target.labeler import Labeler


@njit(parallel=True, cache=True)
def _first_barrier_touch(
    prices: np.ndarray,
    pt: np.ndarray,
    sl: np.ndarray,
    lookforward: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Offset of the first barrier hit and which barrier (0=none, 1=pt, 2=sl), profit-first on ties."""
    n = prices.shape[0]
    off = np.zeros(n, dtype=np.int32)
    which = np.zeros(n, dtype=np.int8)
    for i in prange(n):
        pt_i = pt[i]
        sl_i = sl[i]
        if np.isnan(pt_i) or np.isnan(sl_i):
            continue
        max_j = i + lookforward
        if max_j >= n:
            max_j = n - 1
        for k in range(1, max_j - i + 1):
            p = prices[i + k]
            if p >= pt_i:
                off[i] = k
                which[i] = 1
                break
            if p <= sl_i:
                off[i] = k
                which[i] = 2
                break
    return off, which


@dataclass
@register_target("time_to_barrier")
class TimeToBarrierLabeler(Labeler):
    """
    Time-to-first-touch triple barrier with survival-style outputs.

    Algorithm:
        1. Realised volatility ``vol = rolling_std(log_ret, vol_window)``.
        2. Volatility-scaled barriers:
            * ``pt = close * exp(+vol * profit_multiplier)``
            * ``sl = close * exp(-vol * stop_loss_multiplier)``
        3. Scan ``[t+1 .. t+horizon]`` for the first crossing. Ties favour
           the profit barrier (matches :class:`TripleBarrierLabeler`).
        4. Emit four columns:
            * ``hit_event`` - ``"pt" | "sl" | "vertical"`` (string label).
            * ``hit_time`` - ``offset / horizon`` ∈ ``(0, 1]``, ``1.0`` if
              vertical barrier (i.e., no horizontal touch).
            * ``hit_ret`` - ``log(prices[t + offset] / prices[t])`` at the
              first-touch bar.
            * ``censored`` - ``True`` for vertical barriers (right-censored
              observations for survival models).

    Soft outputs (``p_pt_fast, p_vertical, p_sl_fast``) are derived from the
    timing gap between profit and stop touches: trades that hit the profit
    barrier early load mass on ``p_pt_fast``; trades that hit the stop early
    load on ``p_sl_fast``; trades that never touch either load on
    ``p_vertical``. This is useful for ranking trades by *expected speed* of
    resolution, not just direction.

    ``horizon`` accepts a bar count (int, assuming 1-minute data for the default) or a
    duration string (``"1d"``) resolved against the dataset interval.
    """

    signal_category: SignalCategory = SignalCategory.PRICE_DIRECTION

    soft_classes: ClassVar[tuple[str, ...]] = ("sl_fast", "vertical", "pt_fast")
    duration_fields: ClassVar[tuple[str, ...]] = ("horizon",)

    price_col: str = "close"
    vol_window: int = 60
    horizon: int | str = 1440
    profit_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0

    meta_columns: tuple[str, ...] = ("hit_time", "hit_ret", "censored")
    softness_k: float = 6.0

    def __post_init__(self) -> None:
        if self.vol_window <= 1:
            raise ValueError("vol_window must be > 1")
        if isinstance(self.horizon, int) and self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.profit_multiplier <= 0 or self.stop_loss_multiplier <= 0:
            raise ValueError("profit_multiplier/stop_loss_multiplier must be > 0")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def _with_barriers(self, group_df: pl.DataFrame) -> pl.DataFrame:
        return group_df.with_columns(
            (pl.col(self.price_col) / pl.col(self.price_col).shift(1))
            .log()
            .rolling_std(window_size=self.vol_window, ddof=1)
            .alias("_vol")
        ).with_columns(
            (pl.col(self.price_col) * (pl.col("_vol") * self.profit_multiplier).exp()).alias("_pt"),
            (pl.col(self.price_col) * (-pl.col("_vol") * self.stop_loss_multiplier).exp()).alias("_sl"),
        )

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        df = self._with_barriers(group_df)
        prices = df.get_column(self.price_col).to_numpy().astype(np.float64)
        pt = df.get_column("_pt").fill_null(np.nan).to_numpy().astype(np.float64)
        sl = df.get_column("_sl").fill_null(np.nan).to_numpy().astype(np.float64)

        off, which = _first_barrier_touch(prices, pt, sl, int(self.horizon))
        n = prices.shape[0]
        h = float(self.horizon)

        finite = ~(np.isnan(pt) | np.isnan(sl))

        hit_off = np.where(which > 0, off, self.horizon).astype(np.float64)
        hit_time = hit_off / h

        idx = np.arange(n)
        end_idx = np.clip(idx + hit_off.astype(np.int64), 0, n - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            hit_ret = np.where(prices > 0, np.log(prices[end_idx] / prices), np.nan)

        censored = (which == 0) & finite

        labels: list[str | None] = [None] * n
        for i in range(n):
            if not finite[i]:
                continue
            if which[i] == 1:
                labels[i] = "pt"
            elif which[i] == 2:
                labels[i] = "sl"
            else:
                labels[i] = "vertical"

        df = df.with_columns(pl.Series(self.out_col, labels, dtype=pl.Utf8))
        if self.include_meta:
            ht_arr = np.where(finite, hit_time, np.nan)
            hr_arr = np.where(finite, hit_ret, np.nan)
            df = df.with_columns(
                pl.Series("hit_time", ht_arr, dtype=pl.Float64).fill_nan(None),
                pl.Series("hit_ret", hr_arr, dtype=pl.Float64).fill_nan(None),
                pl.Series("censored", censored, dtype=pl.Boolean),
            )

        df = df.drop(["_vol", "_pt", "_sl"])

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Soft triple ``(p_sl_fast, p_vertical, p_pt_fast)`` from timing gap."""
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        df = self._with_barriers(group_df)
        prices = df.get_column(self.price_col).to_numpy().astype(np.float64)
        pt = df.get_column("_pt").fill_null(np.nan).to_numpy().astype(np.float64)
        sl = df.get_column("_sl").fill_null(np.nan).to_numpy().astype(np.float64)

        n = prices.shape[0]
        h = int(self.horizon)
        pt_off = np.zeros(n, dtype=np.int32)
        sl_off = np.zeros(n, dtype=np.int32)

        for i in range(n):
            if np.isnan(pt[i]) or np.isnan(sl[i]):
                continue
            max_j = min(i + h, n - 1)
            for k in range(1, max_j - i + 1):
                p = prices[i + k]
                if pt_off[i] == 0 and p >= pt[i]:
                    pt_off[i] = k
                if sl_off[i] == 0 and p <= sl[i]:
                    sl_off[i] = k
                if pt_off[i] > 0 and sl_off[i] > 0:
                    break

        e_pt = np.where(pt_off > 0, pt_off, h).astype(np.float64)
        e_sl = np.where(sl_off > 0, sl_off, h).astype(np.float64)
        gap = (e_sl - e_pt) / float(h)
        finite = ~(np.isnan(pt) | np.isnan(sl))
        neither = (pt_off == 0) & (sl_off == 0)

        df = df.with_columns(
            pl.Series("_gap", gap, dtype=pl.Float64),
            pl.Series("_finite", finite, dtype=pl.Boolean),
            pl.Series("_neither", neither, dtype=pl.Boolean),
        )

        p_pt_raw = sigmoid_expr(pl.col("_gap"), self.softness_k)
        p_sl_raw = sigmoid_expr(-pl.col("_gap"), self.softness_k)
        p_pt_clamped = pl.when(pl.col("_neither")).then(pl.lit(0.0)).otherwise(p_pt_raw)
        p_sl_clamped = pl.when(pl.col("_neither")).then(pl.lit(0.0)).otherwise(p_sl_raw)
        p_vertical_raw = (pl.lit(1.0) - p_pt_clamped - p_sl_clamped).clip(lower_bound=0.0)
        total = p_pt_clamped + p_sl_clamped + p_vertical_raw
        safe = pl.when(total > 0).then(total).otherwise(pl.lit(1.0))
        null_mask = ~pl.col("_finite")

        df = df.with_columns(
            pl.when(null_mask)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(p_sl_clamped / safe)
            .alias(f"{self.soft_col_prefix}sl_fast"),
            pl.when(null_mask)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(p_vertical_raw / safe)
            .alias(f"{self.soft_col_prefix}vertical"),
            pl.when(null_mask)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(p_pt_clamped / safe)
            .alias(f"{self.soft_col_prefix}pt_fast"),
        )
        df = df.drop(["_vol", "_pt", "_sl", "_gap", "_finite", "_neither"])
        return df
