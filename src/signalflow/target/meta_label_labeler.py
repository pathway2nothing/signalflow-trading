"""
Meta-labeling (López de Prado, *Advances in Financial Machine Learning*, ch. 3).

Two-stage workflow:

1. A *primary* model (or rule) emits a directional signal at certain bars
       - long (``side = +1``) or short (``side = -1``).
    2. A *meta* model learns a binary "take this trade or skip it" filter
       conditional on the primary side, trained against this labeler's
       output: ``1`` when the side-adjusted PnL within the horizon exceeded a
       minimum threshold, ``0`` otherwise.

Compared with a raw triple-barrier label, meta-labeling lets the secondary
model focus on *precision* (avoid bad trades) without having to also predict
direction - historically reduces false positives and improves Sharpe even
when the primary recall stays the same.

Two configurable resolution modes:

* ``mode="fixed_horizon"`` - PnL at ``t + horizon`` (signed by side) vs
      ``min_return``. Cheap and matches the original AFML formulation.
    * ``mode="triple_barrier"`` - first-touch of a profit barrier at
      ``min_return`` or a stop barrier at ``-max_loss``; ``take`` if PT is hit
      first, otherwise ``skip``. Captures the path-dependence the iter-32
      experiments showed was missing in TP-binary labels.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
import polars as pl

from signalflow.enums import SignalCategory
from signalflow.target._numba import njit, prange
from signalflow.target._soft_helpers import sigmoid_expr
from signalflow.target.base import register_target
from signalflow.target.labeler import Labeler, Signals


@njit(parallel=True, cache=True)
def _meta_triple_barrier(
    log_close: np.ndarray,
    side: np.ndarray,
    min_return: float,
    max_loss: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward triple-barrier meta-label (take/skip) and signed realised pnl per bar."""
    n = log_close.shape[0]
    out_label = np.full(n, -1, dtype=np.int8)
    out_pnl = np.full(n, np.nan, dtype=np.float64)
    for i in prange(n):
        s = side[i]
        if s == 0:
            continue
        max_j = i + horizon
        if max_j >= n:
            max_j = n - 1
        c0 = log_close[i]
        decided = 0
        for k in range(1, max_j - i + 1):
            r = (log_close[i + k] - c0) * s
            if r >= min_return:
                out_label[i] = 1
                out_pnl[i] = r
                decided = 1
                break
            if r <= -max_loss:
                out_label[i] = 0
                out_pnl[i] = r
                decided = 1
                break
        if decided == 0:
            r_end = (log_close[max_j] - c0) * s
            out_label[i] = 0
            out_pnl[i] = r_end
    return out_label, out_pnl


@dataclass
@register_target("meta_label")
class MetaLabelLabeler(Labeler):
    """
    Binary meta-label conditional on a primary directional signal.

    Inputs:
        * ``signal_keys`` (in ``data_context``) - required; ``(pair, timestamp)``
          rows marking primary-signal bars. Bars outside this set get null
          labels (mask_to_signals semantics).
        * Direction is taken from the optional ``side_col`` column in
          ``signal_keys``: values ``+1`` / ``-1`` for long / short, or the
          string aliases ``"long"`` / ``"short"``. If ``side_col`` is absent
          we fall back to :attr:`default_side`.

    Output:
        * Hard label: ``"take"`` (1) when the side-adjusted forward path
          meets the success criterion, ``"skip"`` (0) otherwise. Null at
          non-signal bars.
        * Meta column ``signed_pnl`` - realised side-adjusted log return at
          the decision bar (helpful for downstream cost-sensitive models).
        * Soft output ``p_take`` / ``p_skip`` - sigmoid over signed PnL minus
          ``min_return``.
    """

    signal_category: SignalCategory = SignalCategory.PRICE_DIRECTION

    soft_classes: ClassVar[tuple[str, ...]] = ("skip", "take")
    softness_k: float = 80.0

    price_col: str = "close"
    horizon: int = 240
    mode: Literal["fixed_horizon", "triple_barrier"] = "triple_barrier"
    min_return: float = 0.01
    max_loss: float = 0.01
    default_side: int = 1
    side_col: str = "side"

    meta_columns: tuple[str, ...] = ("signed_pnl",)

    mask_to_signals: bool = True

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.mode not in ("fixed_horizon", "triple_barrier"):
            raise ValueError(f"mode must be 'fixed_horizon' or 'triple_barrier', got {self.mode!r}")
        if self.min_return <= 0:
            raise ValueError("min_return must be > 0")
        if self.mode == "triple_barrier" and self.max_loss <= 0:
            raise ValueError("max_loss must be > 0 for triple_barrier mode")
        if self.default_side not in (-1, 1):
            raise ValueError("default_side must be -1 or +1")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def labels(self, data, at=None) -> pl.DataFrame:
        """Target entry point for meta-labeling."""
        signal_keys = data.frame.select([self.pair_col, self.ts_col])
        computed = self.compute(data.frame, data_context={"signal_keys": signal_keys})
        from signalflow.target.base import LABEL_COL

        numeric = self._label_to_numeric(computed.get_column(self.out_col), self.positive_classes)
        distinct = sorted(numeric.drop_nulls().unique().to_list())
        if len(distinct) < 2:
            from signalflow.errors import DegenerateTargetError

            raise DegenerateTargetError(
                f"{type(self).__name__}.labels coerced to a degenerate target: non-null labels "
                f"collapse to {distinct} distinct value(s). Multi-class labelers need an explicit "
                f"positive-class mapping via the `positive_classes` class attribute."
            )
        result = (
            computed.select([self.pair_col, self.ts_col])
            .with_columns(numeric.alias(LABEL_COL))
            .rename({self.pair_col: "pair", self.ts_col: "ts"})
            .select(["pair", "ts", LABEL_COL])
        )
        return self._restrict(result, at)

    def compute(
        self,
        df: pl.DataFrame,
        signals: Signals | None = None,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Validate that ``signal_keys`` was provided before group-by dispatch."""
        if data_context is None or "signal_keys" not in data_context:
            raise ValueError(
                "MetaLabelLabeler requires data_context['signal_keys'] - "
                "meta-labels are conditional on a primary signal set"
            )
        return super().compute(df, signals=signals, data_context=data_context)

    def compute_soft(
        self,
        df: pl.DataFrame,
        signals: Signals | None = None,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        if data_context is None or "signal_keys" not in data_context:
            raise ValueError("MetaLabelLabeler.compute_soft requires data_context['signal_keys']")
        return super().compute_soft(df, signals=signals, data_context=data_context)

    def _side_array_for_group(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None,
    ) -> np.ndarray:
        """Build a per-bar side array (+1 / -1 / 0) from ``signal_keys``."""
        n = group_df.height
        side = np.zeros(n, dtype=np.int8)
        if data_context is None or "signal_keys" not in data_context:
            return side

        signal_keys: pl.DataFrame = data_context["signal_keys"]
        pair_value = group_df.get_column(self.pair_col)[0]
        sk = signal_keys.filter(pl.col(self.pair_col) == pair_value)
        if sk.height == 0:
            return side

        select_cols: list[str] = [self.ts_col]
        if self.side_col in sk.columns:
            select_cols.append(self.side_col)
        sk = sk.select(select_cols).unique(subset=[self.ts_col])

        joined = group_df.select(self.ts_col).with_row_index("_row").join(sk, on=self.ts_col, how="inner")
        if joined.height == 0:
            return side

        rows = joined.get_column("_row").to_numpy().astype(np.int64)
        if self.side_col in joined.columns:
            raw_side = joined.get_column(self.side_col).to_list()
            mapped: list[int] = []
            for v in raw_side:
                if v is None:
                    mapped.append(self.default_side)
                elif isinstance(v, str):
                    sv = v.lower()
                    if sv in ("long", "rise", "buy", "+1", "1"):
                        mapped.append(1)
                    elif sv in ("short", "fall", "sell", "-1"):
                        mapped.append(-1)
                    else:
                        mapped.append(self.default_side)
                else:
                    iv = int(v)
                    mapped.append(1 if iv > 0 else (-1 if iv < 0 else self.default_side))
            side_vals = np.array(mapped, dtype=np.int8)
        else:
            side_vals = np.full(rows.shape[0], self.default_side, dtype=np.int8)

        side[rows] = side_vals
        return side

    def _fixed_horizon_outcomes(
        self,
        log_close: np.ndarray,
        side: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = log_close.shape[0]
        h = int(self.horizon)
        signed_pnl = np.full(n, np.nan, dtype=np.float64)
        label = np.full(n, -1, dtype=np.int8)
        end = min(n, n)
        for i in range(n):
            s = side[i]
            if s == 0:
                continue
            j = i + h
            if j >= end:
                continue
            r = (log_close[j] - log_close[i]) * s
            signed_pnl[i] = r
            label[i] = 1 if r >= self.min_return else 0
        return label, signed_pnl

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")
        if data_context is None or "signal_keys" not in data_context:
            raise ValueError(
                "MetaLabelLabeler requires data_context['signal_keys'] - "
                "meta-labels are conditional on a primary signal set"
            )

        prices = group_df.get_column(self.price_col).to_numpy().astype(np.float64)
        with np.errstate(divide="ignore"):
            log_close = np.log(np.where(prices > 0, prices, np.nan))

        side = self._side_array_for_group(group_df, data_context)

        if self.mode == "triple_barrier":
            label_arr, pnl_arr = _meta_triple_barrier(
                log_close,
                side.astype(np.int8),
                float(self.min_return),
                float(self.max_loss),
                int(self.horizon),
            )
        else:
            label_arr, pnl_arr = self._fixed_horizon_outcomes(log_close, side)

        n = group_df.height
        labels: list[str | None] = [None] * n
        for i in range(n):
            v = label_arr[i]
            if v == 1:
                labels[i] = "take"
            elif v == 0:
                labels[i] = "skip"

        df = group_df.with_columns(pl.Series(self.out_col, labels, dtype=pl.Utf8))
        if self.include_meta:
            df = df.with_columns(pl.Series("signed_pnl", pnl_arr, dtype=pl.Float64).fill_nan(None))

        return df

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Soft ``(p_skip, p_take)`` from the signed PnL margin."""
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")
        if data_context is None or "signal_keys" not in data_context:
            raise ValueError("MetaLabelLabeler.compute_soft requires data_context['signal_keys']")

        prices = group_df.get_column(self.price_col).to_numpy().astype(np.float64)
        with np.errstate(divide="ignore"):
            log_close = np.log(np.where(prices > 0, prices, np.nan))
        side = self._side_array_for_group(group_df, data_context)
        if self.mode == "triple_barrier":
            _, pnl_arr = _meta_triple_barrier(
                log_close,
                side.astype(np.int8),
                float(self.min_return),
                float(self.max_loss),
                int(self.horizon),
            )
        else:
            _, pnl_arr = self._fixed_horizon_outcomes(log_close, side)

        df = group_df.with_columns(pl.Series("_signed_pnl", pnl_arr, dtype=pl.Float64).fill_nan(None))
        p_take = sigmoid_expr(pl.col("_signed_pnl") - pl.lit(self.min_return), self.softness_k)
        null_mask = pl.col("_signed_pnl").is_null()

        df = df.with_columns(
            pl.when(null_mask)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.lit(1.0) - p_take)
            .alias(f"{self.soft_col_prefix}skip"),
            pl.when(null_mask)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(p_take)
            .alias(f"{self.soft_col_prefix}take"),
        )
        df = df.drop("_signed_pnl")
        return df
