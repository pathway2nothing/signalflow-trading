"""Multi-horizon ensemble of mean-reversion events.

Runs :class:`MeanReversionEventLabeler` over several forward horizons and
averages the resulting per-class soft probabilities; the hard label is
the argmax of the averaged triple.

This is the production version of iter-33's
``soft_D3_multi_horizon`` ensemble — slightly outperformed the
single-horizon variant (soft MI 0.179 vs 0.178 on the validated pool)
and showed marginally better train→test stability across labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow.core import labeler
from signalflow.core.enums import SignalCategory
from signalflow.target.base import Labeler
from signalflow.target.path_labeler import MeanReversionEventLabeler


@dataclass
@labeler("multi_horizon_mean_reversion")
class MultiHorizonMeanReversionLabeler(Labeler):
    """Average ``MeanReversionEventLabeler`` posteriors across multiple horizons.

    Algorithm:
        For each ``h`` in :attr:`horizons` instantiate a
        :class:`MeanReversionEventLabeler` with that horizon (other
        parameters shared from this labeler), call its
        ``compute_group_soft`` per pair, then average the resulting
        ``[p_mean_reverted, p_trend_continuation, p_no_reversion]``
        triple across horizons. Rows where every horizon is invalid stay
        null; otherwise rows are normalised to sum to 1.

    Hard label is the argmax over the averaged triple.

    Research provenance:
        iter-33 (sf-profit) ``soft_D3_multi_horizon`` — best soft MI
        0.173 on ``signed_range_60`` in the validated pool, edging out
        ``soft_D3_revert`` (0.172).

    Attributes:
        horizons: Forward windows to ensemble over.
            Default: ``(60, 120, 240, 480)``.
        z_window: Rolling window for µ/σ baseline. Default: 240.
        stretch_threshold: |z| above which a bar is "overstretched".
            Default: 2.0.
        revert_threshold: |z_fwd| below which the move is "reverted".
            Default: 0.5.
    """

    signal_category: SignalCategory = SignalCategory.PRICE_STRUCTURE

    soft_classes: ClassVar[tuple[str, ...]] = (
        "mean_reverted",
        "trend_continuation",
        "no_reversion",
    )

    price_col: str = "close"
    horizons: tuple[int, ...] = (60, 120, 240, 480)
    z_window: int = 240
    stretch_threshold: float = 2.0
    revert_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("horizons must be non-empty")
        if any(h <= 0 for h in self.horizons):
            raise ValueError("all horizons must be > 0")
        if self.z_window <= 1:
            raise ValueError("z_window must be > 1")
        if self.stretch_threshold <= 0 or self.revert_threshold <= 0:
            raise ValueError("thresholds must be > 0")
        if self.revert_threshold >= self.stretch_threshold:
            raise ValueError("revert_threshold must be < stretch_threshold")
        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def _make_member(self, horizon: int) -> MeanReversionEventLabeler:
        return MeanReversionEventLabeler(
            price_col=self.price_col,
            horizon=horizon,
            z_window=self.z_window,
            stretch_threshold=self.stretch_threshold,
            revert_threshold=self.revert_threshold,
            mask_to_signals=False,
        )

    def _ensemble_soft(self, group_df: pl.DataFrame) -> np.ndarray:
        """Run each horizon's compute_group_soft, average → ``(n, 3)``."""
        n = group_df.height
        accum = np.zeros((n, 3), dtype=np.float64)
        valid_count = np.zeros(n, dtype=np.float64)
        for h in self.horizons:
            member = self._make_member(h)
            out = member.compute_group_soft(group_df)
            ps = np.column_stack(
                [
                    out.get_column("p_mean_reverted").to_numpy(),
                    out.get_column("p_trend_continuation").to_numpy(),
                    out.get_column("p_no_reversion").to_numpy(),
                ]
            )
            row_valid = np.isfinite(ps).all(axis=1)
            accum[row_valid] += ps[row_valid]
            valid_count[row_valid] += 1
        avg = np.full((n, 3), np.nan)
        nonzero = valid_count > 0
        avg[nonzero] = accum[nonzero] / valid_count[nonzero, None]
        # Renormalise rows (clipping can leave tiny drift)
        rsum = avg.sum(axis=1, keepdims=True)
        avg = np.where(rsum > 0, avg / np.where(rsum > 0, rsum, 1.0), np.nan)
        return avg

    def compute_group(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")
        if group_df.height == 0:
            return group_df

        avg = self._ensemble_soft(group_df)
        valid = np.isfinite(avg[:, 0])
        labels = np.full(group_df.height, None, dtype=object)
        argmax_idx = np.where(valid, avg.argmax(axis=1), -1)
        name_map = np.array(["mean_reverted", "trend_continuation", "no_reversion"])
        labels[valid] = name_map[argmax_idx[valid]]
        df = group_df.with_columns(pl.Series(self.out_col, labels.tolist(), dtype=pl.Utf8))
        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)
        return df

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")
        if group_df.height == 0:
            return group_df
        avg = self._ensemble_soft(group_df)
        return group_df.with_columns(
            pl.Series(f"{self.soft_col_prefix}mean_reverted", avg[:, 0], dtype=pl.Float64),
            pl.Series(f"{self.soft_col_prefix}trend_continuation", avg[:, 1], dtype=pl.Float64),
            pl.Series(f"{self.soft_col_prefix}no_reversion", avg[:, 2], dtype=pl.Float64),
        )
