"""ReversionBarrier target - revert to the SMA anchor before losing sl_pct."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.target.base import LABEL_COL, Target, register_target


@register_target("reversion_barrier")
@dataclass
class ReversionBarrier(Target):
    """From a below-anchor entry: 1 if price touches the rolling SMA anchor first, 0 if the stop.

    Only bars whose close sits below their own anchor are labeled (a revert setup
    exists); all other rows get a null label. Target-first on intrabar ambiguity.
    """

    anchor_bars: int = 240
    sl_pct: float = 0.02
    max_bars: int = 240

    @property
    def horizon(self) -> int:
        return self.max_bars

    def labels(self, data: Dataset, at: pl.DataFrame | None = None) -> pl.DataFrame:
        frame = data.frame.sort(["pair", "ts"]).with_columns(
            pl.col("close").rolling_mean(self.anchor_bars).over("pair").alias("_anchor")
        )
        frames = []
        for pair, sub in frame.group_by("pair", maintain_order=True):
            pair_name = pair[0] if isinstance(pair, tuple) else pair
            close = sub.get_column("close").to_numpy()
            high = sub.get_column("high").to_numpy()
            low = sub.get_column("low").to_numpy()
            anchor = sub.get_column("_anchor").to_numpy()
            lab = pl.Series(LABEL_COL, self._scan(close, high, low, anchor)).fill_nan(None).cast(pl.Int64)
            frames.append(pl.DataFrame({"pair": [pair_name] * len(close), "ts": sub.get_column("ts"), LABEL_COL: lab}))
        return self._restrict(pl.concat(frames), at)

    def _scan(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        n = len(close)
        out = np.full(n, np.nan)
        for i in range(n):
            a = anchor[i]
            if not np.isfinite(a) or close[i] >= a:
                continue
            stop = close[i] * (1.0 - self.sl_pct)
            end = min(n, i + self.max_bars + 1)
            label = 0.0
            for j in range(i + 1, end):
                if np.isfinite(anchor[j]) and high[j] >= anchor[j]:
                    label = 1.0
                    break
                if low[j] <= stop:
                    label = 0.0
                    break
            out[i] = label
        return out
