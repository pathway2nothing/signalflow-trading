"""TripleBarrier target (Lopez de Prado) - tp/sl/timeout race over a horizon (numba kernel, python fallback)."""


from dataclasses import dataclass

import numpy as np
import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.target._numba import njit, prange
from signalflow.target.base import LABEL_COL, Target, register_target


@njit(parallel=True, cache=True)
def _first_barrier(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, tp: float, sl: float, max_bars: int
) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        entry = close[i]
        up = entry * (1.0 + tp)
        down = entry * (1.0 - sl)
        end = min(n, i + max_bars + 1)
        label = 0
        for j in range(i + 1, end):
            if high[j] >= up:
                label = 1
                break
            if low[j] <= down:
                label = 0
                break
        out[i] = label
    return out


@register_target("triple_barrier")
@dataclass
class TripleBarrier(Target):
    """Meta-labeling target: success = tp hit before sl within the horizon."""

    tp: float = 0.03
    sl: float = 0.015
    max_bars: int = 100

    @property
    def horizon(self) -> int:
        return self.max_bars

    def labels(self, data: Dataset, at: pl.DataFrame | None = None) -> pl.DataFrame:
        frames = []
        for pair, sub in data.frame.sort(["pair", "ts"]).group_by("pair", maintain_order=True):
            pair_name = pair[0] if isinstance(pair, tuple) else pair
            close = sub.get_column("close").to_numpy().astype(np.float64)
            high = sub.get_column("high").to_numpy().astype(np.float64)
            low = sub.get_column("low").to_numpy().astype(np.float64)
            labels = _first_barrier(close, high, low, float(self.tp), float(self.sl), int(self.max_bars))
            frames.append(
                pl.DataFrame(
                    {"pair": [pair_name] * len(close), "ts": sub.get_column("ts"), LABEL_COL: labels}
                )
            )
        out = pl.concat(frames)
        return self._restrict(out, at)
