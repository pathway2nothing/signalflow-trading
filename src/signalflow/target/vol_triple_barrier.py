"""VolTripleBarrier target - tp/sl/timeout race with volatility-scaled barriers."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.target.base import LABEL_COL, Target, register_target


@register_target("vol_triple_barrier")
@dataclass
class VolTripleBarrier(Target):
    """TripleBarrier with barriers as multiples of the trailing EWMA return volatility.

    Binary by default (1 = tp first, 0 = sl first or timeout); ``three_class``
    emits 1 = tp, 0 = sl, 2 = timeout. Null sigma rows get a null label.
    """

    tp_mult: float = 2.0
    sl_mult: float = 1.0
    vol_window: int = 100
    max_bars: int = 120
    three_class: bool = False

    @property
    def horizon(self) -> int:
        return self.max_bars

    def labels(self, data: Dataset, at: pl.DataFrame | None = None) -> pl.DataFrame:
        frame = (
            data.frame.sort(["pair", "ts"])
            .with_columns((pl.col("close") / pl.col("close").shift(1).over("pair") - 1.0).alias("_r1"))
            .with_columns(pl.col("_r1").ewm_std(span=self.vol_window).over("pair").alias("_sigma"))
        )
        frames = []
        for pair, sub in frame.group_by("pair", maintain_order=True):
            pair_name = pair[0] if isinstance(pair, tuple) else pair
            close = sub.get_column("close").to_numpy()
            high = sub.get_column("high").to_numpy()
            low = sub.get_column("low").to_numpy()
            sigma = sub.get_column("_sigma").to_numpy()
            lab = pl.Series(LABEL_COL, self._scan(close, high, low, sigma)).fill_nan(None).cast(pl.Int64)
            frames.append(pl.DataFrame({"pair": [pair_name] * len(close), "ts": sub.get_column("ts"), LABEL_COL: lab}))
        return self._restrict(pl.concat(frames), at)

    def _scan(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        n = len(close)
        out = np.full(n, np.nan)
        for i in range(n):
            s = sigma[i]
            if not np.isfinite(s) or s <= 0.0:
                continue
            entry = close[i]
            up = entry * (1.0 + self.tp_mult * s)
            down = entry * (1.0 - self.sl_mult * s)
            end = min(n, i + self.max_bars + 1)
            label = 2.0 if self.three_class else 0.0
            for j in range(i + 1, end):
                if high[j] >= up:
                    label = 1.0
                    break
                if low[j] <= down:
                    label = 0.0
                    break
            out[i] = label
        return out
