from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow.core import SfComponentType, SignalType, Signals
from signalflow.labeler.base_labeler import Labeler


@dataclass
class FixedHorizonLabeler(Labeler):
    """
    Fixed-Horizon Labeling (baseline):
      label[t0] = sign(close[t0 + horizon] - close[t0])

    Output (only):
      - label
      - (optional) ret  : log-return to horizon
      - (optional) t1   : timestamp at horizon

    Notes:
      - If there is not enough future data (t0+horizon out of range):
          label = NONE, ret/t1 = null
      - If signals is provided, labels are written only on event rows (like ClassicTripleBarrierLabeler),
        while the horizon is computed on full series.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.DETECTOR
    name: str = "fixed_horizon"

    price_col: str = "close"
    horizon: int = 60 

    out_col: str = "label"
    include_meta: bool = False 

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")

        cols = [self.out_col]
        if self.include_meta:
            cols += ["t1", "ret"]
        self.output_columns = cols

    def extract(
        self,
        df: pl.DataFrame,
        signals: Signals | None = None,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        if not isinstance(df, pl.DataFrame):
            raise TypeError("FixedHorizonLabeler supports only Polars DataFrame")

        self._validate_input_pl(df)
        if self.price_col not in df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        df0 = df.sort([self.pair_col, self.ts_col])

        event_keys: pl.DataFrame | None = None
        if signals is not None:
            s = signals.value
            required = {self.pair_col, self.ts_col, "signal_type"}
            missing = required - set(s.columns)
            if missing:
                raise ValueError(f"Signals missing columns: {sorted(missing)}")
            event_keys = s.select([self.pair_col, self.ts_col]).unique(subset=[self.pair_col, self.ts_col])

        input_cols = set(df0.columns)

        def _wrapped(g: pl.DataFrame) -> pl.DataFrame:
            out = self.compute_pl_group(g, data_context=None)
            if out.height != g.height:
                raise ValueError("compute_pl_group must preserve group length")
            return out

        out = (
            df0.group_by(self.pair_col, maintain_order=True)
            .map_groups(_wrapped)
            .sort([self.pair_col, self.ts_col])
        )

        if event_keys is not None:
            out = (
                out.join(
                    event_keys.with_columns(pl.lit(True).alias("_is_event")),
                    on=[self.pair_col, self.ts_col],
                    how="left",
                )
                .with_columns(
                    [
                        pl.when(pl.col("_is_event") == True)
                        .then(pl.col(self.out_col))
                        .otherwise(pl.lit(SignalType.NONE.value))
                        .alias(self.out_col),
                    ]
                    + (
                        [
                            pl.when(pl.col("_is_event") == True).then(pl.col("t1")).otherwise(pl.lit(None)).alias("t1"),
                            pl.when(pl.col("_is_event") == True).then(pl.col("ret")).otherwise(pl.lit(None)).alias("ret"),
                        ]
                        if self.include_meta
                        else []
                    )
                )
                .drop("_is_event")
            )

        if self.keep_input_columns:
            return out

        label_cols = list(self.output_columns) if self.output_columns is not None else sorted(set(out.columns) - input_cols)
        keep_cols = [self.pair_col, self.ts_col] + label_cols
        missing = [c for c in keep_cols if c not in out.columns]
        if missing:
            raise ValueError(f"Projection error, missing columns: {missing}")

        return out.select(keep_cols)

    def compute_pl_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
        close = group_df.get_column(self.price_col).to_numpy()
        ts = group_df.get_column(self.ts_col).to_numpy()
        n = close.shape[0]
        h = int(self.horizon)

        labels = np.full(n, SignalType.NONE.value, dtype=object)
        t1 = np.full(n, None, dtype=object)
        ret = np.full(n, np.nan, dtype=float)

        for i in range(n):
            j = i + h
            if j >= n:
                continue

            c0 = close[i]
            c1 = close[j]
            if not np.isfinite(c0) or not np.isfinite(c1) or c0 <= 0 or c1 <= 0:
                continue

            if c1 > c0:
                labels[i] = SignalType.RISE.value
            elif c1 < c0:
                labels[i] = SignalType.FALL.value
            else:
                labels[i] = SignalType.NONE.value

            if self.include_meta:
                t1[i] = ts[j]
                ret[i] = float(np.log(c1 / c0))

        out = group_df.with_columns(pl.Series(self.out_col, labels))
        if not self.include_meta:
            return out

        return out.with_columns(
            [
                pl.Series("t1", t1),
                pl.Series("ret", ret),
            ]
        )
