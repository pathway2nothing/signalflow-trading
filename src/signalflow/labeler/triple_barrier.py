from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar
import polars as pl
import numpy as np

from signalflow.core import SfComponentType, SignalType, Signals
from signalflow.labeler.base_labeler import Labeler


@dataclass
class TripleBarrierLabeler(Labeler):
    """
    Classic Triple-Barrier (first-touch) labeling (López de Prado style).

    Core idea:
      - For each event at t0 (default: every row; optionally only rows selected by `signals`)
      - Set vertical barrier t1 = t0 + lookforward_window
      - Set PT/SL barriers using volatility of RETURNS:
          pt_price = close[t0] * exp(+ profit_multiplier * vol[t0])
          sl_price = close[t0] * exp(- stop_loss_multiplier * vol[t0])
      - Label = which barrier was touched FIRST within (t0, t1]:
          RISE if PT touched first
          FALL if SL touched first
          NONE if none touched by t1

    Output (by default): only `label`
    If include_meta=True: also returns `t_hit` (timestamp of first touch or t1) and `ret` (log-return to hit)
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.DETECTOR
    name: str = "classic_triple_barrier"

    price_col: str = "close"

    vol_window: int = 60  
    lookforward_window: int = 1440  # 
    profit_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0

    include_meta: bool = False
    out_col: str = "label"

    def __post_init__(self) -> None:
        if self.vol_window <= 1:
            raise ValueError("vol_window must be > 1")
        if self.lookforward_window <= 0:
            raise ValueError("lookforward_window must be > 0")
        if self.profit_multiplier <= 0 or self.stop_loss_multiplier <= 0:
            raise ValueError("profit_multiplier/stop_loss_multiplier must be > 0")

        cols = [self.out_col]
        if self.include_meta:
            cols += ["t_hit", "ret"]
        self.output_columns = cols


    def extract(
        self,
        df: pl.DataFrame,
        signals: Signals | None = None,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        if not isinstance(df, pl.DataFrame):
            raise TypeError("ClassicTripleBarrierLabeler supports only Polars DataFrame")

        self._validate_input_pl(df)
        if self.price_col not in df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        df0 = df.sort([self.pair_col, self.ts_col])

        # Build event mask per (pair,timestamp) if signals provided:
        # labels computed on full series, but written only for event rows.
        event_keys: pl.DataFrame | None = None
        if signals is not None:
            s = signals.value  # expected pl.DataFrame
            required = {self.pair_col, self.ts_col, "signal_type"}
            missing = required - set(s.columns)
            if missing:
                raise ValueError(f"Signals missing columns: {sorted(missing)}")

            # Treat any row in signals as an event (or customize here if you need a specific signal_type)
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

        # If we have an event set: keep labels only on those keys, set others to NONE and meta to null.
        if event_keys is not None:
            out = out.join(
                event_keys.with_columns(pl.lit(True).alias("_is_event")),
                on=[self.pair_col, self.ts_col],
                how="left",
            ).with_columns(
                [
                    pl.when(pl.col("_is_event") == True)
                    .then(pl.col(self.out_col))
                    .otherwise(pl.lit(SignalType.NONE.value))
                    .alias(self.out_col),
                ]
                + (
                    [
                        pl.when(pl.col("_is_event") == True).then(pl.col("t_hit")).otherwise(pl.lit(None)).alias("t_hit"),
                        pl.when(pl.col("_is_event") == True).then(pl.col("ret")).otherwise(pl.lit(None)).alias("ret"),
                    ]
                    if self.include_meta
                    else []
                )
            ).drop("_is_event")

        if self.keep_input_columns:
            return out

        if self.output_columns is None:
            label_cols = sorted(set(out.columns) - input_cols)
        else:
            label_cols = list(self.output_columns)

        keep_cols = [self.pair_col, self.ts_col] + label_cols
        missing = [c for c in keep_cols if c not in out.columns]
        if missing:
            raise ValueError(f"Projection error, missing columns: {missing}")

        return out.select(keep_cols)

    def compute_pl_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
        close = group_df.get_column(self.price_col).to_numpy()
        ts = group_df.get_column(self.ts_col).to_numpy()
        n = close.shape[0]

        r = np.full(n, np.nan, dtype=float)
        valid = (close[1:] > 0) & (close[:-1] > 0)
        r[1:][valid] = np.log(close[1:][valid] / close[:-1][valid])

        vol = np.full(n, np.nan, dtype=float)
        w = int(self.vol_window)
        if n >= w:
            for i in range(w - 1, n):
                window = r[i - w + 1 : i + 1]
                if np.isfinite(window).sum() == w:
                    vol[i] = np.std(window, ddof=1)

        lf = int(self.lookforward_window)

        labels = np.full(n, SignalType.NONE.value, dtype=object)

        t_hit = np.full(n, None, dtype=object)  
        ret = np.full(n, np.nan, dtype=float)  

        for i in range(n):
            v = vol[i]
            c0 = close[i]

            if not np.isfinite(v) or not np.isfinite(c0) or c0 <= 0:
                continue

            j_end = i + lf
            if j_end >= n:
                j_end = n - 1

            if j_end <= i:
                continue

            pt = c0 * np.exp(self.profit_multiplier * v)
            sl = c0 * np.exp(-self.stop_loss_multiplier * v)

            window = close[i + 1 : j_end + 1]
            if window.size == 0:
                continue

            hit_up = np.where(window >= pt)[0]
            hit_dn = np.where(window <= sl)[0]

            t_up = int(hit_up[0] + 1) if hit_up.size > 0 else None
            t_dn = int(hit_dn[0] + 1) if hit_dn.size > 0 else None

            if t_up is None and t_dn is None:
                if self.include_meta:
                    t_hit[i] = ts[j_end]
                    if close[j_end] > 0:
                        ret[i] = float(np.log(close[j_end] / c0))
                continue

            if t_dn is None or (t_up is not None and t_up <= t_dn):
                j = i + t_up
                labels[i] = SignalType.RISE.value
            else:
                j = i + t_dn
                labels[i] = SignalType.FALL.value

            if self.include_meta:
                t_hit[i] = ts[j]
                if close[j] > 0:
                    ret[i] = float(np.log(close[j] / c0))

        out = group_df.with_columns(pl.Series(self.out_col, labels))

        if not self.include_meta:
            return out

        return out.with_columns(
            [
                pl.Series("t_hit", t_hit),
                pl.Series("ret", ret),
            ]
        )
