from __future__ import annotations
from dataclasses import dataclass
import polars as pl
from signalflow.core.signal_transforms import SignalsTransform


NONE_TYPE = "NONE" 

@dataclass(frozen=True)
class Signals:
    value: pl.DataFrame
    none_type: str = NONE_TYPE

    def apply(self, t: SignalsTransform) -> "Signals":
        out = t(self.value)
        return Signals(out, none_type=self.none_type)

    def pipe(self, *transforms: SignalsTransform) -> "Signals":
        s = self
        for t in transforms:
            s = s.apply(t)
        return s

    def __add__(self, other: "Signals") -> "Signals":
        if not isinstance(other, Signals):
            return NotImplemented

        a = self.value
        b = other.value

        all_cols = list(dict.fromkeys([*a.columns, *b.columns]))

        def align(df: pl.DataFrame) -> pl.DataFrame:
            return (
                df.with_columns(
                    [pl.lit(None).alias(c) for c in all_cols if c not in df.columns]
                )
                .select(all_cols)
            )

        a = align(a).with_columns(pl.lit(0).alias("_src"))
        b = align(b).with_columns(pl.lit(1).alias("_src"))

        merged = pl.concat([a, b], how="vertical")

        merged = merged.with_columns(
            pl.when(pl.col("signal_type") == self.none_type)
            .then(pl.lit(0))
            .otherwise(pl.col("probability"))
            .alias("probability")
        )

        merged = merged.with_columns(
            pl.when(pl.col("signal_type") == self.none_type)
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("_priority")
        )

        merged = (
            merged
            .sort(
                ["pair", "timestamp", "_priority", "_src"],
                descending=[False, False, True, True],
            )
            .unique(
                subset=["pair", "timestamp"],
                keep="first",
            )
            .drop(["_priority", "_src"])
            .sort(["pair", "timestamp"])
        )

        return Signals(merged, none_type=self.none_type)
