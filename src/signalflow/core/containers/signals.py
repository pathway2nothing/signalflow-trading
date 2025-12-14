from __future__ import annotations

from dataclasses import dataclass
import polars as pl
from signalflow.core.signal_transforms import SignalsTransform
from signalflow.core.signal_types import SignalType


@dataclass(frozen=True)
class Signals:
    """Immutable container for trading signals.

    Canonical in-memory format is a Polars DataFrame (long schema).

    Required columns:
        - pair: str
        - timestamp: datetime
        - signal_type: SignalType | int
        - signal: int | float
        - probability: float (optional, but required for merge logic)

    Notes:
        - All transformations return a new Signals instance.
        - No in-place mutation is allowed.
    """

    value: pl.DataFrame

    def apply(self, transform: SignalsTransform) -> "Signals":
        """Apply a single transformation to signals.

        Args:
            transform: A callable transformation implementing SignalsTransform.

        Returns:
            New Signals instance with transformed data.
        """
        out = transform(self.value)
        return Signals(out)

    def pipe(self, *transforms: SignalsTransform) -> "Signals":
        """Apply multiple transformations sequentially.

        Args:
            *transforms: Sequence of transformations.

        Returns:
            New Signals instance after applying all transformations.
        """
        s = self
        for t in transforms:
            s = s.apply(t)
        return s


    def __add__(self, other: "Signals") -> "Signals":
        """Merge two Signals objects.

        Merge rules:
            1. Key: (pair, timestamp)
            2. If signal_type differs:
                - SignalType.NONE has the lowest priority.
                - Non-NONE always overrides NONE.
                - If both are non-NONE, `other` wins.
            3. SignalType.NONE is always normalized to:
                - probability = 0
            4. Merge is deterministic.

        Args:
            other: Another Signals object.

        Returns:
            New merged Signals instance.
        """
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
            pl.when(pl.col("signal_type") == SignalType.NONE.value)
            .then(pl.lit(0))
            .otherwise(pl.col("probability"))
            .alias("probability")
        )

        merged = merged.with_columns(
            pl.when(pl.col("signal_type") == SignalType.NONE.value)
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

        return Signals(merged)
