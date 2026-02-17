from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from signalflow.core.signal_transform import SignalsTransform

_NONE_SIGNAL = "none"  # Legacy inactive signal type (deprecated; prefer null)


@dataclass(frozen=True)
class Signals:
    """Immutable container for trading signals.

    Canonical in-memory format is a Polars DataFrame with long schema.

    Required columns:
        - pair (str): Trading pair identifier
        - timestamp (datetime): Signal timestamp
        - signal_type (SignalType | int): Signal type (RISE, FALL, NONE)
        - signal (int | float): Signal value

    Optional columns:
        - probability (float): Signal probability (required for merge logic)

    Attributes:
        value (pl.DataFrame): Polars DataFrame containing signal data.

    Example:
        ```python
        from signalflow.core import Signals, SignalType
        import polars as pl
        from datetime import datetime

        # Create signals
        signals_df = pl.DataFrame({
            "pair": ["BTCUSDT", "ETHUSDT"],
            "timestamp": [datetime.now(), datetime.now()],
            "signal_type": [SignalType.RISE.value, SignalType.FALL.value],
            "signal": [1, -1],
            "probability": [0.8, 0.7]
        })

        signals = Signals(signals_df)

        # Apply transformation
        filtered = signals.apply(filter_transform)

        # Chain transformations
        processed = signals.pipe(
            transform1,
            transform2,
            transform3
        )

        # Merge signals
        combined = signals1 + signals2
        ```

    Note:
        All transformations return new Signals instance.
        No in-place mutation is allowed.
    """

    value: pl.DataFrame

    def apply(self, transform: SignalsTransform) -> Signals:
        """Apply a single transformation to signals.

        Args:
            transform (SignalsTransform): Callable transformation implementing
                SignalsTransform protocol.

        Returns:
            Signals: New Signals instance with transformed data.

        Example:
            ```python
            from signalflow.core import Signals
            import polars as pl

            def filter_high_probability(df: pl.DataFrame) -> pl.DataFrame:
                return df.filter(pl.col("probability") > 0.7)

            filtered = signals.apply(filter_high_probability)
            ```
        """
        out = transform(self.value)
        return Signals(out)

    def pipe(self, *transforms: SignalsTransform) -> Signals:
        """Apply multiple transformations sequentially.

        Args:
            *transforms (SignalsTransform): Sequence of transformations to apply in order.

        Returns:
            Signals: New Signals instance after applying all transformations.

        Example:
            ```python
            result = signals.pipe(
                filter_none_signals,
                normalize_probabilities,
                add_metadata
            )
            ```
        """
        s = self
        for t in transforms:
            s = s.apply(t)
        return s

    def __add__(self, other: Signals) -> Signals:
        """Merge two Signals objects.

        Merge rules:
            1. Key: (pair, timestamp) or (pair, timestamp, signal_category)
               if signal_category column is present.
            2. Signal type priority:
               - null signal_type has lowest priority
               - SignalType.NONE ("none") has lowest priority (backward compat)
               - Non-null/non-NONE always overrides
               - If both non-null, ``other`` wins
            3. Low-priority signals normalized to probability = 0
            4. Merge is deterministic

        Args:
            other (Signals): Another Signals object to merge.

        Returns:
            Signals: New merged Signals instance.

        Raises:
            TypeError: If other is not a Signals instance.

        Example:
            ```python
            # Detector 1 signals
            signals1 = detector1.run(data)

            # Detector 2 signals
            signals2 = detector2.run(data)

            # Merge with priority to signals2
            merged = signals1 + signals2

            # NONE/null signals overridden by non-NONE
            # Non-NONE conflicts resolved by taking signals2
            ```
        """
        if not isinstance(other, Signals):
            return NotImplemented

        a = self.value
        b = other.value

        all_cols = list(dict.fromkeys([*a.columns, *b.columns]))

        def align(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns([pl.lit(None).alias(c) for c in all_cols if c not in df.columns]).select(all_cols)

        a = align(a).with_columns(pl.lit(0).alias("_src"))
        b = align(b).with_columns(pl.lit(1).alias("_src"))

        merged = pl.concat([a, b], how="vertical")

        # Determine merge key: include signal_category if present
        has_category = "signal_category" in all_cols
        key_cols = ["pair", "timestamp"]
        if has_category:
            key_cols = ["pair", "timestamp", "signal_category"]

        # Priority: null and "none" signal_type are lowest priority
        merged = merged.with_columns(
            pl.when(pl.col("signal_type").is_null() | (pl.col("signal_type") == _NONE_SIGNAL))
            .then(pl.lit(0))
            .otherwise(pl.col("probability"))
            .alias("probability")
        )

        merged = merged.with_columns(
            pl.when(pl.col("signal_type").is_null() | (pl.col("signal_type") == _NONE_SIGNAL))
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("_priority")
        )

        sort_cols = [*key_cols, "_priority", "_src"]
        sort_desc = [False] * len(key_cols) + [True, True]

        merged = (
            merged.sort(sort_cols, descending=sort_desc)
            .unique(subset=key_cols, keep="first")
            .drop(["_priority", "_src"])
            .sort(key_cols[:2])  # always sort output by pair, timestamp
        )

        return Signals(merged)
