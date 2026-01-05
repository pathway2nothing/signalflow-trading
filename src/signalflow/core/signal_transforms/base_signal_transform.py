from typing import Protocol
import polars as pl
from signalflow.core.enums import SfComponentType


class SignalsTransform(Protocol):
    """Protocol for signal transformations.

    Defines the interface for functions that transform signal DataFrames.
    Transformations can filter, modify, or augment signals while preserving
    the core schema (pair, timestamp, signal_type, signal, probability).

    Protocol-based design allows:
        - Any callable with matching signature
        - Functional composition via Signals.pipe()
        - Type checking without inheritance

    Common use cases:
        - Filter signals by probability threshold
        - Normalize probability values
        - Add metadata columns
        - Remove duplicate signals
        - Apply time-based filters (cooldown periods)

    Attributes:
        name (str): Descriptive name for the transformation.
        component_type (SfComponentType): Always SIGNALS_TRANSFORM for registry.

    Example:
        ```python
        import polars as pl
        from signalflow.core import Signals, SignalsTransform

        # Simple function-based transform
        def filter_high_probability(df: pl.DataFrame) -> pl.DataFrame:
            '''Keep only signals with probability > 0.7'''
            return df.filter(pl.col("probability") > 0.7)

        filter_high_probability.name = "filter_high_prob"
        filter_high_probability.component_type = SfComponentType.SIGNALS_TRANSFORM

        # Class-based transform
        class NormalizeProbability:
            name = "normalize_prob"
            component_type = SfComponentType.SIGNALS_TRANSFORM

            def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
                '''Normalize probabilities to [0, 1] range'''
                return df.with_columns(
                    (pl.col("probability") / pl.col("probability").max())
                    .alias("probability")
                )

        # Use in signal pipeline
        signals = detector.detect(data)
        
        # Single transform
        filtered = signals.apply(filter_high_probability)

        # Chained transforms
        processed = signals.pipe(
            filter_high_probability,
            NormalizeProbability(),
            add_metadata_transform
        )
        ```

    Example:
        ```python
        # Register transform in registry
        from signalflow.core import sf_component

        @sf_component(name="cooldown_filter")
        class CooldownFilter:
            component_type = SfComponentType.SIGNALS_TRANSFORM
            name = "cooldown_filter"

            def __init__(self, cooldown_minutes: int = 60):
                self.cooldown_minutes = cooldown_minutes

            def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
                '''Filter signals within cooldown period'''
                return (
                    df.sort(["pair", "timestamp"])
                    .with_columns(
                        pl.col("timestamp")
                        .diff()
                        .over("pair")
                        .dt.total_minutes()
                        .alias("minutes_since_last")
                    )
                    .filter(
                        (pl.col("minutes_since_last").is_null()) |
                        (pl.col("minutes_since_last") >= self.cooldown_minutes)
                    )
                    .drop("minutes_since_last")
                )

        # Use registered transform
        from signalflow.core.registry import default_registry

        cooldown = default_registry.create(
            SfComponentType.SIGNALS_TRANSFORM,
            "cooldown_filter",
            cooldown_minutes=120
        )
        
        filtered_signals = signals.apply(cooldown)
        ```

    Note:
        Transformations should be pure functions (no side effects).
        Input DataFrame schema should be preserved where possible.
        Return DataFrame with same or compatible schema for chaining.

    See Also:
        Signals: Container class with apply() and pipe() methods.
        sf_component: Decorator for registering transforms.
    """

    name: str
    component_type: SfComponentType = SfComponentType.SIGNALS_TRANSFORM

    def __call__(self, value: pl.DataFrame) -> pl.DataFrame:
        """Apply transformation to signals DataFrame.

        Core method that performs the actual transformation logic.
        Must accept and return Polars DataFrame with signals schema.

        Expected input schema:
            - pair (str): Trading pair
            - timestamp (datetime): Signal timestamp
            - signal_type (int): SignalType enum value
            - signal (int|float): Signal value
            - probability (float): Signal probability (optional but common)

        Args:
            value (pl.DataFrame): Input signals DataFrame with standard schema.

        Returns:
            pl.DataFrame: Transformed signals DataFrame. Should maintain
                compatible schema for chaining with other transforms.

        Example:
            ```python
            # Function-based transform
            def remove_none_signals(df: pl.DataFrame) -> pl.DataFrame:
                from signalflow.core.enums import SignalType
                return df.filter(
                    pl.col("signal_type") != SignalType.NONE.value
                )

            # Apply transform
            filtered = signals.apply(remove_none_signals)

            # Class-based transform with state
            class ThresholdFilter:
                def __init__(self, threshold: float = 0.5):
                    self.threshold = threshold

                def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
                    return df.filter(
                        pl.col("probability") >= self.threshold
                    )

            # Use with different thresholds
            filter_50 = ThresholdFilter(threshold=0.5)
            filter_70 = ThresholdFilter(threshold=0.7)

            signals_50 = signals.apply(filter_50)
            signals_70 = signals.apply(filter_70)

            # Combine multiple transforms
            processed = signals.pipe(
                remove_none_signals,
                filter_50,
                lambda df: df.sort(["pair", "timestamp"])
            )
            ```

        Note:
            Should be deterministic - same input produces same output.
            Avoid modifying input DataFrame (return new DataFrame).
            Consider performance for large datasets (vectorized operations).
        """
        ...