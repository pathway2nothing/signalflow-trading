from typing import Protocol
import polars as pl


class SignalsTransform(Protocol):
    """Protocol for signal transformations."""

    name: str

    def __call__(self, value: pl.DataFrame) -> pl.DataFrame:
        """Apply transformation to signals dataframe.

        Args:
            value: Input signals dataframe.

        Returns:
            Transformed signals dataframe.
        """
        ...