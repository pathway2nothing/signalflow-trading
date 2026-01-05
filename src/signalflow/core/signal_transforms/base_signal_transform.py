from typing import Protocol
import polars as pl
from signalflow.core.enums import SfComponentType


class SignalsTransform(Protocol):
    """Protocol for signal transformations."""

    name: str
    component_type: SfComponentType = SfComponentType.SIGNALS_TRANSFORM

    def __call__(self, value: pl.DataFrame) -> pl.DataFrame:
        """Apply transformation to signals dataframe.

        Args:
            value: Input signals dataframe.

        Returns:
            Transformed signals dataframe.
        """
        ...