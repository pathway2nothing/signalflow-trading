from dataclasses import dataclass, field
import polars as pl
from signalflow.feature.base import Feature
from signalflow.core import sf_component
from typing import Any

@dataclass
class GlobalFeature(Feature):
    """Base class for features computed across all pairs.
    
    Override compute() with custom aggregation logic.
    """
    
    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Must override - compute global feature across all pairs."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement compute()")
