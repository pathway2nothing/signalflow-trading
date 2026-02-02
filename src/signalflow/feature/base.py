from dataclasses import dataclass, field
from typing import ClassVar, Any
import polars as pl
from signalflow.core import SfComponentType, sf_component
from signalflow.utils import KwargsTolerantMixin


@dataclass
class Feature(KwargsTolerantMixin):
    """Base class for all features.
    
    Two methods to implement:
        - compute(df): all pairs, abstract for GlobalFeature/Pipeline
        - compute_pair(df): one pair, for regular features
    
    Attributes:
        requires: Input column templates, e.g. ["{price_col}"]
        outputs: Output column templates, e.g. ["rsi_{period}"]
    """
    component_type: ClassVar[SfComponentType] = SfComponentType.FEATURE
    requires: ClassVar[list[str]] = []
    outputs: ClassVar[list[str]] = []
    
    group_col: str = "pair"
    ts_col: str = "timestamp"
    
    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Compute feature for all pairs"""
        return df.group_by(self.group_col, maintain_order=True).map_groups(self.compute_pair)
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute feature for single pair. Override for per-pair features."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement compute_pair()")
    
    def output_cols(self, prefix: str = "") -> list[str]:
        """Actual output column names with parameter substitution."""
        return [f"{prefix}{tpl.format(**self.__dict__)}" for tpl in self.outputs]
    
    def required_cols(self) -> list[str]:
        """Actual required column names with parameter substitution."""
        return [
            tpl.format(**self.__dict__) if "{" in tpl else tpl 
            for tpl in self.requires
        ]

    test_params: ClassVar[list[dict]] = []

@dataclass
class GlobalFeature(Feature):
    """Base class for features computed across all pairs.
    
    Override compute() with custom aggregation logic.
    """
    
    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Must override - compute global feature across all pairs."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement compute()")
