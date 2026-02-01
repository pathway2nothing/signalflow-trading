from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Any
from signalflow.core import RawDataView, RawDataType
import polars as pl
from signalflow.feature.base import Feature
from signalflow.feature.global_feature import GlobalFeature
from signalflow.feature.offset_feature import OffsetFeature


@dataclass
class FeaturePipeline(Feature):
    """Orchestrates multiple features with optimized execution.
    
    Groups consecutive per-pair features into batches for single group_by.
    
    Args:
        features: List of features to compute.
        raw_data_type: Type of raw data (defines available columns).
    
    Example:
        >>> pipeline = FeaturePipeline(
        ...     features=[
        ...         RsiFeature(period=14),
        ...         SmaFeature(period=20),
        ...         GlobalFeature(base=RsiFeature(period=14), reference_pair="BTCUSDT"),
        ...     ],
        ...     raw_data_type=RawDataType.SPOT,
        ... )
        >>> df = pipeline.run(raw_data_view)
    """
    
    features: list[Feature] = field(default_factory=list)
    raw_data_type: RawDataType = RawDataType.SPOT
    
    requires: ClassVar[list[str]] = []
    
    def __post_init__(self):
        if not self.features:
            raise ValueError("FeaturePipeline requires at least one feature")
        self._validate()
    
    @property
    def outputs(self) -> list[str]:
        """Aggregated outputs from all features."""
        result = []
        for f in self.features:
            result.extend(f.output_cols())
        return result
    
    def output_cols(self, prefix: str = "") -> list[str]:
        return [f"{prefix}{col}" for col in self.outputs]
    
    def _validate(self):
        """Validate all dependencies are satisfied."""
        available = self.raw_data_type.columns.copy()
        
        for f in self.features:
            required = set(f.required_cols())
            missing = required - available
            
            if missing:
                raise ValueError(
                    f"{f.__class__.__name__} requires {missing}, "
                    f"available: {sorted(available)}"
                )
            
            available.update(f.output_cols())
    
    def _group_into_batches(self) -> list[list[Feature]]:
        """Group features: consecutive per-pair → batch, global → separate."""
        batches = []
        current_batch = []
        
        for f in self.features:
            is_global = isinstance(f, (GlobalFeature, FeaturePipeline))
            
            if is_global:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                batches.append([f])
            else:
                current_batch.append(f)
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _is_per_pair_batch(self, batch: list[Feature]) -> bool:
        """Check if batch contains only per-pair features."""
        return not any(
            isinstance(f, (GlobalFeature, FeaturePipeline)) 
            for f in batch
        )
    
    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Compute all features with optimized batching."""
        df = df.sort([self.group_col, self.ts_col])
        
        batches = self._group_into_batches()
        
        for batch in batches:
            if self._is_per_pair_batch(batch):
                def apply_batch(pair_df: pl.DataFrame, features=batch) -> pl.DataFrame:
                    for f in features:
                        pair_df = f.compute_pair(pair_df)
                    return pair_df
                
                df = df.group_by(self.group_col, maintain_order=True).map_groups(apply_batch)
            else:
                for f in batch:
                    df = f.compute(df, context=context)
        
        return df
    
    def run(self, raw_data_view: RawDataView, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Entry point: load from RawDataView and compute."""
        df = raw_data_view.to_polars(self.raw_data_type)
        return self.compute(df)
    
    def to_mermaid(self) -> str:
        """Generate Mermaid diagram of feature dependencies."""
        lines = ["graph LR"]
        lines.append("    subgraph Input")
        for col in sorted(self.raw_data_type.columns):
            lines.append(f"        {col}[{col}]")
        lines.append("    end")
        
        for f in self.features:
            name = f.__class__.__name__
            if hasattr(f, 'period'):
                name = f"{name}_{f.period}"
            
            for req in f.required_cols():
                lines.append(f"    {req} --> {name}")
            for out in f.output_cols():
                lines.append(f"    {name} --> {out}[{out}]")
        
        return "\n".join(lines)

