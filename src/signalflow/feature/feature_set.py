from __future__ import annotations

from dataclasses import dataclass
import polars as pl
import pandas as pd
from signalflow.feature.extractor.base import FeatureExtractor
from signalflow.core import RawDataView


@dataclass
class FeatureSet:
    """Collection of independent extractors/pipelines for parallel execution.
    
    FeatureSet orchestrates multiple feature extractors and pipelines,
    automatically handling:
    - Data fetching based on each extractor's raw_data_type
    - Backend conversion (Polars/Pandas) per extractor
    - Parallel execution of independent extractors
    - Result combination into single Polars DataFrame
    
    Unlike FeaturePipeline (sequential with shared context), FeatureSet
    extractors are independent and can be parallelized.
    
    Example:
        >>> feature_set = FeatureSet(
        ...     extractors=[
        ...         # Independent extractors (can parallelize)
        ...         RsiExtractor(period=14),
        ...         PandasTaBbandsExtractor(length=20),
        ...         
        ...         # Pipelines (sequential internally, but independent from each other)
        ...         FeaturePipeline(
        ...             name="momentum",
        ...             extractors=[RsiExtractor(period=14), RsiZScoreExtractor()]
        ...         ),
        ...         FeaturePipeline(
        ...             name="volume",
        ...             extractors=[VolumeExtractor(), VolumeBreakoutExtractor()]
        ...         ),
        ...     ]
        ... )
        >>> 
        >>> # Extract all features
        >>> dataset = feature_set.extract(raw_data, context)
        >>> # Result: Polars DataFrame with all features
    
    Attributes:
        extractors: List of extractors/pipelines (all independent)
        parallel: Whether to execute extractors in parallel (future feature)
    """
    extractors: list[FeatureExtractor]
    parallel: bool = False  
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.extractors:
            raise ValueError("At least one extractor must be provided")
    
    def extract(
        self, 
        raw_data: RawDataView, 
        context: dict
    ) -> pl.DataFrame:
        """Extract all features and combine into single Polars DataFrame.
        
        Args:
            raw_data: Raw data view providing access to market data
            context: Execution context with symbols, timeframe, etc.
            
        Returns:
            Polars DataFrame with all features and index columns (pair, timestamp)
        """
        feature_dfs = []
        
        for extractor in self.extractors:
            input_df = raw_data.get_data(
                extractor.raw_data_type, 
                extractor.df_type
            )
            
            result_df = extractor.extract(input_df, context)
            
            if isinstance(result_df, pd.DataFrame):
                result_df = pl.from_pandas(result_df)
            
            feature_dfs.append(result_df)
        
        combined = self._combine_features(feature_dfs)
        
        return combined
    
    def _combine_features(self, feature_dfs: list[pl.DataFrame]) -> pl.DataFrame:
        """Combine feature DataFrames into single DataFrame.
        
        Joins all features on (pair, timestamp) index columns.
        
        Args:
            feature_dfs: List of feature DataFrames
            
        Returns:
            Combined DataFrame with all features
        """
        if not feature_dfs:
            raise ValueError("No feature DataFrames to combine")
        
        combined = feature_dfs[0]
        
        if 'pair' not in combined.columns or 'timestamp' not in combined.columns:
            raise ValueError(
                "First feature DataFrame missing index columns (pair, timestamp). "
                "Set include_index_columns=True on first extractor."
            )
        
        for df in feature_dfs[1:]:
            has_index = 'pair' in df.columns and 'timestamp' in df.columns
            
            if has_index:
                combined = combined.join(
                    df, 
                    on=['pair', 'timestamp'], 
                    how='left'
                )
            else:
                feature_cols = [col for col in df.columns 
                               if col not in ['pair', 'timestamp']]
                combined = pl.concat([combined, df.select(feature_cols)], how='horizontal')
        
        return combined