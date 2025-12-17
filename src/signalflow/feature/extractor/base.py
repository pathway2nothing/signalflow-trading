from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
import polars as pl
import pandas as pd
from signalflow.core import DataFrameType, SfComponentType


@dataclass(frozen=True)
class FeatureExtractor:
    """Base class for feature extraction from market data.

    Feature extractors transform raw OHLCV data into derived features
    (indicators, statistics, etc.) while preserving row alignment.

    Contract:
      - Pipeline provides DataFrame in requested backend (df_type)
      - extract() must preserve exact row count and order
      - If phased_computation=True, extract_phased() may be called instead
      - Index columns (pair, timestamp) are optionally included in output

    Context dict expected keys:
      - symbols: list[str] - trading pairs being processed
      - timeframe: str - timeframe (e.g., '1m', '5m')
      - phase: str | None - optional phase identifier (e.g., 'backtest', 'live')

    Example:
        >>> extractor = RSIExtractor(periods=[7, 14, 28])
        >>> features = extractor.extract(ohlcv_data, {'symbols': ['BTCUSDT'], 'timeframe': '1m'})
        
    Attributes:
        df_type: Required DataFrame backend (POLARS or PANDAS)
        phased_computation: Whether extractor supports optimized phased calculation
        include_index_columns: Include key columns (pair, timestamp) in output
    """
    component_type: ClassVar[SfComponentType] = SfComponentType.FEATURE_EXTRACTOR
    
    df_type: DataFrameType = DataFrameType.POLARS
    raw_data_type: str = "spot"
    phased_computation: bool = False
    include_index_columns: bool = True

    def validate_input(self, df: pl.DataFrame | pd.DataFrame) -> None:
        """Validate input DataFrame matches required backend.
        
        Args:
            df: Input DataFrame to validate
            
        Raises:
            TypeError: If DataFrame backend doesn't match df_type
        """
        if self.df_type == DataFrameType.POLARS and not isinstance(df, pl.DataFrame):
            raise TypeError(
                f"Expected polars.DataFrame for {self.__class__.__name__}, "
                f"got {type(df).__name__}"
            )
        if self.df_type == DataFrameType.PANDAS and not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame for {self.__class__.__name__}, "
                f"got {type(df).__name__}"
            )

    def extract(
        self, 
        df: pl.DataFrame | pd.DataFrame, 
        context: dict
    ) -> pl.DataFrame | pd.DataFrame:
        """Extract features from market data.
        
        Main entry point for feature calculation. Must preserve input row
        alignment - output must have same number of rows as input.
        
        Args:
            df: Market data (OHLCV) in required backend format
            context: Dict with 'symbols', 'timeframe', optionally 'phase'
            
        Returns:
            DataFrame with computed features, same row count as input
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement extract()"
        )

    def extract_phased(
        self, 
        df: pl.DataFrame | pd.DataFrame, 
        context: dict
    ) -> pl.DataFrame | pd.DataFrame:
        """Extract features with phase-aware optimization.
        
        Optional method for extractors that can optimize computation when
        processing data in phases (e.g., backtesting vs live trading).
        Only called when phased_computation=True and pipeline opts in.
        
        Args:
            df: Market data in required backend format
            context: Dict with 'symbols', 'timeframe', 'phase'
            
        Returns:
            DataFrame with computed features
            
        Raises:
            NotImplementedError: If phased_computation=True but not implemented
        """
        if self.phased_computation:
            raise NotImplementedError(
                f"{self.__class__.__name__} has phased_computation=True "
                "but doesn't implement extract_phased()"
            )
        return self.extract(df, context)
    
    def _attach_index_columns(
        self, 
        source_df: pl.DataFrame | pd.DataFrame,
        features_df: pl.DataFrame | pd.DataFrame,
        index_columns: list[str]
    ) -> pl.DataFrame | pd.DataFrame:
        """Prepend index columns to feature DataFrame.
        
        Args:
            source_df: Original input DataFrame with index columns
            features_df: Computed features without index
            index_columns: Names of columns to attach (e.g., ['pair', 'timestamp'])
            
        Returns:
            Features with index columns prepended
        """
        if not self.include_index_columns:
            return features_df
        
        if isinstance(source_df, pl.DataFrame):
            index_data = source_df.select(index_columns)
            return pl.concat([index_data, features_df], how="horizontal")
        else:
            index_data = source_df[index_columns]
            return pd.concat([index_data, features_df], axis=1)