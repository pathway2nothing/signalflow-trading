from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Any
import pandas as pd
from signalflow.core import DataFrameType, SfComponentType
from signalflow.feature.extractor.base import FeatureExtractor


@dataclass(frozen=True)
class PandasTaExtractor(FeatureExtractor):
    """Adapter for pandas-ta indicators.
    
    Wraps pandas-ta library indicators to work with SignalFlow's FeatureExtractor API.
    Works exclusively with Pandas DataFrames.
    
    The extractor calls pandas-ta indicators using their standard functional API:
    `ta.indicator_name(df["column"], **kwargs)` and adapts the result to match
    SignalFlow's conventions.
    
    Note: pandas-ta returns column names in UPPERCASE format (e.g., RSI_14).
    This extractor converts them to lowercase with underscores (e.g., rsi_14).
    
    Example:
        >>> # RSI indicator
        >>> extractor = PandasTaExtractor(
        ...     indicator="rsi",
        ...     params={"length": 14},
        ...     input_column="close"
        ... )
        >>> features = extractor.extract(ohlcv_df, context)
        >>> # Output: rsi_14
        
        >>> # Bollinger Bands (multi-column output)
        >>> extractor = PandasTaExtractor(
        ...     indicator="bbands",
        ...     params={"length": 20, "std": 2},
        ...     input_column="close"
        ... )
        >>> features = extractor.extract(ohlcv_df, context)
        >>> # Output: bbl_20_2_0, bbm_20_2_0, bbu_20_2_0, bbb_20_2_0, bbp_20_2_0
        
        >>> # ATR (requires multiple input columns)
        >>> extractor = PandasTaExtractor(
        ...     indicator="atr",
        ...     params={"length": 14},
        ...     input_column="high",
        ...     additional_inputs={"low": "low", "close": "close"}
        ... )
        >>> features = extractor.extract(ohlcv_df, context)
        >>> # Output: atr_14
    
    Attributes:
        indicator: Name of pandas-ta indicator (e.g., "rsi", "bbands", "macd")
        params: Dict of indicator-specific parameters
        input_column: Primary input column name (default: "close")
        additional_inputs: Dict mapping pandas-ta param names to df column names
                          (e.g., {"low": "low", "close": "close"} for ATR)
        rename_outputs: Optional dict to rename output columns
                       (e.g., {"rsi_14": "rsi"})
    """
    component_type: ClassVar[SfComponentType] = SfComponentType.FEATURE_EXTRACTOR
    
    indicator: str
    params: dict[str, Any] = field(default_factory=dict)
    input_column: str = "close"
    additional_inputs: dict[str, str] = field(default_factory=dict)
    rename_outputs: dict[str, str] = field(default_factory=dict)
    
    df_type: DataFrameType = DataFrameType.PANDAS
    phased_computation: bool = False
    include_index_columns: bool = True

    def __post_init__(self):
        """Validate configuration and check pandas-ta availability."""
        try:
            import pandas_ta as ta  # noqa: F401
        except ImportError:
            raise ImportError(
                "pandas-ta is required for PandasTaExtractor. "
                "Install with: pip install pandas-ta"
            )
        
        if not self.indicator:
            raise ValueError("indicator name must be specified")

    def extract(
        self, 
        df: pd.DataFrame, 
        context: dict
    ) -> pd.DataFrame:
        """Extract features using pandas-ta indicator.
        
        Args:
            df: OHLCV data with required columns
            context: Dict with 'symbols', 'timeframe'
            
        Returns:
            DataFrame with indicator features, preserving input row alignment
            
        Raises:
            ValueError: If required columns missing
            AttributeError: If indicator not found in pandas-ta
        """
        self.validate_input(df)
        
        required = ['pair', 'timestamp', self.input_column]
        required.extend(self.additional_inputs.values())
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        result_frames = []
        
        for pair, group in df.groupby('pair'):
            indicator_result = self._calculate_indicator(group)
            
            if isinstance(indicator_result, pd.Series):
                indicator_result = indicator_result.to_frame()
            
            indicator_result.insert(0, 'pair', pair)
            indicator_result.insert(1, 'timestamp', group['timestamp'].values)
            
            result_frames.append(indicator_result)
        
        result = pd.concat(result_frames, ignore_index=True)
        
        result = self._normalize_column_names(result)
        
        if self.rename_outputs:
            result = result.rename(columns=self.rename_outputs)
        
        if not self.include_index_columns:
            result = result.drop(columns=['pair', 'timestamp'])
        
        return result
    
    def _calculate_indicator(self, group: pd.DataFrame) -> pd.DataFrame | pd.Series:
        """Calculate pandas-ta indicator for a single pair.
        
        Args:
            group: DataFrame for single trading pair
            
        Returns:
            Indicator result as Series or DataFrame
        """
        import pandas_ta as ta
        
        try:
            indicator_func = getattr(ta, self.indicator)
        except AttributeError:
            raise AttributeError(
                f"Indicator '{self.indicator}' not found in pandas-ta. "
                f"Available indicators: {ta.indicators()}"
            )
        
        kwargs = self.params.copy()
        
        primary_input = group[self.input_column]
        
        for param_name, column_name in self.additional_inputs.items():
            kwargs[param_name] = group[column_name]
        
        result = indicator_func(primary_input, **kwargs)
        
        return result
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert pandas-ta UPPERCASE column names to lowercase.
        
        pandas-ta returns columns like: RSI_14, MACD_12_26_9, BBL_20_2.0
        We normalize to: rsi_14, macd_12_26_9, bbl_20_2_0
        
        Args:
            df: DataFrame with pandas-ta column names
            
        Returns:
            DataFrame with normalized column names
        """
        rename_map = {}
        
        for col in df.columns:
            if col in ['pair', 'timestamp']:
                continue
            
            normalized = col.lower().replace('.', '_')
            rename_map[col] = normalized
        
        return df.rename(columns=rename_map)


@dataclass(frozen=True)
class PandasTaRsiExtractor(PandasTaExtractor):
    """Convenience wrapper for pandas-ta RSI indicator.
    
    Pre-configured PandasTaExtractor for RSI calculation.
    
    Example:
        >>> extractor = PandasTaRsiExtractor(length=14)
        >>> features = extractor.extract(ohlcv_df, context)
        >>> # Output: rsi_14
    
    Attributes:
        length: RSI period (default: 14)
    """
    length: int = 14
    
    def __post_init__(self):
        """Initialize with RSI-specific configuration."""
        object.__setattr__(self, 'indicator', 'rsi')
        object.__setattr__(self, 'params', {'length': self.length})
        object.__setattr__(self, 'input_column', 'close')
        super().__post_init__()


@dataclass(frozen=True)
class PandasTaBbandsExtractor(PandasTaExtractor):
    """Convenience wrapper for pandas-ta Bollinger Bands indicator.
    
    Pre-configured PandasTaExtractor for Bollinger Bands calculation.
    Returns 5 columns: lower band, middle band, upper band, bandwidth, %B.
    
    Example:
        >>> extractor = PandasTaBbandsExtractor(length=20, std=2.0)
        >>> features = extractor.extract(ohlcv_df, context)
        >>> # Output: bbl_20_2_0, bbm_20_2_0, bbu_20_2_0, bbb_20_2_0, bbp_20_2_0
    
    Attributes:
        length: Moving average period (default: 20)
        std: Number of standard deviations (default: 2.0)
    """
    length: int = 20
    std: float = 2.0
    
    def __post_init__(self):
        """Initialize with Bollinger Bands-specific configuration."""
        object.__setattr__(self, 'indicator', 'bbands')
        object.__setattr__(self, 'params', {'length': self.length, 'std': self.std})
        object.__setattr__(self, 'input_column', 'close')
        super().__post_init__()


@dataclass(frozen=True)
class PandasTaMacdExtractor(PandasTaExtractor):
    """Convenience wrapper for pandas-ta MACD indicator.
    
    Pre-configured PandasTaExtractor for MACD calculation.
    Returns 3 columns: MACD line, histogram, signal line.
    
    Example:
        >>> extractor = PandasTaMacdExtractor(fast=12, slow=26, signal=9)
        >>> features = extractor.extract(ohlcv_df, context)
        >>> # Output: macd_12_26_9, macdh_12_26_9, macds_12_26_9
    
    Attributes:
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)
    """
    fast: int = 12
    slow: int = 26
    signal: int = 9
    
    def __post_init__(self):
        """Initialize with MACD-specific configuration."""
        object.__setattr__(self, 'indicator', 'macd')
        object.__setattr__(
            self, 
            'params', 
            {'fast': self.fast, 'slow': self.slow, 'signal': self.signal}
        )
        object.__setattr__(self, 'input_column', 'close')
        super().__post_init__()


@dataclass(frozen=True)
class PandasTaAtrExtractor(PandasTaExtractor):
    """Convenience wrapper for pandas-ta ATR indicator.
    
    Pre-configured PandasTaExtractor for Average True Range calculation.
    Requires high, low, and close columns.
    
    Example:
        >>> extractor = PandasTaAtrExtractor(length=14)
        >>> features = extractor.extract(ohlcv_df, context)
        >>> # Output: atr_14
    
    Attributes:
        length: ATR period (default: 14)
    """
    length: int = 14
    
    def __post_init__(self):
        """Initialize with ATR-specific configuration."""
        object.__setattr__(self, 'indicator', 'atr')
        object.__setattr__(self, 'params', {'length': self.length})
        object.__setattr__(self, 'input_column', 'high')
        object.__setattr__(
            self, 
            'additional_inputs', 
            {'low': 'low', 'close': 'close'}
        )
        super().__post_init__()