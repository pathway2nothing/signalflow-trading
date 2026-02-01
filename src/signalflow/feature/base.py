from dataclasses import dataclass, field
from typing import ClassVar
import polars as pl
from signalflow.core import SfComponentType, sf_component



@dataclass
class Feature:
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
        """Compute feature for all pairs. Must override."""
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


@dataclass
@sf_component(name="example/rsi")
class ExampleRsiFeature(Feature):
    """Relative Strength Index.
    
    Args:
        period: RSI period. Default: 14.
        price_col: Price column to use. Default: "close".
    
    Example:
        >>> rsi = RsiFeature(period=21)
        >>> rsi.output_cols()  # ["rsi_21"]
    """
    
    period: int = 14
    price_col: str = "close"
    
    requires = ["{price_col}"]
    outputs = ["rsi_{period}"]
    
    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute RSI for all pairs."""
        return df.group_by(self.group_col, maintain_order=True).map_groups(self.compute_pair)
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute RSI for single pair."""
        col_name = f"rsi_{self.period}"
        
        delta = pl.col(self.price_col).diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0)
        
        avg_gain = gain.rolling_mean(window_size=self.period)
        avg_loss = loss.rolling_mean(window_size=self.period)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return df.with_columns(rsi.alias(col_name))



@dataclass
@sf_component(name="example/sma")
class ExampleSmaFeature(Feature):
    """Simple Moving Average."""
    
    period: int = 20
    price_col: str = "close"
    
    requires = ["{price_col}"]
    outputs = ["sma_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        sma = pl.col(self.price_col).rolling_mean(window_size=self.period)
        return df.with_columns(sma.alias(f"sma_{self.period}"))