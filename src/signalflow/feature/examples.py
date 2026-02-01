from dataclasses import dataclass, field
import polars as pl
from signalflow.feature.base import Feature
from signalflow.feature.global_feature import GlobalFeature
from signalflow.core import sf_component
from typing import Any



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


@dataclass
@sf_component(name="example/global_rsi")
class ExampleGlobalMeanRsiFeature(GlobalFeature):
    """Mean RSI across all pairs per timestamp.
    
    1. Compute RSI per pair
    2. Mean across all pairs at time t → global_mean_rsi
    3. Optionally: rsi_diff = pair_rsi - global_mean_rsi
    
    Args:
        period: RSI period. Default: 14.
        add_diff: Add per-pair difference column. Default: False.
    """
    
    period: int = 14
    price_col: str = "close"
    add_diff: bool = False
    
    requires = ["{price_col}"]
    outputs = ["global_mean_rsi_{period}"]
    
    def output_cols(self, prefix: str = "") -> list[str]:
        cols = [f"{prefix}global_mean_rsi_{self.period}"]
        if self.add_diff:
            cols.append(f"{prefix}rsi_{self.period}_diff")
        return cols
    
    def compute(self, df: pl.DataFrame,  context: dict[str, Any] | None = None) -> pl.DataFrame:
        rsi_col = f"rsi_{self.period}"
        out_col = f"global_mean_rsi_{self.period}"
        
        has_rsi = rsi_col in df.columns
        if not has_rsi:
            rsi = ExampleRsiFeature(period=self.period, price_col=self.price_col)
            df = rsi.compute(df)
        
        mean_df = df.group_by(self.ts_col).agg(
            pl.col(rsi_col).mean().alias(out_col)
        )
        
        df = df.join(mean_df, on=self.ts_col, how="left")
        
        if self.add_diff:
            df = df.with_columns(
                (pl.col(rsi_col) - pl.col(out_col)).alias(f"rsi_{self.period}_diff")
            )
        
        if not has_rsi:
            df = df.drop(rsi_col)
        
        return df
        
