from dataclasses import dataclass, field
import polars as pl
from signalflow.feature.base import Feature, ExampleRsiFeature
from signalflow.core import sf_component


@dataclass
class GlobalFeature(Feature):
    """Base class for features computed across all pairs.
    
    Override compute() with custom aggregation logic.
    """
    
    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Must override - compute global feature across all pairs."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement compute()")


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
    
    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
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
        
