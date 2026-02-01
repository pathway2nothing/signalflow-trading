from dataclasses import dataclass, field
import polars as pl
from signalflow.feature.base import Feature
from typing import Any


@dataclass
class OffsetFeature(Feature):
    """Multi-timeframe feature via offset resampling.
    
    Creates `window` parallel time series with different offsets.
    Each offset computes features as if on `window`-minute bars.
    
    For GRU trained on 15m but needing 1m decisions:
    - Creates 15 parallel series (offset 0-14)
    - At signal time t, use offset = t % 15 to get correct features
    
    Args:
        feature_cls: Feature class (for Kedro serialization).
        feature_params: Parameters for feature_cls.
        window: Aggregation window in minutes. Default: 15.
        prefix: Prefix for output columns. Default: "{window}m_".
    
    Example:
        >>> offset = OffsetFeature(
        ...     feature_cls=RsiFeature,
        ...     feature_params={"period": 14},
        ...     window=15,
        ... )
        >>> # Outputs: 15m_rsi_14, offset
    """
    
    feature_cls: type[Feature] = None
    feature_params: dict = field(default_factory=dict)
    window: int = 15
    prefix: str | None = None
    
    requires = ["open", "high", "low", "close", "volume", "timestamp"]
    outputs = ["offset"]
    
    def __post_init__(self):
        if self.feature_cls is None:
            raise ValueError("OffsetFeature requires 'feature_cls'")
        self._base = self.feature_cls(**self.feature_params)
        if self.prefix is None:
            self.prefix = f"{self.window}m_"
    
    def output_cols(self, prefix: str = "") -> list[str]:
        base_cols = self._base.output_cols(prefix=f"{prefix}{self.prefix}")
        return base_cols + [f"{prefix}offset"]
    
    def required_cols(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", self.ts_col]
    
    def _resample_ohlcv(self, df: pl.DataFrame, offset: int) -> pl.DataFrame:
        """Resample 1m OHLCV to window-minute bars with given offset."""
        df = df.with_row_index("_row_idx")
        
        df = df.with_columns(
            ((pl.col("_row_idx").cast(pl.Int64) - offset) // self.window).alias("_grp")
        )
        
        agg_exprs = [
            pl.col(self.ts_col).last(),
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ]
        if self.group_col in df.columns:
            agg_exprs.append(pl.col(self.group_col).first())
        
        return df.group_by("_grp", maintain_order=True).agg(agg_exprs)
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute features for all offsets and merge to original timeframe."""
        df = df.sort(self.ts_col)
        original_len = len(df)
        df = df.with_row_index("_orig_idx")
        
        df = df.with_columns(
            (pl.col("_orig_idx") % self.window).cast(pl.UInt8).alias("offset")
        )
        
        offset_results = []
        for offset in range(self.window):
            resampled = self._resample_ohlcv(df.drop(["_orig_idx", "offset"]), offset)
            
            with_feat = self._base.compute_pair(resampled)
            with_feat = with_feat.with_columns(pl.lit(offset).cast(pl.UInt8).alias("_offset"))
            
            for col in self._base.output_cols():
                if col in with_feat.columns:
                    with_feat = with_feat.rename({col: f"{self.prefix}{col}"})
            
            offset_results.append(with_feat)
        
        all_offsets = pl.concat(offset_results)
        
        df = df.with_columns(
            ((pl.col("_orig_idx").cast(pl.Int64) - pl.col("offset").cast(pl.Int64)) // self.window).alias("_grp")
        )
        
        feature_cols = [f"{self.prefix}{col}" for col in self._base.output_cols()]
        
        result = df.join(
            all_offsets.select(["_grp", "_offset"] + feature_cols),
            left_on=["_grp", "offset"],
            right_on=["_grp", "_offset"],
            how="left",
        )
        
        result = result.drop(["_orig_idx", "_grp"])
        assert len(result) == original_len
        
        return result
    
    def compute(self, df: pl.DataFrame,  context: dict[str, Any] | None = None) -> pl.DataFrame:
        return df.group_by(self.group_col, maintain_order=True).map_groups(self.compute_pair)
    
    def to_dict(self) -> dict:
        """Serialize for Kedro."""
        return {
            "feature_cls": self.feature_cls.__name__,
            "feature_params": self.feature_params,
            "window": self.window,
            "prefix": self.prefix,
        }
    
    @classmethod
    def from_dict(cls, data: dict, registry: dict[str, type[Feature]]) -> "OffsetFeature":
        """Deserialize from config."""
        return cls(
            feature_cls=registry[data["feature_cls"]],
            feature_params=data["feature_params"],
            window=data["window"],
            prefix=data.get("prefix"),
        )


