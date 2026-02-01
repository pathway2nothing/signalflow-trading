from dataclasses import dataclass, field
from typing import Any

import polars as pl

from signalflow.core import default_registry, SfComponentType
from signalflow.feature.base import Feature, GlobalFeature


@dataclass
class OffsetFeature(Feature):
    """Multi-timeframe feature via offset resampling.
    
    Creates `window` parallel time series with different offsets.
    Each offset computes features as if on `window`-minute bars.
    
    Supports both regular Feature (compute_pair) and GlobalFeature (compute).
    
    Args:
        feature_name: Registered component name (sf_component name).
        feature_params: Parameters for feature class.
        window: Aggregation window in minutes. Default: 15.
        prefix: Prefix for output columns. Default: "{window}m_".
    
    Example:
        >>> offset = OffsetFeature(
        ...     feature_name="test_rsi",
        ...     feature_params={"period": 14},
        ...     window=15,
        ... )
        >>> # Outputs: 15m_rsi_14, offset
        
        >>> # With GlobalFeature
        >>> offset = OffsetFeature(
        ...     feature_name="global/market_log_return",
        ...     feature_params={},
        ...     window=15,
        ... )
    """
    
    feature_name: str = None
    feature_params: dict = field(default_factory=dict)
    window: int = 15
    prefix: str | None = None
    
    requires = ["open", "high", "low", "close", "volume", "timestamp"]
    outputs = ["offset"]
    
    def __post_init__(self):
        if self.feature_name is None:
            raise ValueError("OffsetFeature requires 'feature_name'")
        
        self._feature_cls = default_registry.get(SfComponentType.FEATURE, self.feature_name)
        self._base = self._feature_cls(**self.feature_params)
        self._is_global = isinstance(self._base, GlobalFeature)
        
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
    
    def _compute_base_feature(self, resampled: pl.DataFrame) -> pl.DataFrame:
        """Compute base feature - handles both Feature and GlobalFeature."""
        if self._is_global:
            return self._base.compute(resampled)
        else:
            return self._base.compute_pair(resampled)
    
    def _compute_single_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute features for single pair (non-global base)."""
        df = df.sort(self.ts_col)
        original_len = len(df)
        df = df.with_row_index("_orig_idx")
        
        df = df.with_columns(
            (pl.col("_orig_idx") % self.window).cast(pl.UInt8).alias("offset")
        )
        
        offset_results = []
        for offset in range(self.window):
            resampled = self._resample_ohlcv(df.drop(["_orig_idx", "offset"]), offset)
            
            with_feat = self._compute_base_feature(resampled)
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
    
    def _compute_all_pairs_global(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute features for all pairs with global base feature."""
        df = df.sort([self.group_col, self.ts_col])
        original_len = len(df)
        
        df = df.with_columns(
            pl.col(self.ts_col).rank("ordinal").over(self.group_col).cast(pl.UInt32).alias("_orig_idx") - 1
        )
        
        df = df.with_columns(
            (pl.col("_orig_idx") % self.window).cast(pl.UInt8).alias("offset")
        )
        
        offset_results = []
        for offset in range(self.window):
            resampled = (
                df.drop(["_orig_idx", "offset"])
                .with_columns(
                    (pl.col(self.ts_col).rank("ordinal").over(self.group_col).cast(pl.Int64) - 1 - offset)
                    .floordiv(self.window)
                    .alias("_grp")
                )
                .group_by([self.group_col, "_grp"], maintain_order=True)
                .agg([
                    pl.col(self.ts_col).last(),
                    pl.col("open").first(),
                    pl.col("high").max(),
                    pl.col("low").min(),
                    pl.col("close").last(),
                    pl.col("volume").sum(),
                ])
            )
            
            with_feat = self._compute_base_feature(resampled)
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
            all_offsets.select([self.group_col, "_grp", "_offset"] + feature_cols),
            left_on=[self.group_col, "_grp", "offset"],
            right_on=[self.group_col, "_grp", "_offset"],
            how="left",
        )
        
        result = result.drop(["_orig_idx", "_grp"])
        assert len(result) == original_len
        
        return result
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute for single pair (only for non-global base)."""
        if self._is_global:
            raise NotImplementedError("GlobalFeature base requires compute(), not compute_pair()")
        return self._compute_single_pair(df)
    
    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Compute for all pairs."""
        if self._is_global:
            return self._compute_all_pairs_global(df)
        else:
            return df.group_by(self.group_col, maintain_order=True).map_groups(self._compute_single_pair)
    
    def to_dict(self) -> dict:
        """Serialize for Kedro."""
        return {
            "feature_name": self.feature_name,
            "feature_params": self.feature_params,
            "window": self.window,
            "prefix": self.prefix,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "OffsetFeature":
        """Deserialize from config."""
        return cls(
            feature_name=data["feature_name"],
            feature_params=data["feature_params"],
            window=data["window"],
            prefix=data.get("prefix"),
        )
