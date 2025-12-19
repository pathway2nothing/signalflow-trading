# src/signalflow/feature/extractor/sma_extractor.py
from __future__ import annotations

from dataclasses import dataclass
import polars as pl
import pandas as pd

from signalflow.feature.base_extractor import FeatureExtractor


@dataclass
class SmaExtractor(FeatureExtractor):
    """
    SMA per (pair, resample_offset) group.

    Notes:
    - offset_window here is for OffsetResampler (your framework requirement).
      SMA window is `sma_period`.
    - In v1 you said only spot -> keep data_type="spot" by default.
    """
    offset_window: int = 1

    sma_period: int = 20
    price_col: str = "close"
    out_col: str = "sma"

    use_resample: bool = True
    data_type: str = "spot"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sma_period <= 0:
            raise ValueError(f"sma_period must be > 0, got {self.sma_period}")
        if not self.out_col:
            self.out_col = "sma"

    def compute_pl_group(self, group_df: pl.DataFrame, data_context: dict | None) -> pl.DataFrame:
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column: {self.price_col}")

        sma = (
            pl.col(self.price_col)
            .rolling_mean(window_size=self.sma_period, min_samples=self.sma_period)
            .alias(self.out_col)
        )
        return group_df.with_columns(sma)

    def compute_pd_group(self, group_df: pd.DataFrame, data_context: dict | None) -> pd.DataFrame:
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column: {self.price_col}")

        price = pd.to_numeric(group_df[self.price_col], errors="coerce")
        sma = price.rolling(window=self.sma_period, min_periods=self.sma_period).mean()

        out = group_df.copy()
        out[self.out_col] = sma
        return out
