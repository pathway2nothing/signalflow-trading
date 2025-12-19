# src/signalflow/feature/extractor/sma.py
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from signalflow.feature.extractor.base_extractor import FeatureExtractor
from signalflow.core import RawDataView


@dataclass
class SmaExtractor(FeatureExtractor):
    """
    Simple Moving Average (SMA) extractor (Polars).

    Expects raw data (per row) to contain at least:
      - pair column (default: "pair")
      - timestamp column (default: "timestamp")
      - price column (default: "close")

    Output:
      - [pair, timestamp, <out_col>]
    """
    window: int = 20
    price_col: str = "close"
    out_col: str | None = None

    pair_col: str = "pair"
    ts_col: str = "timestamp"
    raw_data_type: str = "spot"

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be > 0")
        if self.out_col is None:
            self.out_col = f"sma_{self.window}"

    def extract(self, raw_data_view: RawDataView, **kwargs) -> pl.DataFrame:
        """
        Fetch raw data from RawDataView and compute SMA.

        NOTE: Replace the one line that fetches `df` with whatever your RawDataView API is.
        """
        df = self._get_polars(raw_data_view)

        required = [self.pair_col, self.ts_col, self.price_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"SmaExtractor: missing columns in raw data: {missing}")

        df = df.sort([self.pair_col, self.ts_col])

        out = (
            df.select([self.pair_col, self.ts_col, self.price_col])
            .with_columns(
                pl.col(self.price_col)
                .rolling_mean(window_size=self.window, min_periods=self.window)
                .over(self.pair_col)
                .alias(self.out_col)
            )
            .select([self.pair_col, self.ts_col, self.out_col])
        )
        return out

    def _get_polars(self, raw_data_view: RawDataView) -> pl.DataFrame:
        """
        Adapter for RawDataView.

        Pick whichever exists in your codebase and delete the rest.
        """
        if hasattr(raw_data_view, "get"):
            try:
                return raw_data_view.get(self.raw_data_type, backend="polars")
            except TypeError:
                return raw_data_view.get(self.raw_data_type)

        if hasattr(raw_data_view, "to_polars"):
            return raw_data_view.to_polars(self.raw_data_type)

        raise AttributeError(
            "RawDataView has no supported method to fetch Polars DataFrame. "
            "Expected `.get(...)` or `.to_polars(...)`."
        )
