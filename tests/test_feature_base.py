"""Tests for signalflow.feature.base.Feature."""

import pytest
import polars as pl
from dataclasses import dataclass
from typing import ClassVar

from signalflow.feature.base import Feature, GlobalFeature


@dataclass
class DummyFeature(Feature):
    """Simple feature that adds a constant column."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["dummy_out"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.lit(1.0).alias("dummy_out"))


@dataclass
class ParameterizedFeature(Feature):
    """Feature with template-based outputs."""
    requires: ClassVar[list[str]] = ["{price_col}"]
    outputs: ClassVar[list[str]] = ["sma_{period}"]

    price_col: str = "close"
    period: int = 10

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(self.price_col).rolling_mean(window_size=self.period).alias(f"sma_{self.period}")
        )

    @property
    def warmup(self) -> int:
        return self.period


class TestFeatureBase:
    def test_output_cols(self):
        f = DummyFeature()
        assert f.output_cols() == ["dummy_out"]

    def test_required_cols(self):
        f = DummyFeature()
        assert f.required_cols() == ["close"]

    def test_warmup_default(self):
        f = DummyFeature()
        assert f.warmup == 0

    def test_compute_pair_not_implemented(self):
        f = Feature()
        with pytest.raises(NotImplementedError):
            f.compute_pair(pl.DataFrame())


class TestParameterizedFeature:
    def test_output_cols_substitution(self):
        f = ParameterizedFeature(period=20)
        assert f.output_cols() == ["sma_20"]

    def test_required_cols_substitution(self):
        f = ParameterizedFeature(price_col="high")
        assert f.required_cols() == ["high"]

    def test_warmup(self):
        f = ParameterizedFeature(period=14)
        assert f.warmup == 14

    def test_compute_pair(self, sample_ohlcv_df):
        f = ParameterizedFeature(period=3)
        # Get one pair
        pair_df = sample_ohlcv_df.filter(pl.col("pair") == "BTCUSDT")
        result = f.compute_pair(pair_df)
        assert "sma_3" in result.columns
        assert len(result) == len(pair_df)

    def test_compute_all_pairs(self, sample_ohlcv_df):
        f = ParameterizedFeature(period=3)
        result = f.compute(sample_ohlcv_df)
        assert "sma_3" in result.columns
        assert len(result) == len(sample_ohlcv_df)


class TestGlobalFeature:
    def test_compute_not_implemented(self):
        f = GlobalFeature()
        with pytest.raises(NotImplementedError):
            f.compute(pl.DataFrame())

    def test_output_cols_prefix(self):
        f = DummyFeature()
        assert f.output_cols(prefix="pre_") == ["pre_dummy_out"]
