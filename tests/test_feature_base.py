"""Tests for signalflow.feature.base.Feature."""

from dataclasses import dataclass
from typing import ClassVar

import polars as pl
import pytest

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
        return df.with_columns(pl.col(self.price_col).rolling_mean(window_size=self.period).alias(f"sma_{self.period}"))

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


class TestGlobalFeatureSourceMethods:
    """Tests for GlobalFeature.get_source_data() and iter_sources()."""

    @pytest.fixture
    def mock_raw_data(self):
        """Create a mock RawData with nested structure for testing."""
        from unittest.mock import MagicMock

        # Create mock DataFrames for different sources
        binance_df = pl.DataFrame({"pair": ["BTCUSDT"], "close": [100.0]})
        okx_df = pl.DataFrame({"pair": ["BTCUSDT"], "close": [101.0]})

        # Create mock accessor for perpetual data type
        perpetual_accessor = MagicMock()
        perpetual_accessor.sources = ["binance", "okx"]
        perpetual_accessor.binance = binance_df
        perpetual_accessor.okx = okx_df
        perpetual_accessor.__contains__ = lambda self, key: key in ["binance", "okx"]

        # Create mock RawData
        raw = MagicMock()
        raw.perpetual = perpetual_accessor
        raw.__contains__ = lambda self, key: key == "perpetual"
        raw.get = MagicMock(
            side_effect=lambda dt, source=None: (
                binance_df if source == "binance" else okx_df if source == "okx" else binance_df
            )
        )

        return raw

    def test_get_source_data_with_specific_source(self, mock_raw_data):
        """Test get_source_data with explicit source parameter."""
        f = GlobalFeature()
        f.get_source_data(mock_raw_data, "perpetual", source="binance")
        mock_raw_data.get.assert_called_with("perpetual", source="binance")

    def test_get_source_data_default_source(self, mock_raw_data):
        """Test get_source_data without source (uses default)."""
        f = GlobalFeature()
        f.get_source_data(mock_raw_data, "perpetual")
        mock_raw_data.get.assert_called_with("perpetual")

    def test_iter_sources_all_sources(self, mock_raw_data):
        """Test iter_sources iterates all available sources."""
        f = GlobalFeature(sources=None)  # Use all sources
        results = list(f.iter_sources(mock_raw_data, "perpetual"))
        assert len(results) == 2
        assert results[0][0] == "binance"
        assert results[1][0] == "okx"

    def test_iter_sources_specific_sources(self, mock_raw_data):
        """Test iter_sources with specific sources list."""
        f = GlobalFeature(sources=["okx"])
        results = list(f.iter_sources(mock_raw_data, "perpetual"))
        assert len(results) == 1
        assert results[0][0] == "okx"

    def test_iter_sources_missing_data_type(self, mock_raw_data):
        """Test iter_sources with non-existent data type returns empty."""
        f = GlobalFeature()
        results = list(f.iter_sources(mock_raw_data, "nonexistent"))
        assert len(results) == 0

    def test_iter_sources_skips_missing_source(self, mock_raw_data):
        """Test iter_sources skips sources not present in accessor."""
        f = GlobalFeature(sources=["binance", "missing_source"])
        results = list(f.iter_sources(mock_raw_data, "perpetual"))
        # Only binance should be yielded since "missing_source" is not in accessor
        assert len(results) == 1
        assert results[0][0] == "binance"
