"""Tests for OffsetFeature."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.feature.offset_feature import OffsetFeature
from signalflow.core.registry import default_registry
from signalflow.core import SfComponentType


TS = datetime(2024, 1, 1)


def _make_df(n=100, pair="BTCUSDT"):
    rows = []
    for i in range(n):
        rows.append(
            {
                "pair": pair,
                "timestamp": TS + timedelta(minutes=i),
                "open": 100.0 + i * 0.1,
                "high": 101.0 + i * 0.1,
                "low": 99.0 + i * 0.1,
                "close": 100.0 + i * 0.1,
                "volume": 1000.0,
            }
        )
    return pl.DataFrame(rows)


class TestOffsetFeatureValidation:
    def test_no_feature_name_raises(self):
        with pytest.raises(ValueError, match="feature_name"):
            OffsetFeature(feature_name=None)

    def test_unknown_feature_raises(self):
        with pytest.raises(Exception):
            OffsetFeature(feature_name="nonexistent_feature_12345")


class TestOffsetFeatureBasic:
    def test_output_cols(self):
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=15)
        cols = f.output_cols()
        assert "offset" in cols

    def test_required_cols(self):
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=15)
        req = f.required_cols()
        assert "open" in req
        assert "close" in req
        assert "timestamp" in req

    def test_prefix_default(self):
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=15)
        assert f.prefix == "15m_"

    def test_prefix_custom(self):
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=30, prefix="custom_")
        assert f.prefix == "custom_"

    def test_to_dict(self):
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=15, prefix="pfx_")
        d = f.to_dict()
        assert d["feature_name"] == "example/rsi"
        assert d["window"] == 15
        assert d["prefix"] == "pfx_"

    def test_from_dict(self):
        data = {
            "feature_name": "example/rsi",
            "feature_params": {"period": 14},
            "window": 15,
            "prefix": "pfx_",
        }
        f = OffsetFeature.from_dict(data)
        assert f.window == 15
        assert f.prefix == "pfx_"

    def test_from_dict_no_prefix(self):
        """Test from_dict with missing prefix (should use default)."""
        data = {
            "feature_name": "example/rsi",
            "feature_params": {"period": 14},
            "window": 10,
        }
        f = OffsetFeature.from_dict(data)
        assert f.prefix == "10m_"


class TestOffsetFeatureCompute:
    """Tests for OffsetFeature compute methods."""

    def test_compute_pair_single_pair(self):
        """Test compute_pair with single pair data."""
        df = _make_df(100)
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=5)

        result = f.compute_pair(df)

        # Should have same length
        assert result.height == df.height
        # Should have offset column
        assert "offset" in result.columns
        # Should have prefixed RSI column
        rsi_col = "5m_rsi_14"
        assert rsi_col in result.columns

    def test_compute_pair_preserves_columns(self):
        """Test that compute_pair preserves original columns."""
        df = _make_df(100)
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=5)

        result = f.compute_pair(df)

        # Original columns should still exist
        for col in ["pair", "timestamp", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_compute_all_pairs(self):
        """Test compute with multiple pairs."""
        df1 = _make_df(100, pair="BTCUSDT")
        df2 = _make_df(100, pair="ETHUSDT")
        df = pl.concat([df1, df2])

        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=5)

        result = f.compute(df)

        # Should have same total length
        assert result.height == df.height
        # Both pairs should be present
        pairs = result["pair"].unique().to_list()
        assert "BTCUSDT" in pairs
        assert "ETHUSDT" in pairs

    def test_compute_offset_values(self):
        """Test that offset column has correct values."""
        df = _make_df(100)
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=5)

        result = f.compute_pair(df)

        offsets = result["offset"].unique().sort().to_list()
        # Should have offsets 0, 1, 2, 3, 4 for window=5
        assert offsets == [0, 1, 2, 3, 4]

    def test_compute_small_window(self):
        """Test with small window size."""
        df = _make_df(50)
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 5}, window=3)

        result = f.compute_pair(df)
        assert result.height == df.height

    def test_compute_large_window(self):
        """Test with larger window size."""
        df = _make_df(200)
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=30)

        result = f.compute_pair(df)
        assert result.height == df.height
        offsets = result["offset"].unique().to_list()
        assert len(offsets) == 30

    def test_compute_with_sma(self):
        """Test with SMA feature instead of RSI."""
        df = _make_df(100)
        f = OffsetFeature(feature_name="example/sma", feature_params={"period": 10}, window=5)

        result = f.compute_pair(df)
        assert result.height == df.height
        assert "5m_sma_10" in result.columns


class TestOffsetFeatureGlobal:
    """Tests for OffsetFeature with GlobalFeature base."""

    def test_is_global_detection(self):
        """Test that _is_global flag is set correctly for GlobalFeature."""
        # Note: Need to find an actual GlobalFeature in the registry
        # For now, test with regular feature which should have _is_global=False
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=5)
        assert f._is_global is False

    def test_compute_pair_raises_for_global(self):
        """Test that compute_pair raises error for global features."""
        # This would require a registered GlobalFeature
        # Skip if no global feature available
        pass


class TestOffsetFeatureResample:
    """Tests for resampling logic."""

    def test_resample_produces_correct_aggregation(self):
        """Test that OHLCV resampling is correct."""
        df = _make_df(100)
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=10)

        # Access private method for testing
        result = f._resample_ohlcv(df, offset=0)

        # Should have fewer rows (aggregated)
        assert result.height < df.height
        # Should have OHLCV columns
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_resample_different_offsets(self):
        """Test resampling with different offsets produces different timestamps."""
        df = _make_df(100)
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=10)

        result0 = f._resample_ohlcv(df, offset=0)
        result5 = f._resample_ohlcv(df, offset=5)

        # Both should produce results
        assert result0.height > 0
        assert result5.height > 0
        # Timestamps should differ due to different starting points
        ts0 = result0[f.ts_col].to_list()
        ts5 = result5[f.ts_col].to_list()
        # The first timestamp should differ (different alignment)
        assert ts0[0] != ts5[0]

    def test_resample_with_pair_column(self):
        """Test that pair column is preserved in resampling."""
        df = _make_df(100)
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=5)

        result = f._resample_ohlcv(df, offset=0)
        assert "pair" in result.columns


class TestOffsetFeatureEdgeCases:
    """Edge case tests for OffsetFeature."""

    def test_short_data(self):
        """Test with data shorter than window."""
        df = _make_df(10)
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 5}, window=5)

        result = f.compute_pair(df)
        assert result.height == df.height

    def test_data_length_equals_window(self):
        """Test when data length equals window size."""
        df = _make_df(15)
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 5}, window=15)

        result = f.compute_pair(df)
        assert result.height == df.height

    def test_window_of_one(self):
        """Test with window=1 (essentially no aggregation)."""
        df = _make_df(50)
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=1)

        result = f.compute_pair(df)
        assert result.height == df.height

    def test_consistent_results(self):
        """Test that same input produces same output."""
        df = _make_df(100)
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=5)

        result1 = f.compute_pair(df.clone())
        result2 = f.compute_pair(df.clone())

        # Results should be identical
        assert result1.equals(result2)
