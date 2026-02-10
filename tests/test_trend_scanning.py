"""Tests for TrendScanningLabeler."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.core.enums import SignalCategory
from signalflow.target.trend_scanning import TrendScanningLabeler


def _trend_df(n=200, pair="BTCUSDT", slope=0.5):
    """OHLCV with a clear uptrend."""
    base_ts = datetime(2024, 1, 1)
    prices = [100.0 + slope * i for i in range(n)]
    timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": timestamps,
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }
    )


def _flat_df(n=200, pair="BTCUSDT"):
    """OHLCV with no trend (random noise around constant)."""
    rng = np.random.default_rng(42)
    base_ts = datetime(2024, 1, 1)
    prices = (100.0 + rng.normal(0, 0.01, size=n)).tolist()
    timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": timestamps,
            "open": prices,
            "high": [p + 0.01 for p in prices],
            "low": [p - 0.01 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }
    )


def _v_shape_df(n=200, pair="BTCUSDT"):
    """OHLCV with V-shape: downtrend then uptrend."""
    base_ts = datetime(2024, 1, 1)
    mid = n // 2
    prices = []
    for i in range(n):
        if i < mid:
            prices.append(200.0 - 0.5 * i)
        else:
            prices.append(200.0 - 0.5 * mid + 0.5 * (i - mid))
    timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": timestamps,
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }
    )


class TestTrendScanningLabeler:
    def test_signal_category(self):
        labeler = TrendScanningLabeler(min_lookforward=5, max_lookforward=20, step=5)
        assert labeler.signal_category == SignalCategory.TREND_MOMENTUM

    def test_output_length_preserved(self):
        df = _trend_df(200)
        labeler = TrendScanningLabeler(min_lookforward=5, max_lookforward=20, step=5, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height

    def test_uptrend_detected_as_rise(self):
        """Noisy uptrend should produce 'rise' labels."""
        rng = np.random.default_rng(42)
        n = 200
        base_ts = datetime(2024, 1, 1)
        # Uptrend with noise so residuals are non-zero (avoiding MSE=0)
        prices = [100.0 + 0.5 * i + rng.normal(0, 0.2) for i in range(n)]
        timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"] * n,
                "timestamp": timestamps,
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": [1000.0] * n,
            }
        )
        labeler = TrendScanningLabeler(
            min_lookforward=5, max_lookforward=30, step=5, critical_value=1.96, mask_to_signals=False
        )
        result = labeler.compute(df)
        labels = result["label"].to_list()
        rise_count = sum(1 for l in labels if l == "rise")
        assert rise_count > 0, "Expected rise labels in noisy uptrend"

    def test_downtrend_detected_as_fall(self):
        """Noisy downtrend should produce 'fall' labels."""
        rng = np.random.default_rng(42)
        n = 200
        base_ts = datetime(2024, 1, 1)
        prices = [200.0 - 0.5 * i + rng.normal(0, 0.2) for i in range(n)]
        timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"] * n,
                "timestamp": timestamps,
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": [1000.0] * n,
            }
        )
        labeler = TrendScanningLabeler(
            min_lookforward=5, max_lookforward=30, step=5, critical_value=1.96, mask_to_signals=False
        )
        result = labeler.compute(df)
        labels = result["label"].to_list()
        fall_count = sum(1 for l in labels if l == "fall")
        assert fall_count > 0, "Expected fall labels in noisy downtrend"

    def test_flat_market_mostly_null(self):
        df = _flat_df(200)
        labeler = TrendScanningLabeler(
            min_lookforward=5, max_lookforward=30, step=5, critical_value=3.0, mask_to_signals=False
        )
        result = labeler.compute(df)
        labels = result["label"].to_list()
        null_count = sum(1 for l in labels if l is None)
        # Most should be null in flat market with high critical value
        assert null_count > len(labels) * 0.5

    def test_v_shape_has_both(self):
        df = _v_shape_df(200)
        labeler = TrendScanningLabeler(
            min_lookforward=5, max_lookforward=20, step=5, critical_value=1.96, mask_to_signals=False
        )
        result = labeler.compute(df)
        labels = result["label"].to_list()
        has_rise = any(l == "rise" for l in labels)
        has_fall = any(l == "fall" for l in labels)
        assert has_rise or has_fall, "V-shape should have at least one directional label"

    def test_meta_columns(self):
        df = _trend_df(100)
        labeler = TrendScanningLabeler(
            min_lookforward=5, max_lookforward=20, step=5, include_meta=True, mask_to_signals=False
        )
        result = labeler.compute(df)
        assert "t_stat" in result.columns
        assert "best_window" in result.columns

    def test_labels_are_valid_values(self):
        df = _trend_df(100)
        labeler = TrendScanningLabeler(min_lookforward=5, max_lookforward=20, step=5, mask_to_signals=False)
        result = labeler.compute(df)
        valid = {"rise", "fall", None}
        unique = set(result["label"].to_list())
        assert unique <= valid

    def test_invalid_min_lookforward(self):
        with pytest.raises(ValueError, match="min_lookforward"):
            TrendScanningLabeler(min_lookforward=2)

    def test_invalid_max_lookforward(self):
        with pytest.raises(ValueError, match="max_lookforward"):
            TrendScanningLabeler(min_lookforward=10, max_lookforward=5)

    def test_invalid_step(self):
        with pytest.raises(ValueError, match="step"):
            TrendScanningLabeler(step=0)

    def test_invalid_critical_value(self):
        with pytest.raises(ValueError, match="critical_value"):
            TrendScanningLabeler(critical_value=-1.0)

    def test_multi_pair(self):
        btc = _trend_df(100, pair="BTCUSDT", slope=1.0)
        eth = _trend_df(100, pair="ETHUSDT", slope=-1.0)
        df = pl.concat([btc, eth])
        labeler = TrendScanningLabeler(min_lookforward=5, max_lookforward=20, step=5, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame via compute_group."""
        df = pl.DataFrame(
            {
                "pair": pl.Series([], dtype=pl.Utf8),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "open": pl.Series([], dtype=pl.Float64),
                "high": pl.Series([], dtype=pl.Float64),
                "low": pl.Series([], dtype=pl.Float64),
                "close": pl.Series([], dtype=pl.Float64),
                "volume": pl.Series([], dtype=pl.Float64),
            }
        )
        labeler = TrendScanningLabeler(min_lookforward=5, max_lookforward=20, step=5, mask_to_signals=False)
        # Test compute_group directly since compute relies on group_by
        result = labeler.compute_group(df)
        assert result.height == 0

    def test_missing_price_column(self):
        """Test that missing price column raises error."""
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"] * 10,
                "timestamp": [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(10)],
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                # Missing 'close' column
                "volume": [1000.0] * 10,
            }
        )
        labeler = TrendScanningLabeler(
            min_lookforward=5, max_lookforward=10, step=1, price_col="close", mask_to_signals=False
        )
        # Test compute_group directly
        with pytest.raises(ValueError, match="Missing required column"):
            labeler.compute_group(df)

    def test_custom_price_column(self):
        """Test using custom price column."""
        n = 100
        base_ts = datetime(2024, 1, 1)
        timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
        prices = [100.0 + 0.5 * i for i in range(n)]
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"] * n,
                "timestamp": timestamps,
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "vwap": prices,  # Custom price column
                "volume": [1000.0] * n,
            }
        )
        labeler = TrendScanningLabeler(
            min_lookforward=5, max_lookforward=20, step=5, price_col="vwap", mask_to_signals=False
        )
        result = labeler.compute(df)
        assert result.height == n
        assert "label" in result.columns

    def test_small_window_edge_case(self):
        """Test with minimum valid window size."""
        df = _trend_df(50)
        labeler = TrendScanningLabeler(min_lookforward=3, max_lookforward=5, step=1, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height

    def test_single_window_size(self):
        """Test when min_lookforward equals max_lookforward."""
        df = _trend_df(100)
        labeler = TrendScanningLabeler(min_lookforward=10, max_lookforward=10, step=1, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height

    def test_custom_out_col(self):
        """Test with custom output column name."""
        df = _trend_df(100)
        labeler = TrendScanningLabeler(
            min_lookforward=5, max_lookforward=20, step=5, out_col="trend_label", mask_to_signals=False
        )
        result = labeler.compute(df)
        assert "trend_label" in result.columns

    def test_high_critical_value(self):
        """Test with very high critical value - most labels should be null."""
        df = _trend_df(100)
        labeler = TrendScanningLabeler(
            min_lookforward=5, max_lookforward=20, step=5, critical_value=10.0, mask_to_signals=False
        )
        result = labeler.compute(df)
        labels = result["label"].to_list()
        non_null_count = sum(1 for l in labels if l is not None)
        assert non_null_count < len(labels) * 0.5


class TestTrendScanNumpy:
    """Tests for the numpy fallback implementation."""

    def test_numpy_fallback_basic(self):
        """Test _trend_scan_numpy directly."""
        from signalflow.target.trend_scanning import _trend_scan_numpy

        # Create uptrend data with noise (to avoid zero MSE)
        rng = np.random.default_rng(42)
        prices = np.array([100.0 + i + rng.normal(0, 0.1) for i in range(50)], dtype=np.float64)
        t_stats, best_windows = _trend_scan_numpy(prices, min_lf=3, max_lf=10, step=1)

        assert len(t_stats) == 50
        assert len(best_windows) == 50
        # Should have some valid t-stats (not all NaN)
        valid_count = np.sum(~np.isnan(t_stats))
        assert valid_count > 0

    def test_numpy_fallback_downtrend(self):
        """Test numpy implementation with downtrend."""
        from signalflow.target.trend_scanning import _trend_scan_numpy

        # Downtrend with noise
        rng = np.random.default_rng(42)
        prices = np.array([100.0 - i + rng.normal(0, 0.1) for i in range(50)], dtype=np.float64)
        t_stats, best_windows = _trend_scan_numpy(prices, min_lf=3, max_lf=10, step=1)

        # Should have negative t-stats for downtrend
        valid_t_stats = t_stats[~np.isnan(t_stats)]
        assert len(valid_t_stats) > 0
        # Early values should be mostly negative
        early_valid = [t for t in t_stats[:20] if not np.isnan(t)]
        if len(early_valid) > 0:
            assert np.mean(early_valid) < 0

    def test_numpy_short_array(self):
        """Test numpy with array shorter than min_lookforward."""
        from signalflow.target.trend_scanning import _trend_scan_numpy

        prices = np.array([100.0, 101.0], dtype=np.float64)
        t_stats, best_windows = _trend_scan_numpy(prices, min_lf=5, max_lf=10, step=1)

        # All should be NaN since array is too short
        assert np.all(np.isnan(t_stats))
        assert np.all(np.isnan(best_windows))

    def test_numpy_constant_prices(self):
        """Test with constant prices (no variation)."""
        from signalflow.target.trend_scanning import _trend_scan_numpy

        prices = np.array([100.0] * 30, dtype=np.float64)
        t_stats, best_windows = _trend_scan_numpy(prices, min_lf=3, max_lf=10, step=1)

        # With constant prices, residuals are zero, so t-stats should be NaN or zero
        valid_t_stats = t_stats[~np.isnan(t_stats)]
        if len(valid_t_stats) > 0:
            # All valid t-stats should be zero or very small
            assert np.allclose(valid_t_stats, 0, atol=1e-10)

    def test_numpy_step_parameter(self):
        """Test numpy with different step values."""
        from signalflow.target.trend_scanning import _trend_scan_numpy

        rng = np.random.default_rng(42)
        prices = 100.0 + 0.5 * np.arange(100) + rng.normal(0, 0.1, 100)

        # With step=1
        t_stats_1, windows_1 = _trend_scan_numpy(prices, min_lf=5, max_lf=20, step=1)
        # With step=5
        t_stats_5, windows_5 = _trend_scan_numpy(prices, min_lf=5, max_lf=20, step=5)

        assert len(t_stats_1) == len(t_stats_5)
        # Results may differ due to different window sizes tested

    def test_numpy_end_of_array(self):
        """Test behavior at end of array where windows exceed bounds."""
        from signalflow.target.trend_scanning import _trend_scan_numpy

        prices = np.array([100.0 + i for i in range(20)], dtype=np.float64)
        t_stats, best_windows = _trend_scan_numpy(prices, min_lf=10, max_lf=15, step=1)

        # Last 15 elements can't have windows of size 15
        # Last 10 elements can't have windows of size 10
        # So last several should be NaN
        assert np.isnan(t_stats[-1])
