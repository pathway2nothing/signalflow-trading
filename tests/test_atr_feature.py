"""Tests for ATRFeature."""

import numpy as np
import polars as pl
import pytest

from signalflow.feature.atr import ATRFeature


def _make_ohlcv(n: int = 50) -> pl.DataFrame:
    """Create sample OHLCV data with realistic price movements."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2

    return pl.DataFrame({
        "pair": ["BTCUSDT"] * n,
        "timestamp": pl.datetime_range(
            pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1) + pl.duration(hours=n - 1),
            interval="1h", eager=True
        ),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.rand(n) * 1000,
    })


class TestATRFeature:
    def test_output_column_name(self):
        atr = ATRFeature(period=14)
        assert atr._get_output_name() == "atr_14"

    def test_output_column_name_normalized(self):
        atr = ATRFeature(period=14, normalized=True)
        assert atr._get_output_name() == "atr_14_norm"

    def test_warmup_period(self):
        atr = ATRFeature(period=14)
        assert atr.warmup == 15  # period + 1 for shift

    def test_compute_pair_adds_column(self):
        df = _make_ohlcv(50)
        atr = ATRFeature(period=14)
        result = atr.compute_pair(df)
        assert "atr_14" in result.columns

    def test_atr_values_positive(self):
        df = _make_ohlcv(50)
        atr = ATRFeature(period=14)
        result = atr.compute_pair(df)
        # After warmup, all values should be positive
        valid_atr = result["atr_14"].drop_nulls()
        assert (valid_atr > 0).all()

    def test_atr_sma_smoothing(self):
        df = _make_ohlcv(50)
        atr = ATRFeature(period=14, smoothing="sma")
        result = atr.compute_pair(df)
        assert "atr_14" in result.columns
        # SMA should produce values after period rows
        assert result["atr_14"].drop_nulls().len() > 0

    def test_atr_ema_smoothing(self):
        df = _make_ohlcv(50)
        atr = ATRFeature(period=14, smoothing="ema")
        result = atr.compute_pair(df)
        assert "atr_14" in result.columns
        # EMA produces values faster
        assert result["atr_14"].drop_nulls().len() > 0

    def test_different_periods(self):
        df = _make_ohlcv(50)
        atr_14 = ATRFeature(period=14)
        atr_20 = ATRFeature(period=20)
        result_14 = atr_14.compute_pair(df)
        result_20 = atr_20.compute_pair(df)
        assert "atr_14" in result_14.columns
        assert "atr_20" in result_20.columns

    def test_true_range_calculation(self):
        """Verify TR = max(H-L, |H-prevC|, |L-prevC|)."""
        # Create known data where we can verify TR
        df = pl.DataFrame({
            "pair": ["BTCUSDT"] * 5,
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1) + pl.duration(hours=4),
                interval="1h", eager=True
            ),
            "open": [100.0, 102.0, 98.0, 105.0, 103.0],
            "high": [103.0, 105.0, 102.0, 108.0, 106.0],
            "low": [99.0, 100.0, 95.0, 102.0, 101.0],
            "close": [102.0, 101.0, 100.0, 106.0, 104.0],
            "volume": [1000.0] * 5,
        })

        # For row 1 (index 1):
        # H-L = 105-100 = 5
        # |H-prevC| = |105-102| = 3
        # |L-prevC| = |100-102| = 2
        # TR = max(5, 3, 2) = 5

        # For row 2 (index 2):
        # H-L = 102-95 = 7
        # |H-prevC| = |102-101| = 1
        # |L-prevC| = |95-101| = 6
        # TR = max(7, 1, 6) = 7

        atr = ATRFeature(period=2, smoothing="sma")
        result = atr.compute_pair(df)

        # With period=2 SMA, we should get average of TR values
        atr_values = result["atr_2"].to_list()
        # First value is null (shift), second is null (warmup)
        # Third value should be average of TR[1] and TR[2]
        assert atr_values[2] is not None
        assert atr_values[2] == pytest.approx((5.0 + 7.0) / 2, rel=0.01)

    def test_normalized_output(self):
        df = _make_ohlcv(100)  # Need more data for normalization
        atr = ATRFeature(period=14, normalized=True)
        result = atr.compute_pair(df)
        assert "atr_14_norm" in result.columns
        # Normalized values should be roughly centered around 0
        valid_norm = result["atr_14_norm"].drop_nulls()
        assert valid_norm.len() > 0

    def test_requires_columns(self):
        atr = ATRFeature()
        assert "high" in atr.requires
        assert "low" in atr.requires
        assert "close" in atr.requires

    def test_outputs_template(self):
        atr = ATRFeature()
        assert "atr_{period}" in atr.outputs
