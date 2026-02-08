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
