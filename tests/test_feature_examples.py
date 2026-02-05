"""Tests for signalflow.feature.examples — RSI, SMA, GlobalMeanRSI."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.feature.examples import (
    ExampleGlobalMeanRsiFeature,
    ExampleRsiFeature,
    ExampleSmaFeature,
    _get_norm_window,
    _normalize_zscore,
)


@pytest.fixture
def pair_df_50():
    """50-bar single-pair DataFrame for feature warmup."""
    base = datetime(2024, 1, 1)
    rows = []
    for i in range(50):
        price = 100.0 + 5 * math.sin(i / 3.0) + i * 0.2
        rows.append(
            {
                "pair": "BTCUSDT",
                "timestamp": base + timedelta(minutes=i),
                "open": price - 0.5,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 1000.0,
            }
        )
    return pl.DataFrame(rows)


# ── Helpers ─────────────────────────────────────────────────────────────────


class TestHelpers:
    def test_get_norm_window_default(self):
        assert _get_norm_window(10) == 30  # 10*3

    def test_get_norm_window_min_20(self):
        assert _get_norm_window(5) == 20  # max(15, 20)

    def test_normalize_zscore_shape(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = _normalize_zscore(vals, window=5)
        assert len(result) == 10
        assert np.isnan(result[3])  # window-1 = 4, so idx 0-3 are NaN
        assert not np.isnan(result[4])

    def test_normalize_zscore_constant(self):
        vals = np.array([5.0] * 10)
        result = _normalize_zscore(vals, window=3)
        # std==0, so all should be NaN
        for i in range(2, 10):
            assert np.isnan(result[i])


# ── ExampleRsiFeature ───────────────────────────────────────────────────────


class TestExampleRsiFeature:
    def test_output_cols_default(self):
        f = ExampleRsiFeature()
        assert f.output_cols() == ["rsi_14"]

    def test_output_cols_custom_period(self):
        f = ExampleRsiFeature(period=21)
        assert f.output_cols() == ["rsi_21"]

    def test_warmup(self):
        f = ExampleRsiFeature(period=14)
        assert f.warmup == 42  # 14 * 3

    def test_compute_pair_adds_column(self, pair_df_50):
        f = ExampleRsiFeature(period=14)
        result = f.compute_pair(pair_df_50)
        assert "rsi_14" in result.columns
        assert len(result) == len(pair_df_50)

    def test_rsi_values_range(self, pair_df_50):
        f = ExampleRsiFeature(period=14)
        result = f.compute_pair(pair_df_50)
        non_null = result.filter(pl.col("rsi_14").is_not_null())["rsi_14"]
        assert non_null.min() >= 0
        assert non_null.max() <= 100

    def test_compute_all_pairs(self, ohlcv_100_bars):
        f = ExampleRsiFeature(period=14)
        result = f.compute(ohlcv_100_bars)
        assert "rsi_14" in result.columns
        assert len(result) == len(ohlcv_100_bars)

    def test_normalized_rescales(self, pair_df_50):
        f = ExampleRsiFeature(period=14, normalized=True)
        result = f.compute_pair(pair_df_50)
        col_name = "rsi_14_norm"
        assert col_name in result.columns
        non_null = result.filter(pl.col(col_name).is_not_null())[col_name]
        assert non_null.min() >= -1.0
        assert non_null.max() <= 1.0


# ── ExampleSmaFeature ──────────────────────────────────────────────────────


class TestExampleSmaFeature:
    def test_output_cols(self):
        f = ExampleSmaFeature(period=20)
        assert f.output_cols() == ["sma_20"]

    def test_warmup_base(self):
        f = ExampleSmaFeature(period=10)
        assert f.warmup == 10

    def test_warmup_normalized(self):
        f = ExampleSmaFeature(period=10, normalized=True)
        # 10 + max(10*3, 20) = 10 + 30 = 40
        assert f.warmup == 40

    def test_compute_pair_rolling_mean(self, pair_df_50):
        f = ExampleSmaFeature(period=5)
        result = f.compute_pair(pair_df_50)
        assert "sma_5" in result.columns
        assert len(result) == len(pair_df_50)
        # First 4 values should be null (window=5, min_periods default)
        non_null = result.filter(pl.col("sma_5").is_not_null())
        assert non_null.height > 0

    def test_normalized_applies(self, pair_df_50):
        f = ExampleSmaFeature(period=5, normalized=True)
        result = f.compute_pair(pair_df_50)
        assert "sma_5_norm" in result.columns


# ── ExampleGlobalMeanRsiFeature ─────────────────────────────────────────────


class TestExampleGlobalMeanRsiFeature:
    def test_output_cols(self):
        f = ExampleGlobalMeanRsiFeature(period=14)
        assert f.output_cols() == ["global_mean_rsi_14"]

    def test_output_cols_with_diff(self):
        f = ExampleGlobalMeanRsiFeature(period=14, add_diff=True)
        assert "global_mean_rsi_14" in f.output_cols()
        assert "rsi_14_diff" in f.output_cols()

    def test_compute_cross_pair_mean(self, ohlcv_100_bars):
        f = ExampleGlobalMeanRsiFeature(period=14)
        result = f.compute(ohlcv_100_bars)
        assert "global_mean_rsi_14" in result.columns
        assert len(result) == len(ohlcv_100_bars)
        # RSI column should not be kept (was computed internally)
        assert "rsi_14" not in result.columns

    def test_compute_with_diff(self, ohlcv_100_bars):
        f = ExampleGlobalMeanRsiFeature(period=14, add_diff=True)
        result = f.compute(ohlcv_100_bars)
        assert "rsi_14_diff" in result.columns

    def test_warmup(self):
        f = ExampleGlobalMeanRsiFeature(period=14)
        assert f.warmup == 42
