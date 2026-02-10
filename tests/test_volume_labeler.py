"""Tests for VolumeRegimeLabeler."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.core.enums import SignalCategory
from signalflow.target.volume_labeler import VolumeRegimeLabeler


def _ohlcv_df(n=2000, pair="BTCUSDT", base_vol=1000.0, seed=42):
    """Random walk OHLCV with constant volume."""
    rng = np.random.default_rng(seed)
    prices = np.exp(np.log(100.0) + np.cumsum(rng.normal(0, 0.001, size=n)))
    base_ts = datetime(2024, 1, 1)
    timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": timestamps,
            "open": prices.tolist(),
            "high": (prices * 1.001).tolist(),
            "low": (prices * 0.999).tolist(),
            "close": prices.tolist(),
            "volume": [base_vol] * n,
        }
    )


def _inject_volume_spike(df, start_idx, length, multiplier=5.0):
    """Inject a volume spike period."""
    vol = df["volume"].to_list()
    for i in range(start_idx, min(start_idx + length, len(vol))):
        vol[i] *= multiplier
    return df.with_columns(pl.Series(name="volume", values=vol))


def _inject_volume_drought(df, start_idx, length, multiplier=0.1):
    """Inject a volume drought period."""
    vol = df["volume"].to_list()
    for i in range(start_idx, min(start_idx + length, len(vol))):
        vol[i] *= multiplier
    return df.with_columns(pl.Series(name="volume", values=vol))


class TestVolumeRegimeLabeler:
    def test_signal_category(self):
        labeler = VolumeRegimeLabeler()
        assert labeler.signal_category == SignalCategory.VOLUME_LIQUIDITY

    def test_output_length_preserved(self):
        df = _ohlcv_df(500)
        labeler = VolumeRegimeLabeler(horizon=30, vol_sma_window=200, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height

    def test_constant_volume_mostly_null(self):
        df = _ohlcv_df(500)
        labeler = VolumeRegimeLabeler(horizon=30, vol_sma_window=200, mask_to_signals=False)
        result = labeler.compute(df)
        labels = result["label"].to_list()
        # Constant volume -> ratio ~1.0, no spikes or droughts
        non_null = [l for l in labels if l is not None]
        assert len(non_null) == 0, "Constant volume should have no extreme labels"

    def test_volume_spike_detected(self):
        df = _ohlcv_df(2000)
        df = _inject_volume_spike(df, 800, 200, multiplier=5.0)
        labeler = VolumeRegimeLabeler(
            horizon=30, vol_sma_window=500, spike_threshold=2.0, drought_threshold=0.3, mask_to_signals=False
        )
        result = labeler.compute(df)
        labels = result["label"].to_list()
        # Bars before the spike looking forward into it should see "volume_spike"
        has_spike = any(l == "volume_spike" for l in labels)
        assert has_spike, "Expected volume_spike labels before high-volume period"

    def test_volume_drought_detected(self):
        df = _ohlcv_df(2000)
        df = _inject_volume_drought(df, 800, 200, multiplier=0.05)
        labeler = VolumeRegimeLabeler(
            horizon=30, vol_sma_window=500, spike_threshold=2.0, drought_threshold=0.3, mask_to_signals=False
        )
        result = labeler.compute(df)
        labels = result["label"].to_list()
        has_drought = any(l == "volume_drought" for l in labels)
        assert has_drought, "Expected volume_drought labels before low-volume period"

    def test_labels_are_valid_values(self):
        df = _ohlcv_df(500)
        df = _inject_volume_spike(df, 200, 100, multiplier=3.0)
        labeler = VolumeRegimeLabeler(horizon=30, vol_sma_window=200, mask_to_signals=False)
        result = labeler.compute(df)
        valid = {"volume_spike", "volume_drought", None}
        unique = set(result["label"].to_list())
        assert unique <= valid

    def test_meta_columns(self):
        df = _ohlcv_df(500)
        labeler = VolumeRegimeLabeler(horizon=30, vol_sma_window=200, include_meta=True, mask_to_signals=False)
        result = labeler.compute(df)
        assert "volume_ratio" in result.columns

    def test_invalid_horizon(self):
        with pytest.raises(ValueError, match="horizon"):
            VolumeRegimeLabeler(horizon=0)

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError, match="drought_threshold"):
            VolumeRegimeLabeler(drought_threshold=3.0, spike_threshold=2.0)

    def test_multi_pair(self):
        btc = _ohlcv_df(500, pair="BTCUSDT")
        eth = _ohlcv_df(500, pair="ETHUSDT", seed=99)
        df = pl.concat([btc, eth])
        labeler = VolumeRegimeLabeler(horizon=30, vol_sma_window=200, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height
