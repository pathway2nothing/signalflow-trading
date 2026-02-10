"""Tests for VolatilityRegimeLabeler."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.core.enums import SignalCategory
from signalflow.target.volatility_labeler import VolatilityRegimeLabeler


def _ohlcv_df(n=2000, pair="BTCUSDT", base_price=100.0, seed=42):
    """Random walk OHLCV with small normal returns."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, 0.001, size=n)
    log_ret[0] = 0.0
    prices = np.exp(np.log(base_price) + np.cumsum(log_ret))
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
            "volume": [1000.0] * n,
        }
    )


def _inject_volatile_period(df, start_idx, length, volatility_mult=5.0, seed=99):
    """Inject a high-volatility period into the DataFrame."""
    rng = np.random.default_rng(seed)
    close = df["close"].to_list()
    for i in range(start_idx, min(start_idx + length, len(close))):
        change = rng.normal(0, 0.001 * volatility_mult)
        close[i] = close[i - 1] * np.exp(change) if i > 0 else close[i]
    return df.with_columns(pl.Series(name="close", values=close))


class TestVolatilityRegimeLabeler:
    def test_signal_category(self):
        labeler = VolatilityRegimeLabeler()
        assert labeler.signal_category == SignalCategory.VOLATILITY

    def test_output_length_preserved(self):
        df = _ohlcv_df(500)
        labeler = VolatilityRegimeLabeler(horizon=30, lookback_window=200, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height

    def test_labels_are_valid_values(self):
        df = _ohlcv_df(2000)
        labeler = VolatilityRegimeLabeler(horizon=30, lookback_window=200, mask_to_signals=False)
        result = labeler.compute(df)
        valid = {"vol_high", "vol_low", None}
        labels = result["label"].to_list()
        unique_labels = set(labels)
        assert unique_labels <= valid

    def test_meta_columns(self):
        df = _ohlcv_df(500)
        labeler = VolatilityRegimeLabeler(horizon=30, lookback_window=200, include_meta=True, mask_to_signals=False)
        result = labeler.compute(df)
        assert "realized_vol" in result.columns
        assert "vol_percentile" in result.columns

    def test_no_meta_columns_by_default(self):
        df = _ohlcv_df(500)
        labeler = VolatilityRegimeLabeler(horizon=30, lookback_window=200, mask_to_signals=False)
        result = labeler.compute(df)
        assert "realized_vol" not in result.columns

    def test_volatile_period_detected(self):
        df = _ohlcv_df(2000)
        df = _inject_volatile_period(df, 500, 200, volatility_mult=10.0)
        labeler = VolatilityRegimeLabeler(
            horizon=30,
            lookback_window=500,
            upper_quantile=0.8,
            lower_quantile=0.2,
            mask_to_signals=False,
            include_meta=True,
        )
        result = labeler.compute(df)
        # Before the volatile period, some bars should get labeled
        labels = result["label"].to_list()
        assert any(l == "vol_high" for l in labels), "Expected at least one vol_high label"

    def test_invalid_quantiles_raises(self):
        with pytest.raises(ValueError, match="lower_quantile"):
            VolatilityRegimeLabeler(upper_quantile=0.3, lower_quantile=0.7)

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon"):
            VolatilityRegimeLabeler(horizon=0)

    def test_multi_pair(self):
        btc = _ohlcv_df(500, pair="BTCUSDT")
        eth = _ohlcv_df(500, pair="ETHUSDT", seed=99)
        df = pl.concat([btc, eth])
        labeler = VolatilityRegimeLabeler(horizon=30, lookback_window=200, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height
        pairs = result["pair"].unique().to_list()
        assert set(pairs) == {"BTCUSDT", "ETHUSDT"}

    def test_short_df_no_crash(self):
        """Very short DataFrame should not crash, may produce all-null labels."""
        df = _ohlcv_df(10)
        labeler = VolatilityRegimeLabeler(horizon=5, lookback_window=5, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height
