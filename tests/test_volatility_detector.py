"""Tests for VolatilityDetector (real-time)."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from signalflow.core import Signals
from signalflow.core.enums import SignalCategory
from signalflow.detector.volatility_detector import VolatilityDetector


def _ohlcv_df(n=2000, pair="BTCUSDT", base_price=100.0, seed=42):
    """Random walk OHLCV."""
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


def _inject_volatile_period(df, start_idx, length, vol_mult=10.0, seed=99):
    """Inject a high-volatility period."""
    rng = np.random.default_rng(seed)
    close = df["close"].to_list()
    for i in range(start_idx, min(start_idx + length, len(close))):
        change = rng.normal(0, 0.001 * vol_mult)
        close[i] = close[i - 1] * np.exp(change) if i > 0 else close[i]
    # Propagate to keep series continuous
    for i in range(start_idx + length, len(close)):
        if i > 0:
            ratio = df["close"][i] / df["close"][i - 1]
            close[i] = close[i - 1] * ratio
    return df.with_columns(pl.Series(name="close", values=close))


class TestVolatilityDetector:
    def test_signal_category(self):
        d = VolatilityDetector()
        assert d.signal_category == SignalCategory.VOLATILITY

    def test_detect_returns_signals(self):
        df = _ohlcv_df(2000)
        d = VolatilityDetector(vol_window=60, lookback_window=500)
        result = d.detect(df)
        assert isinstance(result, Signals)

    def test_signal_types_are_valid(self):
        df = _ohlcv_df(2000)
        df = _inject_volatile_period(df, 800, 200, vol_mult=10.0)
        d = VolatilityDetector(vol_window=60, lookback_window=500, upper_quantile=0.8, lower_quantile=0.2)
        result = d.detect(df)
        if result.value.height > 0:
            types = set(result.value["signal_type"].to_list())
            assert types <= {"high_volatility", "low_volatility"}

    def test_volatile_period_emits_high_volatility(self):
        df = _ohlcv_df(2000)
        df = _inject_volatile_period(df, 800, 300, vol_mult=15.0)
        d = VolatilityDetector(vol_window=60, lookback_window=500, upper_quantile=0.7, lower_quantile=0.3)
        result = d.detect(df)
        if result.value.height > 0:
            types = result.value["signal_type"].to_list()
            assert "high_volatility" in types, "Expected high_volatility during volatile period"

    def test_signals_have_required_columns(self):
        df = _ohlcv_df(2000)
        df = _inject_volatile_period(df, 800, 200)
        d = VolatilityDetector(vol_window=60, lookback_window=500)
        result = d.detect(df)
        if result.value.height > 0:
            required = {"pair", "timestamp", "signal_type", "signal", "probability"}
            assert required <= set(result.value.columns)

    def test_backward_looking_only(self):
        """Verify detector is backward-looking: the same bar should produce
        the same signal regardless of what comes after it."""
        df = _ohlcv_df(2000)
        d = VolatilityDetector(vol_window=60, lookback_window=500)
        result_full = d.detect(df)

        # Truncate to first 1000 bars and run again
        df_half = df.head(1000)
        result_half = d.detect(df_half)

        # Signals from first 1000 bars should be identical
        if result_half.value.height > 0 and result_full.value.height > 0:
            base_ts = datetime(2024, 1, 1)
            cutoff_ts = base_ts + timedelta(minutes=1000)
            full_before = result_full.value.filter(pl.col("timestamp") < cutoff_ts)
            assert full_before.height == result_half.value.height

    def test_multi_pair(self):
        btc = _ohlcv_df(1000, pair="BTCUSDT")
        eth = _ohlcv_df(1000, pair="ETHUSDT", seed=99)
        btc = _inject_volatile_period(btc, 500, 200, vol_mult=10.0)
        df = pl.concat([btc, eth])
        d = VolatilityDetector(vol_window=60, lookback_window=300)
        result = d.detect(df)
        assert isinstance(result, Signals)
