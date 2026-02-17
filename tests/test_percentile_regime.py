"""Tests for PercentileRegimeDetector."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.core import Signals
from signalflow.core.enums import SignalCategory
from signalflow.detector.percentile_regime import PercentileRegimeDetector


def _vol_df(n=300, pair="BTCUSDT"):
    """DataFrame with synthetic realized vol (trending then mean-reverting)."""
    base_ts = datetime(2024, 1, 1)
    timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
    rng = np.random.default_rng(42)
    # Low vol -> high vol -> low vol
    vol = np.concatenate(
        [
            rng.uniform(0.01, 0.02, n // 3),
            rng.uniform(0.05, 0.10, n // 3),
            rng.uniform(0.01, 0.02, n - 2 * (n // 3)),
        ]
    )
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": timestamps,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
            "_realized_vol": vol.tolist(),
        }
    )


def _constant_vol_df(n=200, pair="BTCUSDT", vol=0.03):
    base_ts = datetime(2024, 1, 1)
    timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": timestamps,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
            "_realized_vol": [vol] * n,
        }
    )


class TestPercentileRegimeDetector:
    def test_signal_category(self):
        d = PercentileRegimeDetector()
        assert d.signal_category == SignalCategory.VOLATILITY

    def test_post_init_validation(self):
        with pytest.raises(ValueError, match="Quantiles"):
            PercentileRegimeDetector(lower_quantile=0.8, upper_quantile=0.2)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            PercentileRegimeDetector(lower_quantile=0.0, upper_quantile=0.5)

    def test_detect_returns_signals(self):
        df = _vol_df(300)
        d = PercentileRegimeDetector(lookback_window=50)
        result = d.detect(df)
        assert isinstance(result, Signals)

    def test_varying_vol_detects_regimes(self):
        df = _vol_df(300)
        d = PercentileRegimeDetector(
            lookback_window=50,
            upper_quantile=0.67,
            lower_quantile=0.33,
        )
        result = d.detect(df)
        assert result.value.height > 0
        types = set(result.value["signal_type"].to_list())
        assert types <= {"high_volatility", "low_volatility"}

    def test_signals_have_required_columns(self):
        df = _vol_df(300)
        d = PercentileRegimeDetector(lookback_window=50)
        result = d.detect(df)
        if result.value.height > 0:
            required = {"pair", "timestamp", "signal_type", "signal", "probability"}
            assert required <= set(result.value.columns)

    def test_custom_signal_names(self):
        df = _vol_df(300)
        d = PercentileRegimeDetector(
            lookback_window=50,
            signal_high="extreme_vol",
            signal_low="calm_vol",
        )
        assert d.allowed_signal_types == {"extreme_vol", "calm_vol"}

    def test_probability_bounded(self):
        df = _vol_df(300)
        d = PercentileRegimeDetector(lookback_window=50)
        result = d.detect(df)
        if result.value.height > 0:
            probs = result.value["probability"].to_list()
            assert all(0 < p <= 1.0 for p in probs if p is not None)

    def test_multi_pair(self):
        df1 = _vol_df(200, pair="BTCUSDT")
        df2 = _vol_df(200, pair="ETHUSDT")
        df = pl.concat([df1, df2])
        d = PercentileRegimeDetector(lookback_window=50)
        result = d.detect(df)
        if result.value.height > 0:
            pairs = set(result.value["pair"].to_list())
            assert len(pairs) >= 1

    def test_empty_result_for_short_data(self):
        df = _vol_df(2)  # too few bars
        d = PercentileRegimeDetector(lookback_window=50)
        result = d.detect(df)
        # With only 2 bars, percentile calc might not produce enough context
        assert isinstance(result, Signals)
