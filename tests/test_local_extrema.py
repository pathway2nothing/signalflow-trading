"""Tests for LocalExtremaDetector."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.core import Signals
from signalflow.core.enums import SignalCategory
from signalflow.detector.local_extrema import LocalExtremaDetector


def _sine_wave_df(n=500, pair="BTCUSDT", period=100, amplitude=10.0):
    base_ts = datetime(2024, 1, 1)
    timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
    prices = [100.0 + amplitude * np.sin(2 * np.pi * i / period) for i in range(n)]
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": timestamps,
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }
    )


def _flat_df(n=300, pair="BTCUSDT"):
    base_ts = datetime(2024, 1, 1)
    timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": timestamps,
            "open": [100.0] * n,
            "high": [100.1] * n,
            "low": [99.9] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        }
    )


class TestLocalExtremaDetector:
    def test_signal_category(self):
        d = LocalExtremaDetector()
        assert d.signal_category == SignalCategory.PRICE_STRUCTURE

    def test_post_init_validation(self):
        with pytest.raises(ValueError, match="confirmation_bars"):
            LocalExtremaDetector(confirmation_bars=60, lookback=60)

    def test_detect_returns_signals(self):
        df = _sine_wave_df(500)
        d = LocalExtremaDetector(lookback=30, confirmation_bars=5, min_swing_pct=0.01)
        result = d.detect(df)
        assert isinstance(result, Signals)

    def test_sine_wave_detects_extrema(self):
        df = _sine_wave_df(500, period=100, amplitude=10.0)
        d = LocalExtremaDetector(lookback=40, confirmation_bars=5, min_swing_pct=0.01)
        result = d.detect(df)
        assert result.value.height > 0
        types = set(result.value["signal_type"].to_list())
        assert types <= {"local_max", "local_min"}

    def test_flat_market_no_signals(self):
        df = _flat_df(300)
        d = LocalExtremaDetector(lookback=30, confirmation_bars=5, min_swing_pct=0.02)
        result = d.detect(df)
        assert result.value.height == 0

    def test_signals_have_required_columns(self):
        df = _sine_wave_df(500)
        d = LocalExtremaDetector(lookback=30, confirmation_bars=5, min_swing_pct=0.01)
        result = d.detect(df)
        if result.value.height > 0:
            required = {"pair", "timestamp", "signal_type", "signal", "probability"}
            assert required <= set(result.value.columns)

    def test_custom_signal_names(self):
        df = _sine_wave_df(500, period=100, amplitude=10.0)
        d = LocalExtremaDetector(
            lookback=40,
            confirmation_bars=5,
            min_swing_pct=0.01,
            signal_top="my_top",
            signal_bottom="my_bottom",
        )
        assert d.allowed_signal_types == {"my_top", "my_bottom"}
        result = d.detect(df)
        if result.value.height > 0:
            types = set(result.value["signal_type"].to_list())
            assert types <= {"my_top", "my_bottom"}

    def test_multi_pair(self):
        df1 = _sine_wave_df(300, pair="BTCUSDT")
        df2 = _sine_wave_df(300, pair="ETHUSDT")
        df = pl.concat([df1, df2])
        d = LocalExtremaDetector(lookback=30, confirmation_bars=5, min_swing_pct=0.01)
        result = d.detect(df)
        if result.value.height > 0:
            pairs = set(result.value["pair"].to_list())
            assert len(pairs) >= 1

    def test_probability_bounded(self):
        df = _sine_wave_df(500, amplitude=10.0)
        d = LocalExtremaDetector(lookback=40, confirmation_bars=5, min_swing_pct=0.01)
        result = d.detect(df)
        if result.value.height > 0:
            probs = result.value["probability"].to_list()
            assert all(0 < p <= 1.0 for p in probs if p is not None)

    def test_empty_result_for_no_data(self):
        df = _sine_wave_df(5)  # too few bars
        d = LocalExtremaDetector(lookback=30, confirmation_bars=5)
        result = d.detect(df)
        assert result.value.height == 0
