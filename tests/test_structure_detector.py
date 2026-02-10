"""Tests for StructureDetector (real-time)."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.core import Signals
from signalflow.core.enums import SignalCategory
from signalflow.detector.structure_detector import StructureDetector


def _sine_wave_df(n=500, pair="BTCUSDT", period=100, amplitude=10.0):
    """OHLCV with sine wave (clear tops/bottoms)."""
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
    """OHLCV with constant price."""
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


class TestStructureDetector:
    def test_signal_category(self):
        d = StructureDetector()
        assert d.signal_category == SignalCategory.PRICE_STRUCTURE

    def test_detect_returns_signals(self):
        df = _sine_wave_df(500)
        d = StructureDetector(lookback=30, confirmation_bars=5, min_swing_pct=0.01)
        result = d.detect(df)
        assert isinstance(result, Signals)

    def test_sine_wave_detects_structure(self):
        df = _sine_wave_df(500, period=100, amplitude=10.0)
        d = StructureDetector(lookback=40, confirmation_bars=5, min_swing_pct=0.01)
        result = d.detect(df)
        assert result.value.height > 0, "Sine wave should produce structure signals"
        types = set(result.value["signal_type"].to_list())
        assert types <= {"local_top", "local_bottom"}

    def test_flat_market_no_signals(self):
        df = _flat_df(300)
        d = StructureDetector(lookback=30, confirmation_bars=5, min_swing_pct=0.02)
        result = d.detect(df)
        assert result.value.height == 0, "Flat market should produce no structure signals"

    def test_signals_have_required_columns(self):
        df = _sine_wave_df(500)
        d = StructureDetector(lookback=30, confirmation_bars=5, min_swing_pct=0.01)
        result = d.detect(df)
        if result.value.height > 0:
            required = {"pair", "timestamp", "signal_type", "signal", "probability"}
            assert required <= set(result.value.columns)

    def test_probability_is_bounded(self):
        df = _sine_wave_df(500)
        d = StructureDetector(lookback=30, confirmation_bars=5, min_swing_pct=0.01)
        result = d.detect(df)
        if result.value.height > 0:
            probs = result.value["probability"].to_list()
            assert all(0.0 <= p <= 1.0 for p in probs if p is not None)

    def test_backward_looking_only(self):
        """Signals should not appear before lookback + confirmation_bars."""
        df = _sine_wave_df(500)
        lookback = 30
        conf = 5
        d = StructureDetector(lookback=lookback, confirmation_bars=conf, min_swing_pct=0.01)
        result = d.detect(df)
        if result.value.height > 0:
            earliest_ts = result.value["timestamp"].min()
            base_ts = datetime(2024, 1, 1)
            earliest_idx = int((earliest_ts - base_ts).total_seconds() / 60)
            assert earliest_idx >= lookback + conf

    def test_multi_pair(self):
        btc = _sine_wave_df(300, pair="BTCUSDT")
        eth = _sine_wave_df(300, pair="ETHUSDT")
        df = pl.concat([btc, eth])
        d = StructureDetector(lookback=30, confirmation_bars=5, min_swing_pct=0.01)
        result = d.detect(df)
        assert isinstance(result, Signals)
        if result.value.height > 0:
            pairs = set(result.value["pair"].to_list())
            assert len(pairs) >= 1
