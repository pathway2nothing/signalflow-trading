"""Tests for SignalPairPrice analytic."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.core import RawData, Signals
from signalflow.analytic.signals.signals_price import SignalPairPrice


TS = datetime(2024, 1, 1)


def _make_price_df(n=50, pair="BTCUSDT"):
    rows = [
        {
            "pair": pair,
            "timestamp": TS + timedelta(minutes=i),
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1000.0,
        }
        for i in range(n)
    ]
    return pl.DataFrame(rows)


def _make_signals(pairs=None, n=10):
    if pairs is None:
        pairs = ["BTCUSDT"]
    rows = []
    for pair in pairs:
        for i in range(n):
            signal = 1 if i % 3 == 0 else (-1 if i % 3 == 1 else 0)
            rows.append(
                {
                    "pair": pair,
                    "timestamp": TS + timedelta(minutes=i * 5),
                    "signal": signal,
                    "signal_type": "rise" if signal == 1 else ("fall" if signal == -1 else "none"),
                }
            )
    return Signals(pl.DataFrame(rows))


def _make_raw_data(n=50, pairs=None):
    if pairs is None:
        pairs = ["BTCUSDT"]
    dfs = [_make_price_df(n, pair) for pair in pairs]
    df = pl.concat(dfs)
    return RawData(
        datetime_start=TS,
        datetime_end=TS + timedelta(minutes=n),
        pairs=pairs,
        data={"spot": df},
    )


class TestSignalPairPriceCompute:
    def test_basic_compute(self):
        metric = SignalPairPrice()
        raw = _make_raw_data()
        signals = _make_signals()
        result, _ = metric.compute(raw, signals)
        assert "BTCUSDT" in result
        assert "buy_signals" in result["BTCUSDT"]
        assert "sell_signals" in result["BTCUSDT"]
        assert "total_signals" in result["BTCUSDT"]

    def test_specific_pairs(self):
        metric = SignalPairPrice(pairs=["BTCUSDT"])
        raw = _make_raw_data(pairs=["BTCUSDT", "ETHUSDT"])
        signals = _make_signals(pairs=["BTCUSDT", "ETHUSDT"])
        result, _ = metric.compute(raw, signals)
        assert "BTCUSDT" in result
        assert "ETHUSDT" not in result

    def test_single_pair_string(self):
        metric = SignalPairPrice(pairs="BTCUSDT")
        assert metric.pairs == ["BTCUSDT"]

    def test_signal_counts_correct(self):
        metric = SignalPairPrice()
        raw = _make_raw_data()
        signals = _make_signals(n=9)
        result, _ = metric.compute(raw, signals)
        counts = result["BTCUSDT"]
        assert counts["buy_signals"] == 3  # i=0,3,6
        assert counts["sell_signals"] == 3  # i=1,4,7
        assert counts["neutral_signals"] == 3  # i=2,5,8

    def test_all_pairs_if_none(self):
        metric = SignalPairPrice(pairs=None)
        raw = _make_raw_data(pairs=["BTCUSDT", "ETHUSDT"])
        signals = _make_signals(pairs=["BTCUSDT", "ETHUSDT"])
        result, _ = metric.compute(raw, signals)
        assert len(result) == 2


class TestSignalPairPricePlot:
    def test_plot_returns_figures(self):
        metric = SignalPairPrice()
        raw = _make_raw_data()
        signals = _make_signals()
        computed, ctx = metric.compute(raw, signals)
        figs = metric.plot(computed, ctx, raw, signals)
        assert len(figs) == 1  # one figure per pair

    def test_plot_multi_pair(self):
        metric = SignalPairPrice()
        raw = _make_raw_data(pairs=["BTCUSDT", "ETHUSDT"])
        signals = _make_signals(pairs=["BTCUSDT", "ETHUSDT"])
        computed, ctx = metric.compute(raw, signals)
        figs = metric.plot(computed, ctx, raw, signals)
        assert len(figs) == 2

    def test_no_price_data_raises(self):
        metric = SignalPairPrice()
        raw = RawData(
            datetime_start=TS,
            datetime_end=TS + timedelta(minutes=10),
            pairs=["BTCUSDT"],
            data={},
        )
        signals = _make_signals()
        computed, ctx = metric.compute(raw, signals)
        with pytest.raises(ValueError, match="No price data"):
            metric.plot(computed, ctx, raw, signals)
