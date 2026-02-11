"""Tests for SignalDistributionMetric."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.core import RawData, Signals
from signalflow.analytic.signals.distribution_metrics import SignalDistributionMetric


TS = datetime(2024, 1, 1)


def _make_price_df(n=100, pair="BTCUSDT"):
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


def _make_raw_data(n=100, pairs=None):
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


def _make_signals(pairs=None, signals_per_pair=None):
    """Create signals with specified count per pair."""
    if pairs is None:
        pairs = ["BTCUSDT"]
    if signals_per_pair is None:
        signals_per_pair = {p: 10 for p in pairs}

    rows = []
    for pair in pairs:
        n = signals_per_pair.get(pair, 10)
        for i in range(n):
            signal = 1 if i % 2 == 0 else -1
            rows.append(
                {
                    "pair": pair,
                    "timestamp": TS + timedelta(minutes=i),
                    "signal": signal,
                    "signal_type": "rise" if signal == 1 else "fall",
                }
            )
    return Signals(pl.DataFrame(rows))


def _make_uniform_signals(pairs, n_per_pair=10):
    """Create signals with uniform count across pairs."""
    return _make_signals(pairs=pairs, signals_per_pair={p: n_per_pair for p in pairs})


def _make_varied_signals(pairs):
    """Create signals with varied counts across pairs."""
    counts = {pairs[i]: (i + 1) * 5 for i in range(len(pairs))}
    return _make_signals(pairs=pairs, signals_per_pair=counts)


class TestSignalDistributionMetricCompute:
    def test_basic_compute(self):
        metric = SignalDistributionMetric()
        raw = _make_raw_data()
        signals = _make_signals()
        result, ctx = metric.compute(raw, signals)

        assert result is not None
        assert "quant" in result
        assert "series" in result

    def test_quant_metrics_present(self):
        metric = SignalDistributionMetric()
        raw = _make_raw_data(pairs=["BTCUSDT", "ETHUSDT"])
        signals = _make_signals(pairs=["BTCUSDT", "ETHUSDT"])
        result, ctx = metric.compute(raw, signals)

        quant = result["quant"]
        assert "mean_signals_per_pair" in quant
        assert "median_signals_per_pair" in quant
        assert "min_signals_per_pair" in quant
        assert "max_signals_per_pair" in quant
        assert "total_pairs" in quant

    def test_no_signals_returns_none(self):
        metric = SignalDistributionMetric()
        raw = _make_raw_data()
        # All zero signals
        signals_df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"] * 10,
                "timestamp": [TS + timedelta(minutes=i) for i in range(10)],
                "signal": [0] * 10,
                "signal_type": ["none"] * 10,
            }
        )
        signals = Signals(signals_df)
        result, ctx = metric.compute(raw, signals)
        assert result is None

    def test_single_pair_no_histogram(self):
        metric = SignalDistributionMetric()
        raw = _make_raw_data(pairs=["BTCUSDT"])
        signals = _make_signals(pairs=["BTCUSDT"])
        result, ctx = metric.compute(raw, signals)

        assert result is not None
        assert ctx["use_histogram"] is False

    def test_many_pairs_uses_histogram(self):
        pairs = [f"PAIR{i}" for i in range(20)]
        metric = SignalDistributionMetric()
        raw = _make_raw_data(n=200, pairs=pairs)
        signals = _make_varied_signals(pairs=pairs)
        result, ctx = metric.compute(raw, signals)

        assert result is not None
        assert ctx["use_histogram"] is True

    def test_uniform_distribution(self):
        pairs = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        metric = SignalDistributionMetric()
        raw = _make_raw_data(pairs=pairs)
        signals = _make_uniform_signals(pairs=pairs, n_per_pair=10)
        result, ctx = metric.compute(raw, signals)

        quant = result["quant"]
        assert quant["mean_signals_per_pair"] == pytest.approx(10.0)
        assert quant["min_signals_per_pair"] == 10
        assert quant["max_signals_per_pair"] == 10

    def test_varied_distribution(self):
        pairs = ["A", "B", "C", "D"]
        counts = {"A": 5, "B": 10, "C": 15, "D": 20}
        metric = SignalDistributionMetric()
        raw = _make_raw_data(n=100, pairs=pairs)
        signals = _make_signals(pairs=pairs, signals_per_pair=counts)
        result, ctx = metric.compute(raw, signals)

        quant = result["quant"]
        assert quant["min_signals_per_pair"] == 5
        assert quant["max_signals_per_pair"] == 20
        assert quant["total_pairs"] == 4

    def test_rolling_signals_computed(self):
        metric = SignalDistributionMetric(rolling_window_minutes=10)
        raw = _make_raw_data(n=100)
        signals = _make_signals(signals_per_pair={"BTCUSDT": 50})
        result, ctx = metric.compute(raw, signals)

        assert "signals_rolling" in result["series"]
        rolling_df = result["series"]["signals_rolling"]
        assert "rolling_sum" in rolling_df.columns

    def test_signals_per_pair_sorted(self):
        pairs = ["A", "B", "C"]
        counts = {"A": 5, "B": 20, "C": 10}
        metric = SignalDistributionMetric()
        raw = _make_raw_data(n=100, pairs=pairs)
        signals = _make_signals(pairs=pairs, signals_per_pair=counts)
        result, ctx = metric.compute(raw, signals)

        signals_per_pair = result["series"]["signals_per_pair"]
        pair_order = signals_per_pair["pair"].to_list()
        # Should be sorted descending by count
        assert pair_order[0] == "B"  # 20 signals
        assert pair_order[-1] == "A"  # 5 signals

    def test_custom_n_bars(self):
        pairs = [f"P{i}" for i in range(30)]
        metric = SignalDistributionMetric(n_bars=5)
        raw = _make_raw_data(n=200, pairs=pairs)
        signals = _make_varied_signals(pairs=pairs)
        result, ctx = metric.compute(raw, signals)

        # Number of grouped bins should be <= n_bars
        grouped = result["series"]["grouped"]
        assert len(grouped) <= 5

    def test_plots_context_values(self):
        metric = SignalDistributionMetric(rolling_window_minutes=30, ma_window_hours=6)
        raw = _make_raw_data()
        signals = _make_signals()
        result, ctx = metric.compute(raw, signals)

        assert ctx["rolling_window"] == 30
        assert ctx["ma_window"] == 6


class TestSignalDistributionMetricPlot:
    def test_plot_returns_figure(self):
        metric = SignalDistributionMetric()
        raw = _make_raw_data(pairs=["BTCUSDT", "ETHUSDT"])
        signals = _make_signals(pairs=["BTCUSDT", "ETHUSDT"])
        computed, ctx = metric.compute(raw, signals)
        fig = metric.plot(computed, ctx, raw, signals)

        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_none_metrics(self):
        metric = SignalDistributionMetric()
        raw = _make_raw_data()
        signals = _make_signals()
        fig = metric.plot(None, {}, raw, signals)
        assert fig is None

    def test_plot_has_three_subplots(self):
        pairs = [f"P{i}" for i in range(20)]
        metric = SignalDistributionMetric()
        raw = _make_raw_data(n=200, pairs=pairs)
        signals = _make_varied_signals(pairs=pairs)
        computed, ctx = metric.compute(raw, signals)
        fig = metric.plot(computed, ctx, raw, signals)

        # Should have traces for histogram, sorted signals, and rolling
        assert len(fig.data) >= 3

    def test_plot_dimensions(self):
        metric = SignalDistributionMetric(chart_height=1000, chart_width=1200)
        raw = _make_raw_data()
        signals = _make_signals()
        computed, ctx = metric.compute(raw, signals)
        fig = metric.plot(computed, ctx, raw, signals)

        assert fig.layout.height == 1000
        assert fig.layout.width == 1200

    def test_bar_chart_for_few_pairs(self):
        pairs = ["A", "B", "C"]
        metric = SignalDistributionMetric()
        raw = _make_raw_data(pairs=pairs)
        signals = _make_signals(pairs=pairs)
        computed, ctx = metric.compute(raw, signals)
        fig = metric.plot(computed, ctx, raw, signals)

        # First trace should be a bar chart
        assert fig.data[0].type == "bar"

    def test_histogram_for_many_pairs(self):
        pairs = [f"P{i}" for i in range(25)]
        metric = SignalDistributionMetric()
        raw = _make_raw_data(n=200, pairs=pairs)
        signals = _make_varied_signals(pairs=pairs)
        computed, ctx = metric.compute(raw, signals)
        fig = metric.plot(computed, ctx, raw, signals)

        # Should still have bar chart but with binned data
        assert fig.data[0].type == "bar"
