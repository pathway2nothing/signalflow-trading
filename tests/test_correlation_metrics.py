"""Tests for analytic.signals.correlation_metrics."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from signalflow.analytic.signals.correlation_metrics import (
    SignalCorrelationMetric,
    SignalTimingMetric,
)
from signalflow.core import RawData, Signals

TS = datetime(2024, 1, 1)


def _make_price_df(pair="BTCUSDT", n=2000, base_price=100.0, trend=0.0001):
    """Create price DataFrame with optional trend."""
    timestamps = [TS + timedelta(minutes=i) for i in range(n)]
    prices = [base_price * (1 + trend * i + np.random.randn() * 0.001) for i in range(n)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "pair": [pair] * n,
            "close": prices,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "volume": [1000.0] * n,
        }
    )


def _make_signals_df(pair="BTCUSDT", n=50, signal_interval=30):
    """Create signals DataFrame."""
    timestamps = [TS + timedelta(minutes=i * signal_interval) for i in range(n)]
    signals = [1 if i % 2 == 0 else -1 for i in range(n)]
    strengths = [0.5 + np.random.rand() * 0.5 for _ in range(n)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "pair": [pair] * n,
            "signal": signals,
            "strength": strengths,
        }
    )


def _make_raw_data(pairs=None, n=2000):
    """Create RawData with price data."""
    if pairs is None:
        pairs = ["BTCUSDT"]

    dfs = [_make_price_df(pair=p, n=n) for p in pairs]
    combined = pl.concat(dfs)
    return RawData(
        datetime_start=TS,
        datetime_end=TS + timedelta(minutes=n),
        pairs=pairs,
        data={"spot": combined},
    )


def _make_signals(pairs=None, n_per_pair=50):
    """Create Signals object."""
    if pairs is None:
        pairs = ["BTCUSDT"]

    dfs = [_make_signals_df(pair=p, n=n_per_pair) for p in pairs]
    combined = pl.concat(dfs)
    return Signals(combined)


# ── SignalCorrelationMetric ─────────────────────────────────────────────


class TestSignalCorrelationMetricCompute:
    def test_basic_compute(self):
        metric = SignalCorrelationMetric(look_ahead_periods=[15, 60])
        raw = _make_raw_data()
        signals = _make_signals()

        result, ctx = metric.compute(raw, signals)

        assert result is not None
        assert "quant" in result
        assert "correlations" in result["quant"]
        assert "total_signals" in result["quant"]

    def test_no_signals(self):
        metric = SignalCorrelationMetric()
        raw = _make_raw_data()
        signals = Signals(
            pl.DataFrame(
                {
                    "timestamp": [TS],
                    "pair": ["BTCUSDT"],
                    "signal": [0],
                }
            )
        )

        result, ctx = metric.compute(raw, signals)
        assert result is None

    def test_correlation_periods(self):
        metric = SignalCorrelationMetric(look_ahead_periods=[15, 60, 120])
        raw = _make_raw_data()
        signals = _make_signals()

        result, ctx = metric.compute(raw, signals)

        assert result is not None
        correlations = result["quant"]["correlations"]
        # Should have correlations for each period
        for period in [15, 60, 120]:
            key = f"period_{period}"
            if key in correlations:
                assert "pearson_corr" in correlations[key]
                assert "spearman_corr" in correlations[key]
                assert "n_samples" in correlations[key]

    def test_quintile_analysis(self):
        metric = SignalCorrelationMetric(look_ahead_periods=[15])
        raw = _make_raw_data()
        signals = _make_signals(n_per_pair=100)

        result, ctx = metric.compute(raw, signals)

        assert result is not None
        quintile_data = result["quant"]["quintile_analysis"]
        # Should have quintile analysis if enough data
        if quintile_data:
            assert "Q1 (Weakest)" in quintile_data or len(quintile_data) > 0


class TestSignalCorrelationMetricPlot:
    def test_plot_generation(self):
        metric = SignalCorrelationMetric()
        raw = _make_raw_data()
        signals = _make_signals()

        result, ctx = metric.compute(raw, signals)

        if result is not None:
            fig = metric.plot(result, ctx, raw, signals)
            assert fig is not None

    def test_plot_none_metrics(self):
        metric = SignalCorrelationMetric()
        fig = metric.plot(None, {}, None, None)
        assert fig is None


# ── SignalTimingMetric ──────────────────────────────────────────────────


class TestSignalTimingMetricCompute:
    def test_basic_compute(self):
        metric = SignalTimingMetric(max_look_ahead=120, sample_points=12)
        raw = _make_raw_data()
        signals = _make_signals()

        result, ctx = metric.compute(raw, signals)

        assert result is not None
        assert "quant" in result
        assert "optimal_hold_time_mean" in result["quant"]
        assert "optimal_hold_time_sharpe" in result["quant"]
        assert "peak_mean_return" in result["quant"]

    def test_series_data(self):
        metric = SignalTimingMetric(max_look_ahead=60, sample_points=6)
        raw = _make_raw_data()
        signals = _make_signals()

        result, ctx = metric.compute(raw, signals)

        assert result is not None
        assert "series" in result
        series = result["series"]
        assert "time_points" in series
        assert "mean_returns" in series
        assert "sharpe_at_time" in series
        assert "win_rate_at_time" in series
        assert len(series["time_points"]) == 6

    def test_no_signals(self):
        metric = SignalTimingMetric()
        raw = _make_raw_data()
        signals = Signals(
            pl.DataFrame(
                {
                    "timestamp": [TS],
                    "pair": ["BTCUSDT"],
                    "signal": [0],
                }
            )
        )

        result, ctx = metric.compute(raw, signals)
        assert result is None


class TestSignalTimingMetricPlot:
    def test_plot_generation(self):
        metric = SignalTimingMetric(max_look_ahead=60, sample_points=6)
        raw = _make_raw_data()
        signals = _make_signals()

        result, ctx = metric.compute(raw, signals)

        if result is not None:
            fig = metric.plot(result, ctx, raw, signals)
            assert fig is not None

    def test_plot_none_metrics(self):
        metric = SignalTimingMetric()
        fig = metric.plot(None, {}, None, None)
        assert fig is None
