"""Tests for SignalProfileMetric."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.analytic.signals.profile_metrics import SignalProfileMetric
from signalflow.core import RawData, Signals

TS = datetime(2024, 1, 1)


def _make_price_df(n=2000, pair="BTCUSDT", trend=0.001):
    """Create price data with slight upward trend."""
    rows = []
    price = 100.0
    for i in range(n):
        price = price * (1 + trend + np.random.normal(0, 0.005))
        rows.append(
            {
                "pair": pair,
                "timestamp": TS + timedelta(minutes=i),
                "open": price - 0.5,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 1000.0,
            }
        )
    return pl.DataFrame(rows)


def _make_raw_data(n=2000, pairs=None):
    if pairs is None:
        pairs = ["BTCUSDT"]
    np.random.seed(42)
    dfs = [_make_price_df(n, pair) for pair in pairs]
    df = pl.concat(dfs)
    return RawData(
        datetime_start=TS,
        datetime_end=TS + timedelta(minutes=n),
        pairs=pairs,
        data={"spot": df},
    )


def _make_signals(pairs=None, n_signals=5, interval_minutes=100):
    """Create buy signals spread across time."""
    if pairs is None:
        pairs = ["BTCUSDT"]
    rows = []
    for pair in pairs:
        for i in range(n_signals):
            rows.append(
                {
                    "pair": pair,
                    "timestamp": TS + timedelta(minutes=i * interval_minutes),
                    "signal": 1,  # buy signal
                    "signal_type": "rise",
                }
            )
    return Signals(pl.DataFrame(rows))


def _make_mixed_signals(pair="BTCUSDT", n=10):
    """Create mixed buy/sell/neutral signals."""
    rows = []
    for i in range(n):
        signal = 1 if i % 3 == 0 else (-1 if i % 3 == 1 else 0)
        rows.append(
            {
                "pair": pair,
                "timestamp": TS + timedelta(minutes=i * 100),
                "signal": signal,
                "signal_type": "rise" if signal == 1 else ("fall" if signal == -1 else "none"),
            }
        )
    return Signals(pl.DataFrame(rows))


class TestSignalProfileMetricCompute:
    def test_basic_compute(self):
        metric = SignalProfileMetric(look_ahead=100)
        raw = _make_raw_data(n=1000)
        signals = _make_signals(n_signals=5, interval_minutes=100)
        result, _ctx = metric.compute(raw, signals)

        assert result is not None
        assert "quant" in result
        assert "series" in result

    def test_quant_metrics_present(self):
        metric = SignalProfileMetric(look_ahead=100)
        raw = _make_raw_data(n=1000)
        signals = _make_signals(n_signals=5, interval_minutes=100)
        result, _ctx = metric.compute(raw, signals)

        quant = result["quant"]
        assert "n_signals" in quant
        assert "final_mean" in quant
        assert "final_median" in quant
        assert "avg_max_uplift" in quant
        assert "max_mean_pct" in quant
        assert "max_mean_idx" in quant

    def test_series_profiles_present(self):
        metric = SignalProfileMetric(look_ahead=100)
        raw = _make_raw_data(n=1000)
        signals = _make_signals(n_signals=5, interval_minutes=100)
        result, _ctx = metric.compute(raw, signals)

        series = result["series"]
        assert "mean_profile" in series
        assert "std_profile" in series
        assert "median_profile" in series
        assert "lower_quant" in series
        assert "upper_quant" in series
        assert "cummax_mean" in series
        assert "cummin_mean" in series

    def test_no_buy_signals_returns_none(self):
        metric = SignalProfileMetric(look_ahead=100)
        raw = _make_raw_data(n=500)
        # Only sell signals
        signals_df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"] * 5,
                "timestamp": [TS + timedelta(minutes=i * 50) for i in range(5)],
                "signal": [-1] * 5,
                "signal_type": ["fall"] * 5,
            }
        )
        signals = Signals(signals_df)
        result, _ctx = metric.compute(raw, signals)
        assert result is None

    def test_insufficient_future_data_returns_none(self):
        metric = SignalProfileMetric(look_ahead=1000)
        raw = _make_raw_data(n=100)  # Not enough data for look_ahead
        signals = _make_signals(n_signals=1, interval_minutes=0)
        result, _ctx = metric.compute(raw, signals)
        assert result is None

    def test_signal_count_correct(self):
        metric = SignalProfileMetric(look_ahead=100)
        raw = _make_raw_data(n=1500)
        signals = _make_signals(n_signals=10, interval_minutes=100)
        result, _ctx = metric.compute(raw, signals)

        # Some signals may not have enough future data
        assert result["quant"]["n_signals"] <= 10
        assert result["quant"]["n_signals"] > 0

    def test_profile_length_matches_look_ahead(self):
        look_ahead = 50
        metric = SignalProfileMetric(look_ahead=look_ahead)
        raw = _make_raw_data(n=500)
        signals = _make_signals(n_signals=3, interval_minutes=100)
        result, _ctx = metric.compute(raw, signals)

        mean_profile = result["series"]["mean_profile"]
        assert len(mean_profile) == look_ahead + 1

    def test_custom_quantiles(self):
        metric = SignalProfileMetric(look_ahead=100, quantiles=(0.1, 0.9))
        raw = _make_raw_data(n=1000)
        signals = _make_signals(n_signals=5, interval_minutes=100)
        result, _ctx = metric.compute(raw, signals)

        # Should still have quantile series
        assert "lower_quant" in result["series"]
        assert "upper_quant" in result["series"]

    def test_multi_pair(self):
        metric = SignalProfileMetric(look_ahead=100)
        raw = _make_raw_data(n=1000, pairs=["BTCUSDT", "ETHUSDT"])
        signals = _make_signals(pairs=["BTCUSDT", "ETHUSDT"], n_signals=3, interval_minutes=100)
        result, ctx = metric.compute(raw, signals)

        assert result is not None
        assert ctx["pairs_analyzed"] == 2

    def test_futures_data_source(self):
        metric = SignalProfileMetric(look_ahead=100)
        np.random.seed(42)
        price_df = _make_price_df(n=1000)
        raw = RawData(
            datetime_start=TS,
            datetime_end=TS + timedelta(minutes=1000),
            pairs=["BTCUSDT"],
            data={"futures": price_df},  # using futures instead of spot
        )
        signals = _make_signals(n_signals=5, interval_minutes=100)
        result, _ctx = metric.compute(raw, signals)

        assert result is not None

    def test_no_price_data_raises(self):
        metric = SignalProfileMetric(look_ahead=100)
        raw = RawData(
            datetime_start=TS,
            datetime_end=TS + timedelta(minutes=100),
            pairs=["BTCUSDT"],
            data={},  # No price data
        )
        signals = _make_signals(n_signals=5)

        with pytest.raises(ValueError, match="No price data"):
            metric.compute(raw, signals)


class TestSignalProfileMetricPlot:
    def test_plot_returns_figure(self):
        metric = SignalProfileMetric(look_ahead=100)
        raw = _make_raw_data(n=1000)
        signals = _make_signals(n_signals=5, interval_minutes=100)
        computed, ctx = metric.compute(raw, signals)
        fig = metric.plot(computed, ctx, raw, signals)

        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_none_metrics(self):
        metric = SignalProfileMetric()
        raw = _make_raw_data()
        signals = _make_signals()
        fig = metric.plot(None, {}, raw, signals)
        assert fig is None

    def test_plot_has_subplots(self):
        metric = SignalProfileMetric(look_ahead=100)
        raw = _make_raw_data(n=1000)
        signals = _make_signals(n_signals=5, interval_minutes=100)
        computed, ctx = metric.compute(raw, signals)
        fig = metric.plot(computed, ctx, raw, signals)

        # Should have traces for both subplots
        # Mean, std bands, median, percentiles, cummax, cummin, etc.
        assert len(fig.data) >= 5

    def test_plot_dimensions(self):
        metric = SignalProfileMetric(look_ahead=100, chart_height=700, chart_width=1000)
        raw = _make_raw_data(n=1000)
        signals = _make_signals(n_signals=5, interval_minutes=100)
        computed, ctx = metric.compute(raw, signals)
        fig = metric.plot(computed, ctx, raw, signals)

        assert fig.layout.height == 700
        assert fig.layout.width == 1000
