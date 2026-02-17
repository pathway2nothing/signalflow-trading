"""Tests for signalflow.analytic.base module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import polars as pl
import pytest

from signalflow.analytic.base import SignalMetric, StrategyMetric
from signalflow.core import RawData, SfComponentType, Signals, StrategyState


class TestSignalMetric:
    """Tests for SignalMetric base class."""

    def test_component_type(self):
        """SignalMetric has correct component_type."""
        metric = SignalMetric()
        assert metric.component_type == SfComponentType.SIGNAL_METRIC

    def test_compute_default_returns_empty(self, sample_raw_data, sample_signals):
        """Default compute() returns empty dicts and logs warning."""
        metric = SignalMetric()
        result, context = metric.compute(sample_raw_data, sample_signals)
        assert result == {}
        assert context == {}

    def test_compute_with_labels(self, sample_raw_data, sample_signals):
        """compute() accepts optional labels parameter."""
        metric = SignalMetric()
        labels = pl.DataFrame({"label": [1, 0, 1]})
        result, context = metric.compute(sample_raw_data, sample_signals, labels=labels)
        assert result == {}
        assert context == {}

    def test_plot_default_returns_none(self, sample_raw_data, sample_signals):
        """Default plot() returns None and logs warning."""
        metric = SignalMetric()
        result = metric.plot(
            computed_metrics={},
            plots_context={},
            raw_data=sample_raw_data,
            signals=sample_signals,
        )
        assert result is None

    def test_plot_with_labels(self, sample_raw_data, sample_signals):
        """plot() accepts optional labels parameter."""
        metric = SignalMetric()
        labels = pl.DataFrame({"label": [1, 0, 1]})
        result = metric.plot(
            computed_metrics={},
            plots_context={},
            raw_data=sample_raw_data,
            signals=sample_signals,
            labels=labels,
        )
        assert result is None

    def test_call_invokes_compute_and_plot(self, sample_raw_data, sample_signals):
        """__call__ invokes compute() then plot()."""
        metric = SignalMetric()
        computed_metrics, plots = metric(sample_raw_data, sample_signals)
        assert computed_metrics == {}
        assert plots is None

    def test_call_with_labels(self, sample_raw_data, sample_signals):
        """__call__ passes labels to compute and plot."""
        metric = SignalMetric()
        labels = pl.DataFrame({"label": [1, 0, 1]})
        computed_metrics, plots = metric(sample_raw_data, sample_signals, labels=labels)
        assert computed_metrics == {}


class TestStrategyMetric:
    """Tests for StrategyMetric base class."""

    def test_component_type(self):
        """StrategyMetric has correct component_type."""
        # StrategyMetric is abstract but we can check class var
        assert StrategyMetric.component_type == SfComponentType.STRATEGY_METRIC

    def test_compute_default_returns_empty(self):
        """Default compute() returns empty dict and logs warning."""

        # Create concrete subclass for testing
        class ConcreteStrategyMetric(StrategyMetric):
            pass

        metric = ConcreteStrategyMetric()
        state = MagicMock(spec=StrategyState)
        prices = {"BTCUSDT": 50000.0}

        result = metric.compute(state, prices)
        assert result == {}

    def test_compute_accepts_kwargs(self):
        """compute() accepts additional kwargs."""

        class ConcreteStrategyMetric(StrategyMetric):
            pass

        metric = ConcreteStrategyMetric()
        state = MagicMock(spec=StrategyState)
        prices = {"BTCUSDT": 50000.0}

        result = metric.compute(state, prices, extra_param="value")
        assert result == {}

    def test_plot_default_returns_none(self):
        """Default plot() returns None and logs warning."""

        class ConcreteStrategyMetric(StrategyMetric):
            pass

        metric = ConcreteStrategyMetric()
        result = metric.plot(results={})
        assert result is None

    def test_plot_with_state_and_raw_data(self, sample_raw_data):
        """plot() accepts optional state and raw_data."""

        class ConcreteStrategyMetric(StrategyMetric):
            pass

        metric = ConcreteStrategyMetric()
        state = MagicMock(spec=StrategyState)

        result = metric.plot(
            results={"sharpe": 1.5},
            state=state,
            raw_data=sample_raw_data,
        )
        assert result is None

    def test_plot_accepts_kwargs(self):
        """plot() accepts additional kwargs."""

        class ConcreteStrategyMetric(StrategyMetric):
            pass

        metric = ConcreteStrategyMetric()
        result = metric.plot(results={}, extra_param="value")
        assert result is None


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_raw_data():
    """Create sample RawData for tests."""
    base = datetime(2024, 1, 1)
    df = pl.DataFrame(
        {
            "pair": ["BTCUSDT"] * 10,
            "timestamp": [base + timedelta(hours=i) for i in range(10)],
            "open": [100.0 + i for i in range(10)],
            "high": [105.0 + i for i in range(10)],
            "low": [95.0 + i for i in range(10)],
            "close": [102.0 + i for i in range(10)],
            "volume": [1000.0] * 10,
        }
    )
    return RawData(
        datetime_start=base,
        datetime_end=base + timedelta(hours=10),
        pairs=["BTCUSDT"],
        data={"spot": df},
    )


@pytest.fixture
def sample_signals():
    """Create sample Signals for tests."""
    base = datetime(2024, 1, 1)
    df = pl.DataFrame(
        {
            "pair": ["BTCUSDT", "BTCUSDT"],
            "timestamp": [base + timedelta(hours=2), base + timedelta(hours=5)],
            "signal": [1, -1],
        }
    )
    return Signals(df)
