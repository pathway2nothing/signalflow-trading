"""Tests for SignalClassificationMetric."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.analytic.signals.classification_metrics import SignalClassificationMetric
from signalflow.core import RawData, Signals

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


def _make_signals_and_labels(n=20, pair="BTCUSDT", accuracy=0.7):
    """Create signals and labels with specified accuracy."""
    rows_signals = []
    rows_labels = []

    for i in range(n):
        ts = TS + timedelta(minutes=i * 2)
        # Alternating true labels
        true_label = 1 if i % 2 == 0 else 0

        # Prediction matches true label with given accuracy
        pred = (1 if true_label == 1 else -1) if np.random.random() < accuracy else -1 if true_label == 1 else 1

        rows_signals.append(
            {
                "pair": pair,
                "timestamp": ts,
                "signal": pred,
                "signal_type": "rise" if pred == 1 else "fall",
            }
        )
        rows_labels.append(
            {
                "pair": pair,
                "timestamp": ts,
                "label": true_label,
            }
        )

    signals = Signals(pl.DataFrame(rows_signals))
    labels = pl.DataFrame(rows_labels)
    return signals, labels


def _make_perfect_signals_and_labels(n=20, pair="BTCUSDT"):
    """Create perfectly matching signals and labels."""
    rows_signals = []
    rows_labels = []

    for i in range(n):
        ts = TS + timedelta(minutes=i * 2)
        true_label = 1 if i % 2 == 0 else 0
        pred = 1 if true_label == 1 else -1

        rows_signals.append(
            {
                "pair": pair,
                "timestamp": ts,
                "signal": pred,
                "signal_type": "rise" if pred == 1 else "fall",
            }
        )
        rows_labels.append(
            {
                "pair": pair,
                "timestamp": ts,
                "label": true_label,
            }
        )

    signals = Signals(pl.DataFrame(rows_signals))
    labels = pl.DataFrame(rows_labels)
    return signals, labels


def _make_string_labels(n=20, pair="BTCUSDT"):
    """Create signals with string labels."""
    rows_signals = []
    rows_labels = []

    for i in range(n):
        ts = TS + timedelta(minutes=i * 2)
        true_label = "rise" if i % 2 == 0 else "fall"
        pred = 1 if i % 2 == 0 else -1

        rows_signals.append(
            {
                "pair": pair,
                "timestamp": ts,
                "signal": pred,
                "signal_type": "rise" if pred == 1 else "fall",
            }
        )
        rows_labels.append(
            {
                "pair": pair,
                "timestamp": ts,
                "label": true_label,
            }
        )

    signals = Signals(pl.DataFrame(rows_signals))
    labels = pl.DataFrame(rows_labels)
    return signals, labels


class TestSignalClassificationMetricCompute:
    def test_requires_labels(self):
        metric = SignalClassificationMetric()
        raw = _make_raw_data()
        signals, _ = _make_signals_and_labels()
        result, _ = metric.compute(raw, signals, labels=None)
        assert result is None

    def test_basic_compute(self):
        np.random.seed(42)
        metric = SignalClassificationMetric()
        raw = _make_raw_data()
        signals, labels = _make_signals_and_labels(n=100)
        result, _ctx = metric.compute(raw, signals, labels=labels)

        assert result is not None
        assert "quant" in result
        assert "series" in result

        quant = result["quant"]
        assert "precision" in quant
        assert "recall" in quant
        assert "f1" in quant
        assert "auc" in quant
        assert "confusion_matrix" in quant

    def test_perfect_classification(self):
        metric = SignalClassificationMetric()
        raw = _make_raw_data()
        signals, labels = _make_perfect_signals_and_labels(n=50)
        result, _ = metric.compute(raw, signals, labels=labels)

        quant = result["quant"]
        assert quant["precision"] == pytest.approx(1.0)
        assert quant["recall"] == pytest.approx(1.0)
        assert quant["f1"] == pytest.approx(1.0)

    def test_confusion_matrix_values(self):
        metric = SignalClassificationMetric()
        raw = _make_raw_data()
        signals, labels = _make_perfect_signals_and_labels(n=20)
        result, _ = metric.compute(raw, signals, labels=labels)

        cm = result["quant"]["confusion_matrix"]
        # Perfect classification: all predictions correct
        assert cm["tp"] + cm["tn"] == 20
        assert cm["fp"] == 0
        assert cm["fn"] == 0

    def test_string_labels_mapping(self):
        metric = SignalClassificationMetric(
            positive_labels=["rise", "up", 1],
            negative_labels=["fall", "down", 0],
        )
        raw = _make_raw_data()
        signals, labels = _make_string_labels(n=20)
        result, _ = metric.compute(raw, signals, labels=labels)

        assert result is not None
        quant = result["quant"]
        assert quant["total_signals"] == 20

    def test_roc_curve_data(self):
        np.random.seed(42)
        metric = SignalClassificationMetric()
        raw = _make_raw_data()
        signals, labels = _make_signals_and_labels(n=100)
        result, _ = metric.compute(raw, signals, labels=labels)

        roc = result["series"]["roc_curve"]
        assert "fpr" in roc
        assert "tpr" in roc
        assert "thresholds" in roc
        assert len(roc["fpr"]) == len(roc["tpr"])

    def test_no_signals_returns_none(self):
        metric = SignalClassificationMetric()
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
        labels = pl.DataFrame(
            {
                "pair": ["BTCUSDT"] * 10,
                "timestamp": [TS + timedelta(minutes=i) for i in range(10)],
                "label": [1, 0] * 5,
            }
        )
        result, _ = metric.compute(raw, signals, labels=labels)
        assert result is None

    def test_strength_statistics(self):
        metric = SignalClassificationMetric()
        raw = _make_raw_data()
        signals, labels = _make_perfect_signals_and_labels(n=20)
        result, _ = metric.compute(raw, signals, labels=labels)

        quant = result["quant"]
        assert "strength_mean" in quant
        assert "strength_std" in quant
        assert quant["strength_mean"] == pytest.approx(1.0)  # all signals are ±1


class TestSignalClassificationMetricPlot:
    def test_plot_returns_figure(self):
        np.random.seed(42)
        metric = SignalClassificationMetric()
        raw = _make_raw_data()
        signals, labels = _make_signals_and_labels(n=50)
        computed, ctx = metric.compute(raw, signals, labels=labels)
        fig = metric.plot(computed, ctx, raw, signals, labels=labels)

        assert fig is not None
        # Should have multiple traces (ROC, confusion matrix, histogram, table)
        assert len(fig.data) > 0

    def test_plot_none_metrics(self):
        metric = SignalClassificationMetric()
        raw = _make_raw_data()
        signals, labels = _make_signals_and_labels(n=50)
        fig = metric.plot(None, {}, raw, signals, labels=labels)
        assert fig is None

    def test_plot_dimensions(self):
        np.random.seed(42)
        metric = SignalClassificationMetric(chart_height=800, chart_width=1200)
        raw = _make_raw_data()
        signals, labels = _make_signals_and_labels(n=50)
        computed, ctx = metric.compute(raw, signals, labels=labels)
        fig = metric.plot(computed, ctx, raw, signals, labels=labels)

        assert fig.layout.height == 800
        assert fig.layout.width == 1200


class TestSignalClassificationMetricPostInit:
    def test_default_labels(self):
        metric = SignalClassificationMetric()
        assert "rise" in metric.positive_labels
        assert "fall" in metric.negative_labels
        assert 1 in metric.positive_labels
        assert 0 in metric.negative_labels

    def test_custom_labels(self):
        metric = SignalClassificationMetric(
            positive_labels=["buy", "long"],
            negative_labels=["sell", "short"],
        )
        assert metric.positive_labels == ["buy", "long"]
        assert metric.negative_labels == ["sell", "short"]
