"""Tests for signalflow.core.containers.signals.Signals."""

from datetime import datetime

import polars as pl
import pytest

from signalflow.core.containers.signals import Signals
from signalflow.core.enums import SignalType


class TestSignalsCreation:
    def test_create(self, sample_signals_df):
        s = Signals(value=sample_signals_df)
        assert isinstance(s.value, pl.DataFrame)
        assert len(s.value) == 3

    def test_frozen(self, sample_signals_df):
        s = Signals(value=sample_signals_df)
        with pytest.raises(AttributeError):
            s.value = pl.DataFrame()


class TestSignalsApply:
    def test_apply_filter(self, sample_signals_df):
        s = Signals(value=sample_signals_df)

        def keep_rise(df: pl.DataFrame) -> pl.DataFrame:
            return df.filter(pl.col("signal_type") == SignalType.RISE.value)

        result = s.apply(keep_rise)
        assert isinstance(result, Signals)
        assert len(result.value) == 1
        assert result.value["signal_type"][0] == SignalType.RISE.value

    def test_apply_returns_new_instance(self, sample_signals_df):
        s = Signals(value=sample_signals_df)
        result = s.apply(lambda df: df)
        assert result is not s


class TestSignalsPipe:
    def test_pipe_multiple_transforms(self, sample_signals_df):
        s = Signals(value=sample_signals_df)

        def drop_none(df: pl.DataFrame) -> pl.DataFrame:
            return df.filter(pl.col("signal_type") != SignalType.NONE.value)

        def high_prob(df: pl.DataFrame) -> pl.DataFrame:
            return df.filter(pl.col("probability") > 0.85)

        result = s.pipe(drop_none, high_prob)
        assert len(result.value) == 1
        assert result.value["probability"][0] == 0.9

    def test_pipe_no_transforms(self, sample_signals_df):
        s = Signals(value=sample_signals_df)
        result = s.pipe()
        assert len(result.value) == len(s.value)


class TestSignalsMerge:
    def test_add_basic(self):
        base = datetime(2024, 1, 1)
        s1 = Signals(
            value=pl.DataFrame(
                {
                    "pair": ["BTCUSDT"],
                    "timestamp": [base],
                    "signal_type": [SignalType.RISE.value],
                    "signal": [1],
                    "probability": [0.8],
                }
            )
        )
        s2 = Signals(
            value=pl.DataFrame(
                {
                    "pair": ["ETHUSDT"],
                    "timestamp": [base],
                    "signal_type": [SignalType.FALL.value],
                    "signal": [-1],
                    "probability": [0.7],
                }
            )
        )
        merged = s1 + s2
        assert len(merged.value) == 2

    def test_add_none_overridden(self):
        """Non-NONE signal should override NONE on same (pair, timestamp)."""
        base = datetime(2024, 1, 1)
        s1 = Signals(
            value=pl.DataFrame(
                {
                    "pair": ["BTCUSDT"],
                    "timestamp": [base],
                    "signal_type": [SignalType.NONE.value],
                    "signal": [0],
                    "probability": [0.0],
                }
            )
        )
        s2 = Signals(
            value=pl.DataFrame(
                {
                    "pair": ["BTCUSDT"],
                    "timestamp": [base],
                    "signal_type": [SignalType.RISE.value],
                    "signal": [1],
                    "probability": [0.9],
                }
            )
        )
        merged = s1 + s2
        assert len(merged.value) == 1
        assert merged.value["signal_type"][0] == SignalType.RISE.value

    def test_add_type_error(self, sample_signals_df):
        s = Signals(value=sample_signals_df)
        result = s.__add__("not_signals")
        assert result is NotImplemented
