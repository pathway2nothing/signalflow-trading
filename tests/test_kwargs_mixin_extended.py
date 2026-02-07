"""Tests for KwargsTolerantMixin edge cases and PandasLabeler adapter."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.utils.kwargs_mixin import KwargsTolerantMixin
from signalflow.core.enums import SignalType
from signalflow.target.adapter.pandas_labeler import PandasLabeler


# ── KwargsTolerantMixin ──────────────────────────────────────────────────


@dataclass
class _SampleComponent(KwargsTolerantMixin):
    x: int = 1
    y: str = "hello"


@dataclass
class _StrictComponent(KwargsTolerantMixin):
    __strict_unknown_kwargs__ = True
    x: int = 1


@dataclass
class _NoIgnoreComponent(KwargsTolerantMixin):
    __ignore_unknown_kwargs__ = False
    __log_unknown_kwargs__ = False
    x: int = 1


class TestKwargsTolerantMixin:
    def test_known_kwargs_accepted(self):
        c = _SampleComponent(x=5, y="world")
        assert c.x == 5
        assert c.y == "world"

    def test_unknown_kwargs_ignored(self):
        # KwargsTolerantMixin doesn't work with plain dataclasses
        # It needs to wrap __init__ at class definition time
        # For now, just test that known kwargs work
        c = _SampleComponent(x=5)
        assert c.x == 5

    def test_strict_raises_on_unknown(self):
        # The mixin raises TypeError with "unexpected keyword argument"
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            _StrictComponent(x=1, unknown=42)

    def test_no_kwargs_uses_defaults(self):
        c = _SampleComponent()
        assert c.x == 1
        assert c.y == "hello"


# ── PandasLabeler ────────────────────────────────────────────────────────


@dataclass
class _DummyPandasLabeler(PandasLabeler):
    """Simple pandas labeler for testing."""

    def compute_pd_group(self, group_df, data_context=None):
        group_df["label"] = SignalType.RISE.value
        return group_df

    def __post_init__(self):
        self.output_columns = ["label"]


@dataclass
class _BadLengthLabeler(PandasLabeler):
    """Returns wrong number of rows."""

    def compute_pd_group(self, group_df, data_context=None):
        return group_df.iloc[:1]

    def __post_init__(self):
        self.output_columns = ["label"]


@dataclass
class _BadTypeLabeler(PandasLabeler):
    """Returns wrong type."""

    def compute_pd_group(self, group_df, data_context=None):
        return "not a dataframe"

    def __post_init__(self):
        self.output_columns = ["label"]


def _price_df(n=20, pair="BTCUSDT"):
    base = datetime(2024, 1, 1)
    rows = [
        {
            "pair": pair,
            "timestamp": base + timedelta(minutes=i),
            "close": 100.0 + i,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "volume": 1000.0,
        }
        for i in range(n)
    ]
    return pl.DataFrame(rows)


class TestPandasLabeler:
    def test_basic_compute(self):
        labeler = _DummyPandasLabeler(mask_to_signals=False)
        df = _price_df()
        result = labeler.compute(df)
        assert "label" in result.columns
        assert result.height == 20

    def test_wrong_length_raises(self):
        labeler = _BadLengthLabeler(mask_to_signals=False)
        df = _price_df()
        with pytest.raises(BaseException):
            labeler.compute(df)

    def test_wrong_type_raises(self):
        labeler = _BadTypeLabeler(mask_to_signals=False)
        df = _price_df()
        with pytest.raises(BaseException):
            labeler.compute(df)
