"""Tests for target labelers: Labeler base, FixedHorizonLabeler."""

import math
from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.core.containers.signals import Signals
from signalflow.core.enums import SignalType
from signalflow.target.base import Labeler
from signalflow.target.fixed_horizon_labeler import FixedHorizonLabeler


def _price_df(n=20, pair="BTCUSDT", trend=1.0):
    """Simple OHLCV with a linear price trend."""
    base = datetime(2024, 1, 1)
    rows = []
    for i in range(n):
        price = 100.0 + trend * i
        rows.append(
            {
                "pair": pair,
                "timestamp": base + timedelta(minutes=i),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1000.0,
            }
        )
    return pl.DataFrame(rows)


def _multi_pair_df(n=20):
    btc = _price_df(n, pair="BTCUSDT", trend=1.0)
    eth = _price_df(n, pair="ETHUSDT", trend=-1.0)
    return pl.concat([btc, eth])


# ── Labeler base validation ─────────────────────────────────────────────────


class TestLabelerBaseValidation:
    def test_not_dataframe_raises(self):
        labeler = FixedHorizonLabeler(horizon=5)
        with pytest.raises(TypeError, match="pl.DataFrame"):
            labeler.compute("not a df")

    def test_missing_pair_col_raises(self):
        labeler = FixedHorizonLabeler(horizon=5)
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "close": [100.0]})
        with pytest.raises(ValueError, match="pair"):
            labeler.compute(df)

    def test_missing_ts_col_raises(self):
        labeler = FixedHorizonLabeler(horizon=5)
        df = pl.DataFrame({"pair": ["BTCUSDT"], "close": [100.0]})
        with pytest.raises(ValueError, match="timestamp"):
            labeler.compute(df)

    def test_keep_input_columns(self):
        labeler = FixedHorizonLabeler(horizon=5, keep_input_columns=True, mask_to_signals=False)
        df = _price_df(20)
        result = labeler.compute(df)
        assert "close" in result.columns
        assert "open" in result.columns
        assert "label" in result.columns

    def test_output_columns_projection(self):
        labeler = FixedHorizonLabeler(horizon=5, mask_to_signals=False)
        df = _price_df(20)
        result = labeler.compute(df)
        assert "pair" in result.columns
        assert "timestamp" in result.columns
        assert "label" in result.columns
        assert "close" not in result.columns


# ── FixedHorizonLabeler ─────────────────────────────────────────────────────


class TestFixedHorizonLabeler:
    def test_horizon_zero_raises(self):
        with pytest.raises(ValueError, match="horizon must be > 0"):
            FixedHorizonLabeler(horizon=0)

    def test_negative_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon must be > 0"):
            FixedHorizonLabeler(horizon=-5)

    def test_basic_rise_label(self):
        labeler = FixedHorizonLabeler(horizon=5, mask_to_signals=False)
        df = _price_df(20, trend=1.0)  # rising prices
        result = labeler.compute(df)
        # First 15 bars should be RISE (future > current)
        labels = result.filter(pl.col("label") != SignalType.NONE.value)
        assert (labels["label"] == SignalType.RISE.value).all()

    def test_basic_fall_label(self):
        labeler = FixedHorizonLabeler(horizon=5, mask_to_signals=False)
        df = _price_df(20, trend=-1.0)  # falling prices
        result = labeler.compute(df)
        labels = result.filter(pl.col("label") != SignalType.NONE.value)
        assert (labels["label"] == SignalType.FALL.value).all()

    def test_none_at_end(self):
        labeler = FixedHorizonLabeler(horizon=5, mask_to_signals=False)
        df = _price_df(20, trend=1.0)
        result = labeler.compute(df)
        # Last 5 bars have no future data → NONE
        last_5 = result.tail(5)
        assert (last_5["label"] == SignalType.NONE.value).all()

    def test_include_meta(self):
        labeler = FixedHorizonLabeler(horizon=5, include_meta=True, mask_to_signals=False)
        df = _price_df(20)
        result = labeler.compute(df)
        assert "t1" in result.columns
        assert "ret" in result.columns

    def test_ret_is_log_return(self):
        labeler = FixedHorizonLabeler(horizon=5, include_meta=True, mask_to_signals=False)
        df = _price_df(20, trend=1.0)
        result = labeler.compute(df)
        # Check first row: close=100, future=105, ret=log(105/100)
        first_ret = result.filter(pl.col("ret").is_not_null())["ret"][0]
        expected = math.log(105.0 / 100.0)
        assert first_ret == pytest.approx(expected, rel=1e-4)

    def test_empty_group(self):
        labeler = FixedHorizonLabeler(horizon=5, mask_to_signals=False)
        empty = pl.DataFrame(
            {"pair": [], "timestamp": [], "close": [], "open": [], "high": [], "low": [], "volume": []}
        ).cast(
            {
                "timestamp": pl.Datetime,
                "close": pl.Float64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "volume": pl.Float64,
            }
        )
        # Empty group_by never calls compute_group, so "label" column is not produced
        with pytest.raises(Exception):
            labeler.compute(empty)

    def test_multi_pair(self):
        labeler = FixedHorizonLabeler(horizon=5, mask_to_signals=False)
        df = _multi_pair_df(20)
        result = labeler.compute(df)
        assert result.height == 40  # 20 per pair
        # BTC rising → RISE, ETH falling → FALL
        btc = result.filter(pl.col("pair") == "BTCUSDT").filter(pl.col("label") != SignalType.NONE.value)
        eth = result.filter(pl.col("pair") == "ETHUSDT").filter(pl.col("label") != SignalType.NONE.value)
        assert (btc["label"] == SignalType.RISE.value).all()
        assert (eth["label"] == SignalType.FALL.value).all()

    def test_length_preserved(self):
        labeler = FixedHorizonLabeler(horizon=5, mask_to_signals=False)
        df = _price_df(50)
        result = labeler.compute(df)
        assert result.height == df.height

    def test_missing_price_col_raises(self):
        labeler = FixedHorizonLabeler(horizon=5, price_col="nonexistent", mask_to_signals=False)
        df = _price_df(10)
        # ValueError raised inside compute_group is wrapped by Polars as PanicException (BaseException)
        with pytest.raises(BaseException, match="nonexistent"):
            labeler.compute(df)


# ── Labeler signal filtering ───────────────────────────────────────────────


class TestLabelerSignalFiltering:
    def test_filter_by_signal_type(self):
        labeler = FixedHorizonLabeler(horizon=5, filter_signal_type=SignalType.RISE, mask_to_signals=False)
        df = _price_df(20)
        base = datetime(2024, 1, 1)
        signals = Signals(
            pl.DataFrame(
                {
                    "pair": ["BTCUSDT", "BTCUSDT"],
                    "timestamp": [base, base + timedelta(minutes=5)],
                    "signal_type": [SignalType.RISE.value, SignalType.FALL.value],
                    "signal": [1, -1],
                }
            )
        )
        result = labeler.compute(df, signals=signals)
        # Only RISE signal row kept (inner join)
        assert result.height == 1
