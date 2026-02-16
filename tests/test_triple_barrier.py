"""Tests for TripleBarrierLabeler and TakeProfitLabeler."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.core.enums import SignalType
from signalflow.target.take_profit_labeler import TakeProfitLabeler
from signalflow.target.triple_barrier_labeler import TripleBarrierLabeler


def _price_df(n=200, pair="BTCUSDT", trend=0.5):
    """OHLCV with a linear trend, enough rows for vol_window warmup."""
    base = datetime(2024, 1, 1)
    rows = []
    for i in range(n):
        price = 100.0 + trend * i
        rows.append(
            {
                "pair": pair,
                "timestamp": base + timedelta(minutes=i),
                "open": price - 0.2,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 1000.0,
            }
        )
    return pl.DataFrame(rows)


def _multi_pair_df(n=200):
    btc = _price_df(n, pair="BTCUSDT", trend=0.5)
    eth = _price_df(n, pair="ETHUSDT", trend=-0.5)
    return pl.concat([btc, eth])


# ── TripleBarrierLabeler validation ──────────────────────────────────────


class TestTripleBarrierValidation:
    def test_vol_window_too_small_raises(self):
        with pytest.raises(ValueError, match="vol_window"):
            TripleBarrierLabeler(vol_window=1)

    def test_horizon_zero_raises(self):
        with pytest.raises(ValueError, match="horizon"):
            TripleBarrierLabeler(horizon=0)

    def test_negative_multiplier_raises(self):
        with pytest.raises(ValueError, match="profit_multiplier"):
            TripleBarrierLabeler(profit_multiplier=-1.0)

    def test_stop_loss_zero_raises(self):
        with pytest.raises(ValueError, match="profit_multiplier"):
            TripleBarrierLabeler(stop_loss_multiplier=0)

    def test_missing_price_col_raises(self):
        labeler = TripleBarrierLabeler(price_col="nonexistent", mask_to_signals=False)
        df = _price_df(100)
        with pytest.raises(BaseException, match="nonexistent"):
            labeler.compute(df)


# ── TripleBarrierLabeler compute ─────────────────────────────────────────


class TestTripleBarrierCompute:
    def test_basic_labels_produced(self):
        labeler = TripleBarrierLabeler(vol_window=20, horizon=60, mask_to_signals=False)
        df = _price_df(200, trend=0.5)
        result = labeler.compute(df)
        assert "label" in result.columns
        assert result.height == 200
        labels = result["label"].unique().to_list()
        assert SignalType.RISE.value in labels or SignalType.FALL.value in labels

    def test_rising_trend_mostly_rise(self):
        labeler = TripleBarrierLabeler(
            vol_window=20,
            horizon=60,
            profit_multiplier=1.0,
            stop_loss_multiplier=1.0,
            mask_to_signals=False,
        )
        df = _price_df(200, trend=1.0)
        result = labeler.compute(df)
        rise_count = result.filter(pl.col("label") == SignalType.RISE.value).height
        fall_count = result.filter(pl.col("label") == SignalType.FALL.value).height
        assert rise_count > fall_count

    def test_falling_trend_mostly_fall(self):
        labeler = TripleBarrierLabeler(
            vol_window=20,
            horizon=60,
            profit_multiplier=1.0,
            stop_loss_multiplier=1.0,
            mask_to_signals=False,
        )
        df = _price_df(200, trend=-1.0)
        result = labeler.compute(df)
        fall_count = result.filter(pl.col("label") == SignalType.FALL.value).height
        rise_count = result.filter(pl.col("label") == SignalType.RISE.value).height
        assert fall_count > rise_count

    def test_empty_df(self):
        labeler = TripleBarrierLabeler(vol_window=5, horizon=10, mask_to_signals=False)
        empty = pl.DataFrame(
            {
                "pair": [],
                "timestamp": [],
                "close": [],
                "open": [],
                "high": [],
                "low": [],
                "volume": [],
            }
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
        with pytest.raises(Exception):
            labeler.compute(empty)

    def test_include_meta(self):
        labeler = TripleBarrierLabeler(
            vol_window=20,
            horizon=60,
            include_meta=True,
            mask_to_signals=False,
        )
        df = _price_df(200)
        result = labeler.compute(df)
        assert "t_hit" in result.columns
        assert "ret" in result.columns

    def test_multi_pair(self):
        labeler = TripleBarrierLabeler(
            vol_window=20,
            horizon=60,
            mask_to_signals=False,
        )
        df = _multi_pair_df(200)
        result = labeler.compute(df)
        assert result.height == 400

    def test_length_preserved(self):
        labeler = TripleBarrierLabeler(
            vol_window=20,
            horizon=60,
            mask_to_signals=False,
        )
        df = _price_df(200)
        result = labeler.compute(df)
        assert result.height == df.height

    def test_signal_masking(self):
        labeler = TripleBarrierLabeler(
            vol_window=20,
            horizon=60,
            mask_to_signals=True,
        )
        df = _price_df(200)
        base = datetime(2024, 1, 1)
        signal_keys = pl.DataFrame(
            {
                "pair": ["BTCUSDT", "BTCUSDT"],
                "timestamp": [base + timedelta(minutes=50), base + timedelta(minutes=100)],
            }
        )
        result = labeler.compute(df, data_context={"signal_keys": signal_keys})
        non_none = result.filter(pl.col("label") != SignalType.NONE.value)
        assert non_none.height <= 2


# ── TakeProfitLabeler validation ────────────────────────────────


class TestTakeProfitValidation:
    def test_horizon_zero_raises(self):
        with pytest.raises(ValueError, match="horizon"):
            TakeProfitLabeler(horizon=0)

    def test_negative_pct_raises(self):
        with pytest.raises(ValueError, match="barrier_pct"):
            TakeProfitLabeler(barrier_pct=-0.01)

    def test_zero_pct_raises(self):
        with pytest.raises(ValueError, match="barrier_pct"):
            TakeProfitLabeler(barrier_pct=0)

    def test_missing_price_col(self):
        labeler = TakeProfitLabeler(price_col="nonexistent", mask_to_signals=False)
        df = _price_df(50)
        with pytest.raises(BaseException, match="nonexistent"):
            labeler.compute(df)


# ── TakeProfitLabeler compute ───────────────────────────────────


class TestTakeProfitCompute:
    def test_basic_labels(self):
        labeler = TakeProfitLabeler(
            horizon=30,
            barrier_pct=0.02,
            mask_to_signals=False,
        )
        df = _price_df(100, trend=0.5)
        result = labeler.compute(df)
        assert "label" in result.columns
        assert result.height == 100

    def test_rising_trend_rise_labels(self):
        labeler = TakeProfitLabeler(
            horizon=30,
            barrier_pct=0.01,
            mask_to_signals=False,
        )
        df = _price_df(100, trend=1.0)
        result = labeler.compute(df)
        rise_count = result.filter(pl.col("label") == SignalType.RISE.value).height
        fall_count = result.filter(pl.col("label") == SignalType.FALL.value).height
        assert rise_count > fall_count

    def test_falling_trend_fall_labels(self):
        labeler = TakeProfitLabeler(
            horizon=30,
            barrier_pct=0.01,
            mask_to_signals=False,
        )
        df = _price_df(100, trend=-1.0)
        result = labeler.compute(df)
        fall_count = result.filter(pl.col("label") == SignalType.FALL.value).height
        rise_count = result.filter(pl.col("label") == SignalType.RISE.value).height
        assert fall_count > rise_count

    def test_include_meta(self):
        labeler = TakeProfitLabeler(
            horizon=30,
            barrier_pct=0.02,
            include_meta=True,
            mask_to_signals=False,
        )
        df = _price_df(100)
        result = labeler.compute(df)
        assert "t_hit" in result.columns
        assert "ret" in result.columns

    def test_multi_pair(self):
        labeler = TakeProfitLabeler(
            horizon=30,
            barrier_pct=0.02,
            mask_to_signals=False,
        )
        df = _multi_pair_df(100)
        result = labeler.compute(df)
        assert result.height == 200

    def test_length_preserved(self):
        labeler = TakeProfitLabeler(
            horizon=30,
            barrier_pct=0.02,
            mask_to_signals=False,
        )
        df = _price_df(100)
        result = labeler.compute(df)
        assert result.height == df.height

    def test_empty_df(self):
        labeler = TakeProfitLabeler(
            horizon=10,
            mask_to_signals=False,
        )
        empty = pl.DataFrame(
            {
                "pair": [],
                "timestamp": [],
                "close": [],
                "open": [],
                "high": [],
                "low": [],
                "volume": [],
            }
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
        with pytest.raises(Exception):
            labeler.compute(empty)

    def test_signal_masking(self):
        labeler = TakeProfitLabeler(
            horizon=30,
            barrier_pct=0.02,
            mask_to_signals=True,
        )
        df = _price_df(100)
        base = datetime(2024, 1, 1)
        signal_keys = pl.DataFrame(
            {
                "pair": ["BTCUSDT", "BTCUSDT"],
                "timestamp": [base + timedelta(minutes=10), base + timedelta(minutes=50)],
            }
        )
        result = labeler.compute(df, data_context={"signal_keys": signal_keys})
        non_none = result.filter(pl.col("label") != SignalType.NONE.value)
        assert non_none.height <= 2

    def test_keep_input_columns(self):
        labeler = TakeProfitLabeler(
            horizon=30,
            barrier_pct=0.02,
            keep_input_columns=True,
            mask_to_signals=False,
        )
        df = _price_df(100)
        result = labeler.compute(df)
        assert "close" in result.columns
        assert "label" in result.columns
