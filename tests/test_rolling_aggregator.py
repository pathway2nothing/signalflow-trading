"""Tests for signalflow.core.rolling_aggregator.RollingAggregator."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.core.rolling_aggregator import RollingAggregator


@pytest.fixture
def ohlcv_30():
    """30-bar 1-minute OHLCV for two pairs."""
    base = datetime(2024, 1, 1, 10, 0)
    rows = []
    for pair in ["BTCUSDT", "ETHUSDT"]:
        for i in range(30):
            rows.append(
                {
                    "pair": pair,
                    "timestamp": base + timedelta(minutes=i),
                    "open": 100.0 + i,
                    "high": 105.0 + i,
                    "low": 95.0 + i,
                    "close": 102.0 + i,
                    "volume": 1000.0 + i * 10,
                }
            )
    return pl.DataFrame(rows)


# ── Validation ──────────────────────────────────────────────────────────────


class TestRollingAggregatorValidation:
    def test_zero_offset_raises(self, ohlcv_30):
        agg = RollingAggregator(offset_window=0)
        with pytest.raises(ValueError, match="offset_window must be > 0"):
            agg.resample(ohlcv_30)

    def test_negative_offset_raises(self, ohlcv_30):
        agg = RollingAggregator(offset_window=-1)
        with pytest.raises(ValueError, match="offset_window must be > 0"):
            agg.resample(ohlcv_30)

    def test_missing_ts_col_raises(self):
        agg = RollingAggregator(offset_window=5)
        df = pl.DataFrame({"pair": ["BTC"], "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]})
        with pytest.raises(ValueError, match="timestamp"):
            agg.resample(df)

    def test_missing_pair_col_raises(self):
        agg = RollingAggregator(offset_window=5)
        df = pl.DataFrame(
            {"timestamp": [datetime(2024, 1, 1)], "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]}
        )
        with pytest.raises(ValueError, match="pair"):
            agg.resample(df)

    def test_missing_ohlc_cols_raises(self):
        agg = RollingAggregator(offset_window=5)
        df = pl.DataFrame({"pair": ["BTC"], "timestamp": [datetime(2024, 1, 1)]})
        with pytest.raises(ValueError, match="missing"):
            agg.resample(df)

    def test_non_spot_raises(self, ohlcv_30):
        agg = RollingAggregator(offset_window=5, raw_data_type="futures")
        with pytest.raises(NotImplementedError, match="spot"):
            agg.resample(ohlcv_30)


# ── Resample ────────────────────────────────────────────────────────────────


class TestRollingAggregatorResample:
    def test_add_mode(self, ohlcv_30):
        agg = RollingAggregator(offset_window=5, mode="add")
        result = agg.resample(ohlcv_30)
        assert f"rs_5m_open" in result.columns
        assert f"rs_5m_high" in result.columns
        assert f"rs_5m_close" in result.columns
        assert f"rs_5m_volume" in result.columns

    def test_replace_mode(self, ohlcv_30):
        agg = RollingAggregator(offset_window=5, mode="replace")
        result = agg.resample(ohlcv_30)
        # Same columns as input (open/high/low/close replaced in-place)
        assert "open" in result.columns
        assert "rs_5m_open" not in result.columns

    def test_length_preserved(self, ohlcv_30):
        agg = RollingAggregator(offset_window=5, mode="add")
        result = agg.resample(ohlcv_30)
        assert result.height == ohlcv_30.height

    def test_first_k_minus_1_null(self, ohlcv_30):
        agg = RollingAggregator(offset_window=5, mode="add")
        result = agg.resample(ohlcv_30)
        btc = result.filter(pl.col("pair") == "BTCUSDT")
        # First 4 rows should have null rs_5m_high (min_periods=5)
        first_4 = btc.head(4)
        assert first_4["rs_5m_high"].null_count() == 4

    def test_rolling_high_is_max(self, ohlcv_30):
        agg = RollingAggregator(offset_window=5, mode="add")
        result = agg.resample(ohlcv_30)
        btc = result.filter(pl.col("pair") == "BTCUSDT").drop_nulls("rs_5m_high")
        # At row 4 (5th bar), high=[105,106,107,108,109], max=109
        assert btc["rs_5m_high"][0] == pytest.approx(109.0)

    def test_rolling_volume_is_sum(self, ohlcv_30):
        agg = RollingAggregator(offset_window=5, mode="add")
        result = agg.resample(ohlcv_30)
        btc = result.filter(pl.col("pair") == "BTCUSDT").drop_nulls("rs_5m_volume")
        # At row 4: vol=[1000,1010,1020,1030,1040], sum=5100
        assert btc["rs_5m_volume"][0] == pytest.approx(5100.0)


# ── Offset column ──────────────────────────────────────────────────────────


class TestRollingAggregatorOffset:
    def test_add_offset_column(self, ohlcv_30):
        agg = RollingAggregator(offset_window=5)
        result = agg.add_offset_column(ohlcv_30)
        assert "resample_offset" in result.columns
        btc = result.filter(pl.col("pair") == "BTCUSDT")
        offsets = btc["resample_offset"].to_list()
        # 10:00→0, 10:01→1, 10:02→2, 10:03→3, 10:04→4, 10:05→0, ...
        assert offsets[:5] == [0, 1, 2, 3, 4]
        assert offsets[5] == 0

    def test_get_last_offset(self, ohlcv_30):
        agg = RollingAggregator(offset_window=5)
        offset = agg.get_last_offset(ohlcv_30)
        # Last timestamp minute is 10:29, 29 % 5 = 4
        assert offset == 4

    def test_out_prefix_default(self):
        agg = RollingAggregator(offset_window=5)
        assert agg.out_prefix == "rs_5m_"

    def test_out_prefix_custom(self):
        agg = RollingAggregator(offset_window=5, prefix="my_")
        assert agg.out_prefix == "my_"

    def test_get_last_offset_empty_raises(self):
        agg = RollingAggregator(offset_window=5)
        df = pl.DataFrame({"pair": [], "timestamp": []}).cast({"timestamp": pl.Datetime})
        with pytest.raises(ValueError, match="Empty"):
            agg.get_last_offset(df)
