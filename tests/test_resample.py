"""Tests for signalflow.data.resample module."""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.data.resample import (
    EXCHANGE_TIMEFRAMES,
    TIMEFRAME_MINUTES,
    align_to_timeframe,
    can_resample,
    detect_timeframe,
    resample_ohlcv,
    select_best_timeframe,
    timeframe_to_minutes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    pairs: list[str],
    start: datetime,
    n_bars: int,
    tf_minutes: int,
    extra_cols: dict[str, list] | None = None,
) -> pl.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rows: list[dict] = []
    for pair in pairs:
        for i in range(n_bars):
            ts = start + timedelta(minutes=tf_minutes * i)
            rows.append(
                {
                    "pair": pair,
                    "timestamp": ts,
                    "open": 100.0 + i,
                    "high": 105.0 + i,
                    "low": 95.0 + i,
                    "close": 102.0 + i,
                    "volume": 1000.0 + i * 10,
                }
            )
    df = pl.DataFrame(rows)
    if extra_cols:
        for col, vals in extra_cols.items():
            # Repeat values to match total rows
            full_vals = vals * len(pairs)
            df = df.with_columns(pl.Series(col, full_vals[: df.height]))
    return df


# ---------------------------------------------------------------------------
# timeframe_to_minutes
# ---------------------------------------------------------------------------


class TestTimeframeToMinutes:
    def test_known_values(self):
        assert timeframe_to_minutes("1m") == 1
        assert timeframe_to_minutes("1h") == 60
        assert timeframe_to_minutes("4h") == 240
        assert timeframe_to_minutes("1d") == 1440

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown timeframe"):
            timeframe_to_minutes("7h")


# ---------------------------------------------------------------------------
# can_resample
# ---------------------------------------------------------------------------


class TestCanResample:
    def test_exact_multiple(self):
        assert can_resample("1h", "4h") is True
        assert can_resample("1m", "1h") is True
        assert can_resample("15m", "1h") is True

    def test_not_multiple(self):
        # 3h is not a known timeframe → returns False
        assert can_resample("1h", "3h") is False
        # 5m doesn't evenly divide 4h (240/5=48 → actually it does!)
        # Use a real non-divisible combo: 3m → 1h (60/3=20 → divisible)
        # Actually 3m → 4h: 240/3 = 80 → divisible
        # 15m → 8h: 480/15 = 32 → divisible
        # All standard timeframes divide each other... test unknown target
        assert can_resample("1h", "unknown") is False

    def test_downscale_not_allowed(self):
        assert can_resample("4h", "1h") is False

    def test_same_timeframe(self):
        assert can_resample("1h", "1h") is True


# ---------------------------------------------------------------------------
# detect_timeframe
# ---------------------------------------------------------------------------


class TestDetectTimeframe:
    def test_detect_1h(self):
        df = _make_ohlcv(["BTCUSDT"], datetime(2024, 1, 1), n_bars=24, tf_minutes=60)
        assert detect_timeframe(df) == "1h"

    def test_detect_4h(self):
        df = _make_ohlcv(["BTCUSDT"], datetime(2024, 1, 1), n_bars=12, tf_minutes=240)
        assert detect_timeframe(df) == "4h"

    def test_detect_multi_pair(self):
        df = _make_ohlcv(["BTCUSDT", "ETHUSDT"], datetime(2024, 1, 1), n_bars=10, tf_minutes=60)
        assert detect_timeframe(df) == "1h"

    def test_too_few_rows(self):
        df = _make_ohlcv(["BTCUSDT"], datetime(2024, 1, 1), n_bars=1, tf_minutes=60)
        with pytest.raises(ValueError, match="at least 2 rows"):
            detect_timeframe(df)


# ---------------------------------------------------------------------------
# resample_ohlcv
# ---------------------------------------------------------------------------


class TestResampleOhlcv:
    def test_1h_to_4h(self):
        df = _make_ohlcv(["BTCUSDT"], datetime(2024, 1, 1), n_bars=8, tf_minutes=60)
        result = resample_ohlcv(df, "1h", "4h")

        assert result.height == 2  # 8 bars / 4 = 2 groups

        first_group = result.row(0, named=True)
        # open = first bar's open
        assert first_group["open"] == 100.0
        # high = max of first 4 bars (105, 106, 107, 108)
        assert first_group["high"] == 108.0
        # low = min of first 4 bars (95, 96, 97, 98)
        assert first_group["low"] == 95.0
        # close = last bar's close
        assert first_group["close"] == 105.0
        # volume = sum of first 4 bars (1000+1010+1020+1030)
        assert first_group["volume"] == 4060.0

    def test_same_timeframe_noop(self):
        df = _make_ohlcv(["BTCUSDT"], datetime(2024, 1, 1), n_bars=4, tf_minutes=60)
        result = resample_ohlcv(df, "1h", "1h")
        assert result.height == df.height
        # Same object returned (no-op)
        assert result is df

    def test_incompatible_raises(self):
        df = _make_ohlcv(["BTCUSDT"], datetime(2024, 1, 1), n_bars=4, tf_minutes=60)
        # 4h → 1h is downscale = not allowed
        with pytest.raises(ValueError, match="Cannot resample"):
            resample_ohlcv(df, "4h", "1h")

    def test_preserves_pairs(self):
        df = _make_ohlcv(["BTCUSDT", "ETHUSDT"], datetime(2024, 1, 1), n_bars=8, tf_minutes=60)
        result = resample_ohlcv(df, "1h", "4h")
        pairs = result["pair"].unique().sort().to_list()
        assert pairs == ["BTCUSDT", "ETHUSDT"]
        # Each pair: 8 bars → 2 groups
        assert result.height == 4

    def test_custom_fill_rules(self):
        n_bars = 4
        df = _make_ohlcv(
            ["BTCUSDT"],
            datetime(2024, 1, 1),
            n_bars=n_bars,
            tf_minutes=60,
            extra_cols={"funding_rate": [0.01, 0.02, 0.03, 0.04]},
        )
        result = resample_ohlcv(df, "1h", "4h", fill_rules={"funding_rate": "mean"})
        assert result.height == 1
        # mean of [0.01, 0.02, 0.03, 0.04] = 0.025
        assert abs(result["funding_rate"][0] - 0.025) < 1e-9

    def test_extra_column_defaults_to_last(self):
        n_bars = 4
        df = _make_ohlcv(
            ["BTCUSDT"],
            datetime(2024, 1, 1),
            n_bars=n_bars,
            tf_minutes=60,
            extra_cols={"indicator": [10.0, 20.0, 30.0, 40.0]},
        )
        result = resample_ohlcv(df, "1h", "4h")
        assert result.height == 1
        # "last" rule → 40.0
        assert result["indicator"][0] == 40.0


# ---------------------------------------------------------------------------
# align_to_timeframe
# ---------------------------------------------------------------------------


class TestAlignToTimeframe:
    def test_same_tf_noop(self):
        df = _make_ohlcv(["BTCUSDT"], datetime(2024, 1, 1), n_bars=8, tf_minutes=60)
        result = align_to_timeframe(df, "1h")
        assert result.height == df.height

    def test_resample_succeeds(self):
        df = _make_ohlcv(["BTCUSDT"], datetime(2024, 1, 1), n_bars=8, tf_minutes=60)
        result = align_to_timeframe(df, "4h")
        assert result.height == 2

    def test_incompatible_warns_downscale(self):
        # 4h data → 1h (downscale) is not possible
        df = _make_ohlcv(["BTCUSDT"], datetime(2024, 1, 1), n_bars=8, tf_minutes=240)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = align_to_timeframe(df, "1h")
            assert result.height == df.height  # returned unchanged
            assert len(w) == 1
            assert "Cannot resample" in str(w[0].message)


# ---------------------------------------------------------------------------
# select_best_timeframe
# ---------------------------------------------------------------------------


class TestSelectBestTimeframe:
    def test_exact_match(self):
        assert select_best_timeframe("binance", "8h") == "8h"
        assert select_best_timeframe("binance", "4h") == "4h"

    def test_bybit_8h_returns_4h(self):
        # Bybit doesn't support 8h; 4h is largest divisor of 480min
        assert select_best_timeframe("bybit", "8h") == "4h"

    def test_okx_8h_returns_4h(self):
        assert select_best_timeframe("okx", "8h") == "4h"

    def test_kraken_spot_2h_returns_1h(self):
        # Kraken spot doesn't support 2h; 1h is largest divisor
        assert select_best_timeframe("kraken_spot", "2h") == "1h"

    def test_unknown_exchange_raises(self):
        with pytest.raises(ValueError, match="Unknown exchange"):
            select_best_timeframe("unknown_exchange", "1h")

    def test_case_insensitive(self):
        assert select_best_timeframe("Binance", "4h") == "4h"
        assert select_best_timeframe("BYBIT", "8h") == "4h"


# ---------------------------------------------------------------------------
# EXCHANGE_TIMEFRAMES constant
# ---------------------------------------------------------------------------


class TestExchangeTimeframes:
    def test_binance_has_8h(self):
        assert "8h" in EXCHANGE_TIMEFRAMES["binance"]

    def test_bybit_no_8h(self):
        assert "8h" not in EXCHANGE_TIMEFRAMES["bybit"]

    def test_okx_no_8h(self):
        assert "8h" not in EXCHANGE_TIMEFRAMES["okx"]

    def test_kraken_spot_limited(self):
        assert "2h" not in EXCHANGE_TIMEFRAMES["kraken_spot"]
        assert "3m" not in EXCHANGE_TIMEFRAMES["kraken_spot"]

    def test_all_exchanges_have_1m(self):
        for exchange, tfs in EXCHANGE_TIMEFRAMES.items():
            assert "1m" in tfs, f"{exchange} missing 1m"
