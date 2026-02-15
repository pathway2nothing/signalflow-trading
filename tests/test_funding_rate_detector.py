"""Tests for FundingRateDetector."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.core.containers.raw_data import RawData
from signalflow.core.containers.raw_data_view import RawDataView
from signalflow.core.containers.signals import Signals
from signalflow.detector.funding_rate import FundingRateDetector


# ── helpers ──────────────────────────────────────────────────────────────────

BASE = datetime(2024, 1, 1)


def _make_perpetual_df(
    pairs: list[str],
    n_bars: int,
    funding_rates: list[float | None],
    interval_hours: int = 8,
) -> pl.DataFrame:
    """Build a synthetic perpetual OHLCV+funding_rate DataFrame.

    ``funding_rates`` is applied cyclically to the funding_rate column for each pair.
    """
    rows = []
    for pair in pairs:
        for i in range(n_bars):
            fr = funding_rates[i % len(funding_rates)]
            rows.append(
                {
                    "pair": pair,
                    "timestamp": BASE + timedelta(hours=i * interval_hours),
                    "open": 100.0 + i,
                    "high": 105.0 + i,
                    "low": 95.0 + i,
                    "close": 102.0 + i,
                    "volume": 1000.0,
                    "funding_rate": fr,
                }
            )
    return pl.DataFrame(rows)


def _make_view(df: pl.DataFrame, pairs: list[str]) -> RawDataView:
    raw = RawData(
        datetime_start=BASE,
        datetime_end=BASE + timedelta(days=30),
        pairs=pairs,
        data={"perpetual": df},
    )
    return RawDataView(raw=raw)


# ── tests ────────────────────────────────────────────────────────────────────


class TestFundingRateDetectorBasic:
    def test_returns_signals_instance(self):
        # 4 positive (32h) then 1 negative → should trigger
        rates = [0.01, 0.01, 0.01, 0.01, -0.005]
        df = _make_perpetual_df(["BTCUSDT"], n_bars=5, funding_rates=rates)
        view = _make_view(df, ["BTCUSDT"])
        detector = FundingRateDetector(min_positive_hours=24)
        result = detector.run(view)
        assert isinstance(result, Signals)

    def test_signal_on_positive_to_negative_transition(self):
        # 4 positive readings (0h, 8h, 16h, 24h) then negative at 32h
        # Gap from last non-positive (none before) ... first non-positive is at 32h
        # Actually: first reading is positive, so _non_pos_ts is null for it.
        # The first non-positive is at 32h itself. prev non-pos = shift(1) → null.
        # So _last_prev_non_pos_ts is null → _hours_gap is null → no signal.
        # We need a non-positive reading before the positive streak starts.
        rates = [-0.001, 0.01, 0.01, 0.01, 0.01, -0.005]
        df = _make_perpetual_df(["BTCUSDT"], n_bars=6, funding_rates=rates)
        view = _make_view(df, ["BTCUSDT"])
        detector = FundingRateDetector(min_positive_hours=24)
        result = detector.run(view)
        signals_df = result.value
        assert signals_df.height == 1
        assert signals_df["signal_type"][0] == "rise"
        assert signals_df["signal"][0] == 1

    def test_no_signal_when_positive_period_too_short(self):
        # Non-positive at 0h, positive at 8h, negative at 16h → gap = 16h < 24h
        rates = [-0.001, 0.01, -0.005]
        df = _make_perpetual_df(["BTCUSDT"], n_bars=3, funding_rates=rates)
        view = _make_view(df, ["BTCUSDT"])
        detector = FundingRateDetector(min_positive_hours=24)
        result = detector.run(view)
        assert result.value.height == 0

    def test_signal_respects_min_positive_hours(self):
        # Non-positive at 0h, positive at 8h, negative at 16h → gap = 16h
        # With threshold 8 → should trigger
        rates = [-0.001, 0.01, -0.005]
        df = _make_perpetual_df(["BTCUSDT"], n_bars=3, funding_rates=rates)
        view = _make_view(df, ["BTCUSDT"])
        detector = FundingRateDetector(min_positive_hours=8)
        result = detector.run(view)
        assert result.value.height == 1


class TestFundingRateDetectorEdgeCases:
    def test_empty_funding_data(self):
        df = _make_perpetual_df(["BTCUSDT"], n_bars=5, funding_rates=[None])
        view = _make_view(df, ["BTCUSDT"])
        detector = FundingRateDetector()
        result = detector.run(view)
        assert result.value.height == 0

    def test_all_positive_no_signal(self):
        rates = [0.01] * 10
        df = _make_perpetual_df(["BTCUSDT"], n_bars=10, funding_rates=rates)
        view = _make_view(df, ["BTCUSDT"])
        detector = FundingRateDetector()
        result = detector.run(view)
        assert result.value.height == 0

    def test_all_negative_no_signal(self):
        rates = [-0.01] * 10
        df = _make_perpetual_df(["BTCUSDT"], n_bars=10, funding_rates=rates)
        view = _make_view(df, ["BTCUSDT"])
        detector = FundingRateDetector()
        result = detector.run(view)
        # All negative in a row → gap between consecutive non-positive = 8h only
        assert result.value.height == 0


class TestFundingRateDetectorMultiPair:
    def test_signals_per_pair(self):
        # Build data where BTCUSDT triggers but ETHUSDT does not
        btc_rates = [-0.001, 0.01, 0.01, 0.01, 0.01, -0.005]
        eth_rates = [0.01] * 6  # all positive, no signal

        btc_df = _make_perpetual_df(["BTCUSDT"], n_bars=6, funding_rates=btc_rates)
        eth_df = _make_perpetual_df(["ETHUSDT"], n_bars=6, funding_rates=eth_rates)
        df = pl.concat([btc_df, eth_df])

        view = _make_view(df, ["BTCUSDT", "ETHUSDT"])
        detector = FundingRateDetector(min_positive_hours=24)
        result = detector.run(view)
        signals_df = result.value
        assert signals_df.height == 1
        assert signals_df["pair"][0] == "BTCUSDT"


class TestFundingRateDetectorSchema:
    def test_output_columns(self):
        rates = [-0.001, 0.01, 0.01, 0.01, 0.01, -0.005]
        df = _make_perpetual_df(["BTCUSDT"], n_bars=6, funding_rates=rates)
        view = _make_view(df, ["BTCUSDT"])
        detector = FundingRateDetector(min_positive_hours=24)
        result = detector.run(view)
        assert set(result.value.columns) == {"pair", "timestamp", "signal_type", "signal"}

    def test_empty_output_schema(self):
        df = _make_perpetual_df(["BTCUSDT"], n_bars=5, funding_rates=[None])
        view = _make_view(df, ["BTCUSDT"])
        detector = FundingRateDetector()
        result = detector.run(view)
        assert "pair" in result.value.columns
        assert "timestamp" in result.value.columns
        assert "signal_type" in result.value.columns


class TestFundingRateDetectorRegistry:
    def test_component_registered(self):
        from signalflow.core import default_registry, SfComponentType

        comp = default_registry.get(SfComponentType.DETECTOR, "funding/rate_transition")
        assert comp is not None

    def test_import_from_package(self):
        from signalflow.detector import FundingRateDetector as FR

        assert FR is FundingRateDetector
