"""Tests for BacktestRunner and OptimizedBacktestRunner."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest

from signalflow.core.containers.raw_data import RawData
from signalflow.core.containers.signals import Signals
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import SignalType
from signalflow.strategy.runner.backtest_runner import BacktestRunner
from signalflow.strategy.runner.optimized_backtest_runner import OptimizedBacktestRunner


TS = datetime(2024, 1, 1)
PAIRS = ["BTCUSDT", "ETHUSDT"]


def _make_ohlcv(n=10, pairs=None):
    pairs = pairs or PAIRS
    rows = []
    for pair in pairs:
        for i in range(n):
            rows.append(
                {
                    "pair": pair,
                    "timestamp": TS + timedelta(hours=i),
                    "open": 100.0 + i,
                    "high": 105.0 + i,
                    "low": 95.0 + i,
                    "close": 102.0 + i,
                    "volume": 1000.0,
                }
            )
    return pl.DataFrame(rows)


def _make_raw_data(n=10, pairs=None):
    return RawData(
        datetime_start=TS,
        datetime_end=TS + timedelta(hours=n),
        pairs=pairs or PAIRS,
        data={"spot": _make_ohlcv(n, pairs)},
    )


def _make_signals(n=3, pair="BTCUSDT"):
    return Signals(
        pl.DataFrame(
            {
                "pair": [pair] * n,
                "timestamp": [TS + timedelta(hours=i) for i in range(n)],
                "signal_type": [SignalType.RISE.value] * n,
                "signal": [1] * n,
                "probability": [0.9] * n,
            }
        )
    )


def _make_empty_signals():
    return Signals(pl.DataFrame())


# ── BacktestRunner._build_prices ────────────────────────────────────────────


class TestBuildPrices:
    def test_from_bar_data(self):
        runner = BacktestRunner()
        bar = pl.DataFrame(
            {"pair": ["BTCUSDT", "ETHUSDT"], "timestamp": [TS, TS], "close": [100.0, 50.0]}
        )
        prices = runner._build_prices(bar)
        assert prices == {"BTCUSDT": 100.0, "ETHUSDT": 50.0}

    def test_empty_bar(self):
        runner = BacktestRunner()
        bar = pl.DataFrame({"pair": [], "timestamp": [], "close": []}).cast(
            {"timestamp": pl.Datetime, "close": pl.Float64}
        )
        assert runner._build_prices(bar) == {}


# ── BacktestRunner._get_bar_signals ─────────────────────────────────────────


class TestGetBarSignals:
    def test_matching_ts(self):
        runner = BacktestRunner()
        sigs_df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [TS],
                "signal_type": ["rise"],
                "signal": [1],
            }
        )
        result = runner._get_bar_signals(sigs_df, TS)
        assert result.value.height == 1

    def test_no_match(self):
        runner = BacktestRunner()
        sigs_df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [TS + timedelta(hours=99)],
                "signal_type": ["rise"],
                "signal": [1],
            }
        )
        result = runner._get_bar_signals(sigs_df, TS)
        assert result.value.height == 0

    def test_empty_df(self):
        runner = BacktestRunner()
        result = runner._get_bar_signals(pl.DataFrame(), TS)
        assert result.value.height == 0


# ── BacktestRunner properties ───────────────────────────────────────────────


class TestBacktestRunnerProperties:
    def test_trades_empty_initially(self):
        runner = BacktestRunner()
        assert runner.trades == []

    def test_trades_df_empty(self):
        runner = BacktestRunner()
        assert runner.trades_df.height == 0

    def test_metrics_df_empty(self):
        runner = BacktestRunner()
        assert runner.metrics_df.height == 0

    def test_get_results_no_trades(self):
        runner = BacktestRunner()
        results = runner.get_results()
        assert results["total_trades"] == 0


# ── BacktestRunner.run ──────────────────────────────────────────────────────


class TestBacktestRunnerRun:
    def _make_runner(self, tmp_path):
        from signalflow.data.strategy_store import DuckDbStrategyStore
        from signalflow.strategy.broker.backtest import BacktestBroker
        from signalflow.strategy.broker.executor.virtual_spot import VirtualSpotExecutor

        store = DuckDbStrategyStore(str(tmp_path / "test.duckdb"))
        store.init()
        broker = BacktestBroker(
            executor=VirtualSpotExecutor(fee_rate=0.001, slippage_pct=0.0),
            store=store,
        )
        return BacktestRunner(broker=broker, initial_capital=10000.0), store

    def test_empty_data_returns_state(self, tmp_path):
        runner, store = self._make_runner(tmp_path)
        raw = RawData(
            datetime_start=TS,
            datetime_end=TS,
            pairs=PAIRS,
            data={"spot": pl.DataFrame({"pair": [], "timestamp": [], "close": [], "open": [], "high": [], "low": [], "volume": []}).cast({"timestamp": pl.Datetime, "close": pl.Float64, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "volume": pl.Float64})},
        )
        state = runner.run(raw, _make_empty_signals())
        assert state.portfolio.cash == 10000.0
        store.close()

    def test_creates_default_state(self, tmp_path):
        runner, store = self._make_runner(tmp_path)
        raw = _make_raw_data(n=3)
        state = runner.run(raw, _make_empty_signals())
        assert state.strategy_id == "backtest"
        assert state.last_ts is not None
        store.close()

    def test_run_no_signals_no_trades(self, tmp_path):
        runner, store = self._make_runner(tmp_path)
        raw = _make_raw_data(n=5)
        state = runner.run(raw, _make_empty_signals())
        assert len(runner.trades) == 0
        assert state.portfolio.cash == 10000.0
        store.close()

    def test_run_with_entry_rule(self, tmp_path):
        from signalflow.strategy.component.entry.fixed_size import FixedSizeEntryRule

        runner, store = self._make_runner(tmp_path)
        runner.entry_rules = [FixedSizeEntryRule(position_size=0.01, max_positions=1)]
        raw = _make_raw_data(n=5, pairs=["BTCUSDT"])
        sigs = _make_signals(n=1, pair="BTCUSDT")
        state = runner.run(raw, sigs)
        assert len(runner.trades) >= 1
        assert len(state.portfolio.open_positions()) >= 1
        store.close()

    def test_preserves_existing_state(self, tmp_path):
        runner, store = self._make_runner(tmp_path)
        state = StrategyState(strategy_id="custom")
        state.portfolio.cash = 5000.0
        raw = _make_raw_data(n=2)
        result = runner.run(raw, _make_empty_signals(), state=state)
        assert result.strategy_id == "custom"
        assert result.portfolio.cash == 5000.0
        store.close()


# ── OptimizedBacktestRunner ─────────────────────────────────────────────────


class TestOptimizedBacktestRunner:
    def test_build_price_lookup(self):
        runner = OptimizedBacktestRunner()
        df = _make_ohlcv(5, pairs=["BTCUSDT"])
        lookup = runner._build_price_lookup(df)
        assert len(lookup) == 5
        assert lookup[TS]["BTCUSDT"] == 102.0

    def test_build_signal_lookup(self):
        runner = OptimizedBacktestRunner()
        sigs = _make_signals(3).value
        lookup = runner._build_signal_lookup(sigs)
        assert len(lookup) == 3
        assert lookup[TS].height == 1

    def test_build_signal_lookup_empty(self):
        runner = OptimizedBacktestRunner()
        lookup = runner._build_signal_lookup(pl.DataFrame())
        assert lookup == {}

    def test_run_empty_data(self, tmp_path):
        from signalflow.data.strategy_store import DuckDbStrategyStore
        from signalflow.strategy.broker.backtest import BacktestBroker
        from signalflow.strategy.broker.executor.virtual_spot import VirtualSpotExecutor

        store = DuckDbStrategyStore(str(tmp_path / "opt.duckdb"))
        store.init()
        broker = BacktestBroker(
            executor=VirtualSpotExecutor(fee_rate=0.001, slippage_pct=0.0),
            store=store,
        )
        runner = OptimizedBacktestRunner(broker=broker, show_progress=False)
        raw = RawData(
            datetime_start=TS,
            datetime_end=TS,
            pairs=["BTCUSDT"],
            data={"spot": pl.DataFrame({"pair": [], "timestamp": [], "close": [], "open": [], "high": [], "low": [], "volume": []}).cast({"timestamp": pl.Datetime, "close": pl.Float64, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "volume": pl.Float64})},
        )
        state = runner.run(raw, _make_empty_signals())
        assert state.portfolio.cash == 10000.0
        store.close()

    def test_get_results_keys(self, tmp_path):
        from signalflow.data.strategy_store import DuckDbStrategyStore
        from signalflow.strategy.broker.backtest import BacktestBroker
        from signalflow.strategy.broker.executor.virtual_spot import VirtualSpotExecutor

        store = DuckDbStrategyStore(str(tmp_path / "opt2.duckdb"))
        store.init()
        broker = BacktestBroker(
            executor=VirtualSpotExecutor(fee_rate=0.001, slippage_pct=0.0),
            store=store,
        )
        runner = OptimizedBacktestRunner(broker=broker, show_progress=False)
        raw = _make_raw_data(n=3, pairs=["BTCUSDT"])
        runner.run(raw, _make_empty_signals())
        results = runner.get_results()
        assert "total_trades" in results
        assert "metrics_df" in results
        assert "trades_df" in results
        store.close()
