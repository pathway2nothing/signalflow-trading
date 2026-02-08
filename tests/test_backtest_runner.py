"""Tests for OptimizedBacktestRunner."""

from datetime import datetime, timedelta

import polars as pl

from signalflow.core.containers.raw_data import RawData
from signalflow.core.containers.signals import Signals
from signalflow.core.enums import SignalType
from signalflow.strategy.runner import OptimizedBacktestRunner


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
            data={
                "spot": pl.DataFrame(
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
            },
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
