"""Tests for parallel backtest runners."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.core.containers.raw_data import RawData
from signalflow.core.containers.signals import Signals
from signalflow.strategy.runner import (
    create_backtest_runner,
    BacktestMode,
    IsolatedBalanceRunner,
    UnlimitedBalanceRunner,
    IsolatedResults,
    UnlimitedResults,
)
from signalflow.strategy.component.entry import FixedSizeEntryRule


def make_test_data(pairs: list[str], n_bars: int = 10) -> tuple[RawData, Signals]:
    """Create test data with multiple pairs."""
    rows = []
    signal_rows = []
    base_time = datetime(2024, 1, 1)

    for pair in pairs:
        base_price = 100.0 if pair == "BTCUSDT" else 50.0
        for i in range(n_bars):
            ts = base_time + timedelta(hours=i)
            price = base_price * (1 + 0.001 * i)  # Slight uptrend
            rows.append(
                {
                    "pair": pair,
                    "timestamp": ts,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1000.0,
                }
            )
            # Add signal every 3rd bar
            if i % 3 == 0:
                signal_rows.append(
                    {
                        "pair": pair,
                        "timestamp": ts,
                        "signal": 1.0,
                        "signal_type": "long",
                        "signal_strength": 0.8,
                    }
                )

    df = pl.DataFrame(rows)
    signals_df = pl.DataFrame(signal_rows)

    end_time = base_time + timedelta(hours=n_bars)
    raw_data = RawData(
        datetime_start=base_time,
        datetime_end=end_time,
        pairs=pairs,
        data={"spot": df},
    )

    return raw_data, Signals(signals_df)


class TestBacktestMode:
    def test_enum_values(self):
        assert BacktestMode.SEQUENTIAL.value == "sequential"
        assert BacktestMode.OPTIMIZED.value == "optimized"
        assert BacktestMode.ISOLATED.value == "isolated"
        assert BacktestMode.UNLIMITED.value == "unlimited"


class TestCreateBacktestRunner:
    def test_sequential_mode(self):
        from signalflow.strategy.runner import BacktestRunner
        from signalflow.strategy.broker import BacktestBroker
        from signalflow.strategy.broker.executor.virtual_spot import VirtualSpotExecutor
        from signalflow.data.strategy_store.memory import InMemoryStrategyStore

        broker = BacktestBroker(
            executor=VirtualSpotExecutor(),
            store=InMemoryStrategyStore(),
        )
        runner = create_backtest_runner(mode="sequential", broker=broker)
        assert isinstance(runner, BacktestRunner)

    def test_isolated_mode(self):
        runner = create_backtest_runner(mode="isolated", initial_capital=10000)
        assert isinstance(runner, IsolatedBalanceRunner)

    def test_unlimited_mode(self):
        runner = create_backtest_runner(mode="unlimited")
        assert isinstance(runner, UnlimitedBalanceRunner)

    def test_mode_enum(self):
        runner = create_backtest_runner(mode=BacktestMode.ISOLATED)
        assert isinstance(runner, IsolatedBalanceRunner)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            create_backtest_runner(mode="invalid")


class TestIsolatedBalanceRunner:
    def test_run_empty_data(self):
        runner = IsolatedBalanceRunner(initial_capital=10000)
        raw_data = RawData(
            datetime_start=datetime(2024, 1, 1),
            datetime_end=datetime(2024, 1, 2),
            pairs=[],
            data={"spot": pl.DataFrame()},
        )
        signals = Signals(pl.DataFrame())

        result = runner.run(raw_data, signals)

        assert isinstance(result, IsolatedResults)
        assert result.total_equity == 10000
        assert result.total_return == 0.0
        assert len(result.pair_results) == 0

    def test_run_single_pair(self):
        entry_rule = FixedSizeEntryRule(
            position_size=0.1,
            max_positions=1,
        )
        runner = IsolatedBalanceRunner(
            initial_capital=10000,
            entry_rules=[entry_rule],
            max_workers=1,
            show_progress=False,
        )

        raw_data, signals = make_test_data(["BTCUSDT"], n_bars=10)
        result = runner.run(raw_data, signals)

        assert isinstance(result, IsolatedResults)
        assert "BTCUSDT" in result.pair_results
        assert result.initial_capital == 10000

    def test_run_multiple_pairs(self):
        entry_rule = FixedSizeEntryRule(
            position_size=0.1,
            max_positions=1,
        )
        runner = IsolatedBalanceRunner(
            initial_capital=10000,
            entry_rules=[entry_rule],
            max_workers=2,
            show_progress=False,
        )

        raw_data, signals = make_test_data(["BTCUSDT", "ETHUSDT"], n_bars=10)
        result = runner.run(raw_data, signals)

        assert isinstance(result, IsolatedResults)
        assert len(result.pair_results) == 2
        assert "BTCUSDT" in result.pair_results
        assert "ETHUSDT" in result.pair_results
        # Each pair should get half the capital
        for pair_result in result.pair_results.values():
            assert pair_result.initial_capital == 5000

    def test_aggregation(self):
        runner = IsolatedBalanceRunner(
            initial_capital=10000,
            max_workers=1,
            show_progress=False,
        )

        raw_data, signals = make_test_data(["BTCUSDT", "ETHUSDT"], n_bars=5)
        result = runner.run(raw_data, signals)

        # Total equity should be sum of pair equities
        expected_total = sum(r.final_equity for r in result.pair_results.values())
        assert result.total_equity == expected_total


class TestUnlimitedBalanceRunner:
    def test_run_empty_data(self):
        runner = UnlimitedBalanceRunner()
        raw_data = RawData(
            datetime_start=datetime(2024, 1, 1),
            datetime_end=datetime(2024, 1, 2),
            pairs=[],
            data={"spot": pl.DataFrame()},
        )
        signals = Signals(pl.DataFrame())

        result = runner.run(raw_data, signals)

        assert isinstance(result, UnlimitedResults)
        assert result.total_signals == 0
        assert result.executed_trades == 0

    def test_run_with_signals(self):
        entry_rule = FixedSizeEntryRule(
            position_size=1.0,
            max_positions=10,
        )
        runner = UnlimitedBalanceRunner(
            entry_rules=[entry_rule],
            position_size=1.0,
            take_profit_pct=0.02,
            stop_loss_pct=0.01,
            show_progress=False,
        )

        raw_data, signals = make_test_data(["BTCUSDT"], n_bars=20)
        result = runner.run(raw_data, signals)

        assert isinstance(result, UnlimitedResults)
        assert result.total_signals > 0

    def test_tp_sl_exit(self):
        entry_rule = FixedSizeEntryRule(
            position_size=1.0,
            max_positions=10,
        )
        runner = UnlimitedBalanceRunner(
            entry_rules=[entry_rule],
            take_profit_pct=0.001,  # Very tight TP
            stop_loss_pct=0.001,  # Very tight SL
            show_progress=False,
        )

        raw_data, signals = make_test_data(["BTCUSDT"], n_bars=20)
        result = runner.run(raw_data, signals)

        # With tight TP/SL, should have some exits
        assert isinstance(result, UnlimitedResults)

    def test_win_rate_calculation(self):
        entry_rule = FixedSizeEntryRule(
            position_size=1.0,
            max_positions=10,
        )
        runner = UnlimitedBalanceRunner(
            entry_rules=[entry_rule],
            take_profit_pct=0.02,
            stop_loss_pct=0.01,
            show_progress=False,
        )

        raw_data, signals = make_test_data(["BTCUSDT"], n_bars=50)
        result = runner.run(raw_data, signals)

        # Win rate should be between 0 and 1
        assert 0.0 <= result.win_rate <= 1.0


class TestIsolatedResults:
    def test_pair_metrics_df(self):
        from signalflow.strategy.runner.parallel.results import PairResult, IsolatedResults

        pair_results = {
            "BTCUSDT": PairResult(
                pair="BTCUSDT",
                trades=[],
                final_equity=5500,
                final_cash=5500,
                positions=[],
                metrics_history=[],
                initial_capital=5000,
            ),
            "ETHUSDT": PairResult(
                pair="ETHUSDT",
                trades=[],
                final_equity=4800,
                final_cash=4800,
                positions=[],
                metrics_history=[],
                initial_capital=5000,
            ),
        }

        result = IsolatedResults(
            total_equity=10300,
            total_return=0.03,
            initial_capital=10000,
            pair_results=pair_results,
        )

        df = result.pair_metrics_df()
        assert df.height == 2
        assert "pair" in df.columns
        assert "total_return" in df.columns


class TestUnlimitedResults:
    def test_by_pair_empty(self):
        result = UnlimitedResults(
            trades_df=pl.DataFrame(),
            total_signals=0,
            executed_trades=0,
            win_rate=0.0,
            avg_return=0.0,
        )

        df = result.by_pair()
        assert df.height == 0

    def test_loss_rate(self):
        result = UnlimitedResults(
            trades_df=pl.DataFrame(),
            total_signals=100,
            executed_trades=50,
            win_rate=0.6,
            avg_return=0.01,
        )

        assert result.loss_rate == 0.4
