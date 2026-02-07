"""Tests for parallel runner results aggregation."""

from datetime import datetime

import polars as pl
import pytest

from signalflow.strategy.runner.parallel.results import (
    PairResult,
    IsolatedResults,
    UnlimitedResults,
)
from signalflow.core.containers.trade import Trade
from signalflow.core.containers.position import Position
from signalflow.core.enums import PositionType


TS = datetime(2024, 1, 1)


def _make_trade(tid, pair, pnl=10.0):
    return Trade(
        id=tid,
        position_id=f"pos_{tid}",
        pair=pair,
        side="BUY",
        ts=TS,
        price=100.0,
        qty=1.0,
        fee=0.1,
        meta={"pnl": pnl},
    )


def _make_position(pid, pair="BTCUSDT"):
    return Position(
        id=pid,
        pair=pair,
        position_type=PositionType.LONG,
        entry_price=100.0,
        qty=1.0,
    )


class TestPairResult:
    def test_total_return(self):
        result = PairResult(
            pair="BTCUSDT",
            trades=[],
            final_equity=11000.0,
            final_cash=11000.0,
            positions=[],
            metrics_history=[],
            initial_capital=10000.0,
        )
        assert result.total_return == pytest.approx(0.1)

    def test_total_return_zero_capital(self):
        result = PairResult(
            pair="BTCUSDT",
            trades=[],
            final_equity=0.0,
            final_cash=0.0,
            positions=[],
            metrics_history=[],
            initial_capital=0.0,
        )
        assert result.total_return == 0.0

    def test_trade_count(self):
        trades = [_make_trade("t1", "BTCUSDT"), _make_trade("t2", "BTCUSDT")]
        result = PairResult(
            pair="BTCUSDT",
            trades=trades,
            final_equity=11000.0,
            final_cash=11000.0,
            positions=[],
            metrics_history=[],
            initial_capital=10000.0,
        )
        assert result.trade_count == 2

    def test_trades_df(self):
        trades = [_make_trade("t1", "BTCUSDT")]
        result = PairResult(
            pair="BTCUSDT",
            trades=trades,
            final_equity=11000.0,
            final_cash=11000.0,
            positions=[],
            metrics_history=[],
            initial_capital=10000.0,
        )
        df = result.trades_df()
        assert isinstance(df, pl.DataFrame)
        assert df.height == 1

    def test_metrics_df(self):
        metrics = [{"equity": 10000.0, "timestamp": TS}]
        result = PairResult(
            pair="BTCUSDT",
            trades=[],
            final_equity=10000.0,
            final_cash=10000.0,
            positions=[],
            metrics_history=metrics,
            initial_capital=10000.0,
        )
        df = result.metrics_df()
        assert isinstance(df, pl.DataFrame)
        assert df.height == 1

    def test_metrics_df_empty(self):
        result = PairResult(
            pair="BTCUSDT",
            trades=[],
            final_equity=10000.0,
            final_cash=10000.0,
            positions=[],
            metrics_history=[],
            initial_capital=10000.0,
        )
        df = result.metrics_df()
        assert df.height == 0


class TestIsolatedResults:
    def test_all_trades(self):
        pr1 = PairResult(
            pair="BTCUSDT",
            trades=[_make_trade("t1", "BTCUSDT")],
            final_equity=11000.0,
            final_cash=11000.0,
            positions=[],
            metrics_history=[],
            initial_capital=10000.0,
        )
        pr2 = PairResult(
            pair="ETHUSDT",
            trades=[_make_trade("t2", "ETHUSDT")],
            final_equity=11000.0,
            final_cash=11000.0,
            positions=[],
            metrics_history=[],
            initial_capital=10000.0,
        )
        results = IsolatedResults(
            total_equity=22000.0,
            total_return=0.1,
            initial_capital=20000.0,
            pair_results={"BTCUSDT": pr1, "ETHUSDT": pr2},
        )
        assert len(results.all_trades) == 2

    def test_total_trades(self):
        pr1 = PairResult(
            pair="BTCUSDT",
            trades=[_make_trade("t1", "BTCUSDT"), _make_trade("t2", "BTCUSDT")],
            final_equity=11000.0,
            final_cash=11000.0,
            positions=[],
            metrics_history=[],
            initial_capital=10000.0,
        )
        results = IsolatedResults(
            total_equity=11000.0,
            total_return=0.1,
            initial_capital=10000.0,
            pair_results={"BTCUSDT": pr1},
        )
        assert results.total_trades == 2

    def test_trades_df(self):
        pr1 = PairResult(
            pair="BTCUSDT",
            trades=[_make_trade("t1", "BTCUSDT")],
            final_equity=11000.0,
            final_cash=11000.0,
            positions=[],
            metrics_history=[],
            initial_capital=10000.0,
        )
        results = IsolatedResults(
            total_equity=11000.0,
            total_return=0.1,
            initial_capital=10000.0,
            pair_results={"BTCUSDT": pr1},
        )
        df = results.trades_df()
        assert isinstance(df, pl.DataFrame)
        assert df.height == 1

    def test_pair_metrics_df(self):
        pr1 = PairResult(
            pair="BTCUSDT",
            trades=[],
            final_equity=11000.0,
            final_cash=11000.0,
            positions=[],
            metrics_history=[],
            initial_capital=10000.0,
        )
        results = IsolatedResults(
            total_equity=11000.0,
            total_return=0.1,
            initial_capital=10000.0,
            pair_results={"BTCUSDT": pr1},
        )
        df = results.pair_metrics_df()
        assert isinstance(df, pl.DataFrame)
        assert df.height == 1
        assert "pair" in df.columns


class TestUnlimitedResults:
    def test_loss_rate(self):
        trades_df = pl.DataFrame({"pair": [], "pnl": []})
        results = UnlimitedResults(
            trades_df=trades_df,
            total_signals=10,
            executed_trades=5,
            win_rate=0.6,
            avg_return=0.05,
        )
        assert results.loss_rate == pytest.approx(0.4)

    def test_by_pair(self):
        trades = pl.DataFrame(
            {
                "pair": ["BTCUSDT", "BTCUSDT", "ETHUSDT"],
                "pnl": [10.0, 20.0, -5.0],
                "return_pct": [0.1, 0.2, -0.05],
            }
        )
        results = UnlimitedResults(
            trades_df=trades,
            total_signals=3,
            executed_trades=3,
            win_rate=0.67,
            avg_return=0.08,
        )
        by_pair = results.by_pair()
        assert by_pair.height == 2

    def test_by_pair_empty(self):
        trades = pl.DataFrame({"pair": [], "pnl": []})
        results = UnlimitedResults(
            trades_df=trades,
            total_signals=0,
            executed_trades=0,
            win_rate=0.0,
            avg_return=0.0,
        )
        by_pair = results.by_pair()
        assert by_pair.height == 0

    def test_by_side(self):
        trades = pl.DataFrame(
            {
                "pair": ["BTCUSDT", "ETHUSDT"],
                "side": ["LONG", "SHORT"],
                "pnl": [10.0, -5.0],
            }
        )
        results = UnlimitedResults(
            trades_df=trades,
            total_signals=2,
            executed_trades=2,
            win_rate=0.5,
            avg_return=0.025,
        )
        by_side = results.by_side()
        assert by_side.height == 2

    def test_by_side_empty(self):
        trades = pl.DataFrame({"pair": [], "pnl": []})
        results = UnlimitedResults(
            trades_df=trades,
            total_signals=0,
            executed_trades=0,
            win_rate=0.0,
            avg_return=0.0,
        )
        by_side = results.by_side()
        assert by_side.height == 0
