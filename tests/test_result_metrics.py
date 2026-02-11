"""Tests for StrategyMainResult and StrategyPairResult metrics."""

from datetime import datetime, timedelta

import polars as pl

from signalflow.core import RawData
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.containers.position import Position
from signalflow.core.enums import PositionType
from signalflow.analytic.strategy.result_metrics import StrategyMainResult, StrategyPairResult


TS = datetime(2024, 1, 1)


def _make_price_df(n=100, pair="BTCUSDT"):
    rows = [
        {
            "pair": pair,
            "timestamp": TS + timedelta(minutes=i),
            "open": 100.0 + i * 0.1,
            "high": 101.0 + i * 0.1,
            "low": 99.0 + i * 0.1,
            "close": 100.0 + i * 0.1,
            "volume": 1000.0,
        }
        for i in range(n)
    ]
    return pl.DataFrame(rows)


def _make_raw_data(n=100, pairs=None):
    if pairs is None:
        pairs = ["BTCUSDT"]
    dfs = [_make_price_df(n, pair) for pair in pairs]
    df = pl.concat(dfs)
    return RawData(
        datetime_start=TS,
        datetime_end=TS + timedelta(minutes=n),
        pairs=pairs,
        data={"spot": df},
    )


def _make_metrics_df(n=50):
    """Create a metrics DataFrame similar to backtest output."""
    rows = []
    equity = 10000.0
    for i in range(n):
        equity = equity * (1 + 0.001 * (1 if i % 3 == 0 else -0.5))
        rows.append(
            {
                "timestamp": TS + timedelta(minutes=i),
                "total_return": (equity / 10000.0) - 1.0,
                "equity": equity,
                "cash": equity * 0.5,
                "open_positions": i % 5,
                "closed_positions": i // 5,
                "current_drawdown": max(0, (10000 - equity) / 10000),
                "capital_utilization": 0.5,
            }
        )
    return pl.DataFrame(rows)


def _make_state_with_positions(pair="BTCUSDT"):
    """Create a strategy state with some positions."""
    state = StrategyState(strategy_id="test")
    state.portfolio.cash = 9000.0

    # Open position
    pos1 = Position(
        id="pos1",
        pair=pair,
        position_type=PositionType.LONG,
        entry_price=100.0,
        last_price=110.0,
        qty=1.0,
        entry_time=TS + timedelta(minutes=10),
        last_time=TS + timedelta(minutes=20),
    )
    state.portfolio.positions["pos1"] = pos1

    # Closed position
    pos2 = Position(
        id="pos2",
        pair=pair,
        position_type=PositionType.LONG,
        entry_price=100.0,
        last_price=105.0,
        qty=0.5,
        entry_time=TS + timedelta(minutes=30),
        last_time=TS + timedelta(minutes=50),
        is_closed=True,
        realized_pnl=2.5,
    )
    state.portfolio.positions["pos2"] = pos2

    return state


# ── StrategyMainResult Tests ────────────────────────────────────────────────


class TestStrategyMainResultCompute:
    def test_compute_returns_empty(self):
        metric = StrategyMainResult()
        state = StrategyState(strategy_id="test")
        result = metric.compute(state, prices={})
        assert result == {}


class TestStrategyMainResultPlot:
    def test_plot_with_metrics_df(self):
        metric = StrategyMainResult()
        metrics_df = _make_metrics_df(n=50)
        results = {
            "metrics_df": metrics_df,
            "final_return": 0.05,
            "initial_capital": 10000.0,
        }
        figs = metric.plot(results)

        assert figs is not None
        assert len(figs) >= 1  # At least main figure

    def test_plot_no_metrics_df(self):
        metric = StrategyMainResult()
        results = {}
        figs = metric.plot(results)
        assert figs is None

    def test_plot_empty_metrics_df(self):
        metric = StrategyMainResult()
        results = {"metrics_df": pl.DataFrame()}
        figs = metric.plot(results)
        assert figs is None

    def test_main_figure_has_traces(self):
        metric = StrategyMainResult()
        metrics_df = _make_metrics_df(n=50)
        results = {
            "metrics_df": metrics_df,
            "final_return": 0.05,
            "initial_capital": 10000.0,
        }
        figs = metric.plot(results)
        main_fig = figs[0]

        # Should have traces for return, positions, allocation
        assert len(main_fig.data) >= 3

    def test_detailed_figure_with_drawdown(self):
        metric = StrategyMainResult()
        metrics_df = _make_metrics_df(n=50)
        results = {
            "metrics_df": metrics_df,
            "final_return": 0.05,
            "max_drawdown": 0.02,
        }
        figs = metric.plot(results)

        # Should have 2 figures (main + detailed) when drawdown column exists
        assert len(figs) == 2

    def test_plot_without_optional_columns(self):
        metric = StrategyMainResult()
        # Minimal metrics_df
        metrics_df = pl.DataFrame(
            {
                "timestamp": [TS + timedelta(minutes=i) for i in range(10)],
                "total_return": [0.01 * i for i in range(10)],
            }
        )
        results = {"metrics_df": metrics_df, "final_return": 0.09}
        figs = metric.plot(results)

        assert figs is not None
        assert len(figs) >= 1


# ── StrategyPairResult Tests ────────────────────────────────────────────────


class TestStrategyPairResultCompute:
    def test_compute_returns_empty(self):
        metric = StrategyPairResult(pairs=["BTCUSDT"])
        state = StrategyState(strategy_id="test")
        result = metric.compute(state, prices={})
        assert result == {}


class TestStrategyPairResultPlot:
    def test_plot_no_pairs(self):
        metric = StrategyPairResult(pairs=[])
        results = {}
        figs = metric.plot(results)
        assert figs is None

    def test_plot_with_state_and_raw_data(self):
        metric = StrategyPairResult(pairs=["BTCUSDT"])
        state = _make_state_with_positions(pair="BTCUSDT")
        raw = _make_raw_data(n=100, pairs=["BTCUSDT"])
        results = {}

        figs = metric.plot(results, state=state, raw_data=raw)

        assert figs is not None
        assert len(figs) == 1

    def test_plot_multiple_pairs(self):
        metric = StrategyPairResult(pairs=["BTCUSDT", "ETHUSDT"])
        state = _make_state_with_positions(pair="BTCUSDT")

        # Add ETHUSDT position
        pos = Position(
            id="pos_eth",
            pair="ETHUSDT",
            position_type=PositionType.LONG,
            entry_price=50.0,
            last_price=55.0,
            qty=2.0,
            entry_time=TS + timedelta(minutes=15),
            last_time=TS + timedelta(minutes=25),
        )
        state.portfolio.positions["pos_eth"] = pos

        raw = _make_raw_data(n=100, pairs=["BTCUSDT", "ETHUSDT"])
        results = {}

        figs = metric.plot(results, state=state, raw_data=raw)

        assert figs is not None
        assert len(figs) == 2

    def test_plot_no_raw_data(self):
        metric = StrategyPairResult(pairs=["BTCUSDT"])
        state = _make_state_with_positions()
        results = {}

        figs = metric.plot(results, state=state, raw_data=None)

        # Should return empty list or handle gracefully
        assert figs is not None
        assert len(figs) == 0

    def test_plot_no_state(self):
        metric = StrategyPairResult(pairs=["BTCUSDT"])
        raw = _make_raw_data(n=100, pairs=["BTCUSDT"])
        results = {}

        figs = metric.plot(results, state=None, raw_data=raw)

        # Should still plot price, just no markers
        assert figs is not None
        assert len(figs) == 1

    def test_figure_has_price_trace(self):
        metric = StrategyPairResult(pairs=["BTCUSDT"])
        raw = _make_raw_data(n=100, pairs=["BTCUSDT"])
        results = {}

        figs = metric.plot(results, state=None, raw_data=raw)
        fig = figs[0]

        # Should have price line
        assert len(fig.data) >= 1
        assert fig.data[0].name == "Price"

    def test_figure_has_entry_exit_markers(self):
        metric = StrategyPairResult(pairs=["BTCUSDT"])
        state = _make_state_with_positions(pair="BTCUSDT")
        raw = _make_raw_data(n=100, pairs=["BTCUSDT"])
        results = {}

        figs = metric.plot(results, state=state, raw_data=raw)
        fig = figs[0]

        trace_names = [t.name for t in fig.data if t.name]
        assert "Entry" in trace_names
        # Exit marker should be present for closed position
        assert "Exit" in trace_names

    def test_figure_layout_properties(self):
        metric = StrategyPairResult(pairs=["BTCUSDT"], height=800, template="plotly_white")
        raw = _make_raw_data(n=100, pairs=["BTCUSDT"])
        results = {}

        figs = metric.plot(results, state=None, raw_data=raw)
        fig = figs[0]

        assert fig.layout.height == 800
        # Template is applied - just verify the figure was created with our settings
        assert fig.layout.template is not None

    def test_missing_pair_in_data(self):
        metric = StrategyPairResult(pairs=["BTCUSDT", "XRPUSDT"])
        raw = _make_raw_data(n=100, pairs=["BTCUSDT"])  # No XRPUSDT
        results = {}

        figs = metric.plot(results, state=None, raw_data=raw)

        # Should only plot BTCUSDT
        assert len(figs) == 1

    def test_custom_column_names(self):
        metric = StrategyPairResult(
            pairs=["BTCUSDT"],
            price_col="close",
            ts_col="timestamp",
            pair_col="pair",
        )
        raw = _make_raw_data(n=100, pairs=["BTCUSDT"])
        results = {}

        figs = metric.plot(results, state=None, raw_data=raw)

        assert figs is not None
        assert len(figs) == 1


class TestStrategyPairResultHelpers:
    def test_normalize_timeseries_datetime(self):
        metric = StrategyPairResult(pairs=["BTCUSDT"])
        df = pl.DataFrame(
            {
                "timestamp": [TS + timedelta(minutes=i) for i in range(10)],
                "close": [100.0 + i for i in range(10)],
            }
        )

        ts_dt, ts_s, price = metric._normalize_timeseries(df=df)

        assert len(ts_dt) == 10
        assert len(ts_s) == 10
        assert len(price) == 10
        assert all(isinstance(t, int) for t in ts_s)

    def test_normalize_timeseries_epoch(self):
        metric = StrategyPairResult(pairs=["BTCUSDT"])
        base_epoch = int(TS.timestamp())
        df = pl.DataFrame(
            {
                "timestamp": [base_epoch + i * 60 for i in range(10)],
                "close": [100.0 + i for i in range(10)],
            }
        )

        ts_dt, ts_s, price = metric._normalize_timeseries(df=df)

        assert len(ts_dt) == 10
        assert ts_s[0] == base_epoch

    def test_nearest_price(self):
        metric = StrategyPairResult(pairs=["BTCUSDT"])
        ts_s = [100, 200, 300, 400, 500]
        price = [10.0, 20.0, 30.0, 40.0, 50.0]

        # Exact match
        assert metric._nearest_price(epoch_s=200, ts_s=ts_s, price=price) == 20.0

        # Between values - should get nearest
        assert metric._nearest_price(epoch_s=250, ts_s=ts_s, price=price) in [20.0, 30.0]

        # Before first
        assert metric._nearest_price(epoch_s=50, ts_s=ts_s, price=price) == 10.0

        # After last
        assert metric._nearest_price(epoch_s=600, ts_s=ts_s, price=price) == 50.0

    def test_extract_trades(self):
        metric = StrategyPairResult(pairs=["BTCUSDT"])
        state = _make_state_with_positions(pair="BTCUSDT")

        trades = metric._extract_trades(state=state, pair="BTCUSDT")

        assert len(trades) == 2  # 2 positions for BTCUSDT
        assert all("id" in t for t in trades)
        assert all("entry_ts" in t for t in trades)
        assert all("size" in t for t in trades)

    def test_extract_trades_wrong_pair(self):
        metric = StrategyPairResult(pairs=["ETHUSDT"])
        state = _make_state_with_positions(pair="BTCUSDT")

        trades = metric._extract_trades(state=state, pair="ETHUSDT")

        assert len(trades) == 0
