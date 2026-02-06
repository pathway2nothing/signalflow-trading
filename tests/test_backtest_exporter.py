"""Tests for BacktestExporter."""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from signalflow.core.containers.position import Position
from signalflow.core.containers.signals import Signals
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import PositionType
from signalflow.strategy.exporter.parquet_exporter import BacktestExporter


TS = datetime(2024, 1, 1)
TS2 = datetime(2024, 1, 1, 0, 1)


def _make_state(cash: float = 10000.0) -> StrategyState:
    state = StrategyState(strategy_id="test")
    state.portfolio.cash = cash
    return state


def _signals(rows: list[dict]) -> Signals:
    return Signals(pl.DataFrame(rows))


class TestBacktestExporterBar:
    def test_export_bar_with_signals(self):
        exporter = BacktestExporter()
        signals = _signals([
            {"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.8},
        ])
        metrics = {"equity": 10000.0, "max_drawdown": 0.05}
        state = _make_state()

        exporter.export_bar(TS, signals, metrics, state)

        assert exporter.bar_count == 1

    def test_export_bar_without_signals(self):
        exporter = BacktestExporter()
        signals = Signals(pl.DataFrame())
        metrics = {"equity": 10000.0, "max_drawdown": 0.05}
        state = _make_state()

        exporter.export_bar(TS, signals, metrics, state)

        assert exporter.bar_count == 1

    def test_export_multiple_signals_per_bar(self):
        exporter = BacktestExporter()
        signals = _signals([
            {"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.8},
            {"pair": "ETHUSDT", "timestamp": TS, "signal_type": "fall", "signal": -1, "probability": 0.7},
        ])
        metrics = {"equity": 10000.0}
        state = _make_state()

        exporter.export_bar(TS, signals, metrics, state)

        assert exporter.bar_count == 2  # One record per signal


class TestBacktestExporterTrade:
    def test_export_trade(self):
        exporter = BacktestExporter()
        trade_data = {
            "position_id": "p1",
            "pair": "BTCUSDT",
            "entry_price": 50000.0,
            "exit_price": 51000.0,
            "realized_pnl": 100.0,
        }

        exporter.export_trade(trade_data)

        assert exporter.trade_count == 1

    def test_export_position_close(self):
        exporter = BacktestExporter()
        pos = Position(
            id="p1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            qty=0.1,
            entry_time=TS,
            signal_strength=0.8,
            realized_pnl=100.0,
            fees_paid=5.0,
            meta={"signal_type": "rise", "model_confidence": 0.85},
        )

        exporter.export_position_close(pos, TS2, 51000.0, "take_profit")

        assert exporter.trade_count == 1


class TestBacktestExporterFinalize:
    def test_finalize_creates_files(self, tmp_path: Path):
        exporter = BacktestExporter()

        # Export some data
        signals = _signals([
            {"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.8},
        ])
        exporter.export_bar(TS, signals, {"equity": 10000.0}, _make_state())
        exporter.export_trade({"position_id": "p1", "pair": "BTCUSDT", "pnl": 100.0})

        # Finalize
        exporter.finalize(tmp_path)

        # Check files created
        assert (tmp_path / "bars.parquet").exists()
        assert (tmp_path / "trades.parquet").exists()

    def test_finalize_parquet_readable(self, tmp_path: Path):
        exporter = BacktestExporter()

        signals = _signals([
            {"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.8},
        ])
        exporter.export_bar(TS, signals, {"equity": 10000.0, "max_drawdown": 0.05}, _make_state())

        exporter.finalize(tmp_path)

        # Read back
        bars_df = pl.read_parquet(tmp_path / "bars.parquet")
        assert bars_df.height == 1
        assert "pair" in bars_df.columns
        assert "metric_equity" in bars_df.columns
        assert "metric_max_drawdown" in bars_df.columns
        assert "probability" in bars_df.columns

    def test_finalize_creates_directory(self, tmp_path: Path):
        exporter = BacktestExporter()
        signals = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.8}])
        exporter.export_bar(TS, signals, {"equity": 10000.0}, _make_state())

        nested_path = tmp_path / "nested" / "output"
        exporter.finalize(nested_path)

        assert nested_path.exists()
        assert (nested_path / "bars.parquet").exists()

    def test_finalize_empty_no_files(self, tmp_path: Path):
        exporter = BacktestExporter()

        exporter.finalize(tmp_path)

        assert not (tmp_path / "bars.parquet").exists()
        assert not (tmp_path / "trades.parquet").exists()


class TestBacktestExporterReset:
    def test_reset_clears_data(self):
        exporter = BacktestExporter()
        signals = _signals([{"pair": "BTCUSDT", "timestamp": TS, "signal_type": "rise", "signal": 1, "probability": 0.8}])
        exporter.export_bar(TS, signals, {"equity": 10000.0}, _make_state())
        exporter.export_trade({"position_id": "p1"})

        assert exporter.bar_count == 1
        assert exporter.trade_count == 1

        exporter.reset()

        assert exporter.bar_count == 0
        assert exporter.trade_count == 0


class TestBacktestExporterTradeDetails:
    def test_export_position_close_includes_all_fields(self, tmp_path: Path):
        exporter = BacktestExporter()
        pos = Position(
            id="p1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            qty=0.1,
            entry_time=TS,
            signal_strength=0.8,
            realized_pnl=100.0,
            fees_paid=5.0,
            meta={"signal_type": "rise", "model_confidence": 0.85},
        )

        exporter.export_position_close(pos, TS2, 51000.0, "take_profit")
        exporter.finalize(tmp_path)

        trades_df = pl.read_parquet(tmp_path / "trades.parquet")
        row = trades_df.to_dicts()[0]

        assert row["position_id"] == "p1"
        assert row["pair"] == "BTCUSDT"
        assert row["position_type"] == "long"  # PositionType.LONG.value is lowercase
        assert row["entry_price"] == 50000.0
        assert row["exit_price"] == 51000.0
        assert row["qty"] == 0.1
        assert row["realized_pnl"] == 100.0
        assert row["fees_paid"] == 5.0
        assert row["signal_strength"] == 0.8
        assert row["exit_reason"] == "take_profit"
        assert row["entry_signal_type"] == "rise"
        assert row["model_confidence"] == 0.85
