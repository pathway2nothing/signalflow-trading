"""Tests for UI integration features: schema introspection, JSON serialization,
progress callbacks, and builder round-trip serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Event
from typing import Any, ClassVar

import polars as pl
import pytest

from signalflow.core.enums import SfComponentType
from signalflow.core.registry import SignalFlowRegistry


# ─── Fixtures ───────────────────────────────────────────────────────────

TS = datetime(2024, 1, 1)
PAIRS = ["BTCUSDT", "ETHUSDT"]


@dataclass
class _FakeDetector:
    """Fake detector for testing schema introspection."""

    component_type: ClassVar[SfComponentType] = SfComponentType.DETECTOR

    fast_period: int = 20
    slow_period: int = 50
    source_col: str = "close"
    threshold: float = 0.5


@dataclass
class _FakeFeature:
    """Compute a rolling average."""

    component_type: ClassVar[SfComponentType] = SfComponentType.FEATURE
    requires: ClassVar[list[str]] = ["{source_col}"]
    outputs: ClassVar[list[str]] = ["rolling_{period}"]

    source_col: str = "close"
    period: int = 14
    pair_col: str = "pair"
    ts_col: str = "timestamp"


def _make_ohlcv(n: int = 10, pairs: list[str] | None = None) -> pl.DataFrame:
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


# ─── Registry Schema Tests ──────────────────────────────────────────────


class TestRegistryGetSchema:
    def test_returns_parameters(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.DETECTOR, "test/fake", _FakeDetector)
        schema = reg.get_schema(SfComponentType.DETECTOR, "test/fake")

        assert schema["name"] == "test/fake"
        assert schema["class_name"] == "_FakeDetector"
        assert schema["component_type"] == SfComponentType.DETECTOR.value
        assert schema["description"] == "Fake detector for testing schema introspection."

        param_names = {p["name"] for p in schema["parameters"]}
        assert "fast_period" in param_names
        assert "slow_period" in param_names
        assert "source_col" in param_names
        assert "threshold" in param_names

    def test_excludes_base_fields(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.FEATURE, "test/feat", _FakeFeature)
        schema = reg.get_schema(SfComponentType.FEATURE, "test/feat")

        param_names = {p["name"] for p in schema["parameters"]}
        # Base fields should be filtered out
        assert "pair_col" not in param_names
        assert "ts_col" not in param_names
        # User-facing params should be present
        assert "source_col" in param_names
        assert "period" in param_names

    def test_includes_requires_outputs(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.FEATURE, "test/feat", _FakeFeature)
        schema = reg.get_schema(SfComponentType.FEATURE, "test/feat")

        assert schema["requires"] == ["{source_col}"]
        assert schema["outputs"] == ["rolling_{period}"]

    def test_default_and_required_flags(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.DETECTOR, "test/fake", _FakeDetector)
        schema = reg.get_schema(SfComponentType.DETECTOR, "test/fake")

        for p in schema["parameters"]:
            if p["name"] == "fast_period":
                assert p["default"] == 20
                assert p["required"] is False
            if p["name"] == "threshold":
                assert p["default"] == 0.5
                assert p["required"] is False

    def test_schema_is_json_serializable(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.DETECTOR, "test/fake", _FakeDetector)
        schema = reg.get_schema(SfComponentType.DETECTOR, "test/fake")

        # Should not raise
        result = json.dumps(schema)
        assert isinstance(result, str)

    def test_missing_component_raises(self):
        reg = SignalFlowRegistry()
        with pytest.raises(KeyError):
            reg.get_schema(SfComponentType.DETECTOR, "nonexistent")

    def test_non_dataclass_returns_empty_params(self):
        class PlainClass:
            """No dataclass."""

            pass

        reg = SignalFlowRegistry()
        reg.register(SfComponentType.DETECTOR, "plain", PlainClass)
        schema = reg.get_schema(SfComponentType.DETECTOR, "plain")

        assert schema["parameters"] == []
        assert schema["description"] == "No dataclass."


class TestRegistryExportSchemas:
    def test_export_all(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.DETECTOR, "det1", _FakeDetector)
        reg.register(SfComponentType.FEATURE, "feat1", _FakeFeature)

        result = reg.export_schemas()

        det_key = SfComponentType.DETECTOR.value
        feat_key = SfComponentType.FEATURE.value
        assert det_key in result
        assert feat_key in result
        assert len(result[det_key]) == 1
        assert result[det_key][0]["name"] == "det1"

    def test_export_is_json_serializable(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.DETECTOR, "det1", _FakeDetector)

        result = reg.export_schemas()
        json_str = json.dumps(result)
        assert isinstance(json_str, str)


# ─── BacktestResult JSON Export Tests ───────────────────────────────────


class TestBacktestResultJsonExport:
    def _make_result(self, n_trades: int = 3, with_metrics_df: bool = True):
        from signalflow.core.containers.raw_data import RawData
        from signalflow.core.containers.strategy_state import StrategyState
        from signalflow.api.result import BacktestResult

        state = StrategyState(strategy_id="test")
        state.portfolio.cash = 11000.0

        trades = []
        for i in range(n_trades):
            trades.append(
                {
                    "pair": "BTCUSDT",
                    "entry_time": TS + timedelta(hours=i),
                    "exit_time": TS + timedelta(hours=i + 1),
                    "entry_price": 100.0 + i,
                    "exit_price": 102.0 + i,
                    "pnl": 20.0,
                    "pnl_pct": 0.02,
                    "exit_reason": "take_profit",
                }
            )

        metrics_df = None
        if with_metrics_df:
            metrics_df = pl.DataFrame(
                {
                    "timestamp": [(TS + timedelta(hours=i)).timestamp() for i in range(5)],
                    "equity": [10000.0, 10100.0, 10200.0, 10150.0, 10300.0],
                    "total_return": [0.0, 0.01, 0.02, 0.015, 0.03],
                }
            )

        raw = RawData(
            datetime_start=TS,
            datetime_end=TS + timedelta(hours=10),
            pairs=["BTCUSDT"],
            data={"spot": _make_ohlcv(10, ["BTCUSDT"])},
        )

        return BacktestResult(
            state=state,
            trades=trades,
            signals=None,
            raw=raw,
            config={"capital": 10000.0, "fee": 0.001},
            metrics_df=metrics_df,
        )

    def test_to_json_dict_is_serializable(self):
        result = self._make_result()
        d = result.to_json_dict()

        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_to_json_dict_contains_equity_curve(self):
        result = self._make_result(with_metrics_df=True)
        d = result.to_json_dict()

        assert "equity_curve" in d
        assert len(d["equity_curve"]) == 5
        assert "equity" in d["equity_curve"][0]

    def test_to_json_dict_empty_metrics_df(self):
        result = self._make_result(with_metrics_df=False)
        d = result.to_json_dict()

        assert d["equity_curve"] == []

    def test_to_json_dict_trades_have_iso_dates(self):
        result = self._make_result()
        d = result.to_json_dict()

        for trade in d["trades"]:
            if "entry_time" in trade:
                assert isinstance(trade["entry_time"], str)
                # Should be ISO format
                datetime.fromisoformat(trade["entry_time"])

    def test_to_json_dict_handles_inf_nan(self):
        from signalflow.api.result import BacktestResult

        result = self._make_result(n_trades=0)
        d = result.to_json_dict()

        # Should be JSON-serializable even with inf/nan from profit_factor
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ─── BacktestRunner Progress/Cancel Tests ───────────────────────────────


class TestBacktestRunnerProgress:
    def _make_runner(self, **kwargs):
        from signalflow.strategy.runner.backtest_runner import BacktestRunner
        from signalflow.strategy.broker.backtest import BacktestBroker
        from signalflow.strategy.broker.executor.virtual_spot import VirtualSpotExecutor
        from signalflow.data.strategy_store.memory import InMemoryStrategyStore

        broker = BacktestBroker(
            executor=VirtualSpotExecutor(fee_rate=0.001),
            store=InMemoryStrategyStore(),
        )
        return BacktestRunner(
            broker=broker,
            entry_rules=[],
            exit_rules=[],
            show_progress=False,
            **kwargs,
        )

    def _make_data_and_signals(self, n: int = 20):
        from signalflow.core.containers.raw_data import RawData
        from signalflow.core.containers.signals import Signals

        raw = RawData(
            datetime_start=TS,
            datetime_end=TS + timedelta(hours=n),
            pairs=["BTCUSDT"],
            data={"spot": _make_ohlcv(n, ["BTCUSDT"])},
        )
        signals = Signals(pl.DataFrame())
        return raw, signals

    def test_progress_callback_called(self):
        calls: list[tuple[int, int, dict]] = []

        def on_progress(current: int, total: int, metrics: dict):
            calls.append((current, total, metrics))

        runner = self._make_runner(
            progress_callback=on_progress,
            progress_interval=5,
        )
        raw, signals = self._make_data_and_signals(n=20)
        runner.run(raw, signals)

        # Should have been called at intervals + final
        assert len(calls) >= 1
        # Last call should be at 100%
        last = calls[-1]
        assert last[0] == last[1]

    def test_cancel_event_stops_early(self):
        cancel = Event()

        calls: list[int] = []

        def on_progress(current: int, total: int, metrics: dict):
            calls.append(current)
            if current >= 5:
                cancel.set()

        runner = self._make_runner(
            progress_callback=on_progress,
            progress_interval=1,
            cancel_event=cancel,
        )
        raw, signals = self._make_data_and_signals(n=100)
        runner.run(raw, signals)

        # Should have stopped early — not processed all 100 bars
        assert len(runner.trades) == 0  # no signals, no trades
        # The metrics history should be shorter than 100
        assert runner.metrics_df.height < 100

    def test_no_callback_still_works(self):
        runner = self._make_runner()
        raw, signals = self._make_data_and_signals(n=10)
        state = runner.run(raw, signals)
        assert state is not None


# ─── BacktestBuilder Serialization Tests ────────────────────────────────


class TestBacktestBuilderSerialization:
    def test_to_dict_basic(self):
        from signalflow.api.builder import BacktestBuilder

        builder = BacktestBuilder(strategy_id="test_strategy")
        builder.capital(50000)
        builder.fee(0.0005)
        builder.entry(size_pct=0.1, max_positions=5)
        builder.exit(tp=0.03, sl=0.015)

        d = builder.to_dict()

        assert d["strategy_id"] == "test_strategy"
        assert d["capital"] == 50000
        assert d["fee"] == 0.0005
        assert d["entry"]["size_pct"] == 0.1
        assert d["exit"]["tp"] == 0.03

    def test_to_dict_is_json_serializable(self):
        from signalflow.api.builder import BacktestBuilder

        builder = BacktestBuilder(strategy_id="test")
        builder.capital(10000)
        builder.entry(size_pct=0.1)
        builder.exit(tp=0.02, sl=0.01)

        d = builder.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_to_dict_with_data_params(self):
        from signalflow.api.builder import BacktestBuilder

        builder = BacktestBuilder()
        builder.data(
            exchange="binance",
            pairs=["BTCUSDT", "ETHUSDT"],
            start="2024-01-01",
            timeframe="1h",
        )

        d = builder.to_dict()
        assert "data" in d
        assert d["data"]["exchange"] == "binance"
        assert d["data"]["pairs"] == ["BTCUSDT", "ETHUSDT"]

    def test_to_dict_multi_entry_exit(self):
        from signalflow.api.builder import BacktestBuilder

        builder = BacktestBuilder()
        builder.entry(name="aggressive", size_pct=0.2, max_positions=3)
        builder.entry(name="conservative", size_pct=0.05, max_positions=10)
        builder.exit(name="tight", tp=0.02, sl=0.01)
        builder.exit(name="wide", tp=0.05, sl=0.03)

        d = builder.to_dict()
        assert "entries" in d
        assert "aggressive" in d["entries"]
        assert "conservative" in d["entries"]
        assert "exits" in d
        assert "tight" in d["exits"]
        assert "wide" in d["exits"]

    def test_from_dict_round_trip(self):
        from signalflow.api.builder import BacktestBuilder

        original = BacktestBuilder(strategy_id="round_trip")
        original.capital(25000)
        original.fee(0.002)
        original.entry(size_pct=0.15, max_positions=3)
        original.exit(tp=0.04, sl=0.02)

        d = original.to_dict()
        restored = BacktestBuilder.from_dict(d)

        assert restored.strategy_id == "round_trip"
        assert restored._capital == 25000
        assert restored._fee == 0.002
        assert restored._entry_config["size_pct"] == 0.15
        assert restored._exit_config["tp"] == 0.04

    def test_from_dict_with_data_params(self):
        from signalflow.api.builder import BacktestBuilder

        config = {
            "strategy_id": "test",
            "data": {
                "exchange": "binance",
                "pairs": ["BTCUSDT"],
                "start": "2024-01-01",
                "timeframe": "4h",
            },
            "capital": 10000,
            "fee": 0.001,
        }

        builder = BacktestBuilder.from_dict(config)
        assert builder._data_params["exchange"] == "binance"
        assert builder._data_params["pairs"] == ["BTCUSDT"]

    def test_from_dict_multi_components(self):
        from signalflow.api.builder import BacktestBuilder

        config = {
            "strategy_id": "ensemble",
            "entries": {
                "a": {"size_pct": 0.1, "max_positions": 5},
                "b": {"size_pct": 0.05, "max_positions": 10},
            },
            "exits": {
                "tight": {"tp": 0.02, "sl": 0.01},
                "trailing": {"trailing": 0.03},
            },
            "capital": 50000,
            "fee": 0.001,
        }

        builder = BacktestBuilder.from_dict(config)
        assert "a" in builder._named_entries
        assert "b" in builder._named_entries
        assert "tight" in builder._named_exits
        assert "trailing" in builder._named_exits
