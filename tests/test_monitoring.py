"""Tests for monitoring alerts."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.core.containers.position import Position
from signalflow.core.containers.signals import Signals
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import PositionType
from signalflow.strategy.monitoring.alerts import (
    AlertManager,
    MaxDrawdownAlert,
    NoSignalsAlert,
    StuckPositionAlert,
)


@pytest.fixture
def state() -> StrategyState:
    state = StrategyState(strategy_id="test")
    state.portfolio.cash = 10000.0
    return state


@pytest.fixture
def empty_signals() -> Signals:
    return Signals(pl.DataFrame())


@pytest.fixture
def signals_with_data() -> Signals:
    return Signals(pl.DataFrame({"pair": ["BTCUSDT"], "signal": [1]}))


class TestMaxDrawdownAlert:
    def test_no_alert_below_threshold(self, state: StrategyState, empty_signals: Signals) -> None:
        alert = MaxDrawdownAlert(warning_threshold=0.05, critical_threshold=0.10)
        state.metrics = {"max_drawdown": 0.02}

        result = alert.check(state, empty_signals, datetime.now())
        assert result is None

    def test_warning_at_warning_level(self, state: StrategyState, empty_signals: Signals) -> None:
        alert = MaxDrawdownAlert(warning_threshold=0.05, critical_threshold=0.10)
        state.metrics = {"max_drawdown": 0.06}

        result = alert.check(state, empty_signals, datetime.now())
        assert result is not None
        assert result.level == "warning"
        assert result.alert_name == "max_drawdown"
        assert "6" in result.message and "%" in result.message

    def test_critical_at_critical_level(self, state: StrategyState, empty_signals: Signals) -> None:
        alert = MaxDrawdownAlert(warning_threshold=0.05, critical_threshold=0.10)
        state.metrics = {"max_drawdown": 0.11}

        result = alert.check(state, empty_signals, datetime.now())
        assert result is not None
        assert result.level == "critical"
        assert result.details["drawdown"] == 0.11

    def test_critical_takes_precedence(self, state: StrategyState, empty_signals: Signals) -> None:
        alert = MaxDrawdownAlert(warning_threshold=0.05, critical_threshold=0.10)
        state.metrics = {"max_drawdown": 0.15}

        result = alert.check(state, empty_signals, datetime.now())
        assert result.level == "critical"

    def test_disabled_alert_skipped(self, state: StrategyState, empty_signals: Signals) -> None:
        alert = MaxDrawdownAlert(warning_threshold=0.05, enabled=False)
        state.metrics = {"max_drawdown": 0.10}

        # Alert is disabled but check still runs (manager checks enabled flag)
        result = alert.check(state, empty_signals, datetime.now())
        # Should still return result, manager filters by enabled flag
        assert result is not None


class TestNoSignalsAlert:
    def test_resets_on_signal(self, state: StrategyState, signals_with_data: Signals) -> None:
        alert = NoSignalsAlert(max_bars_without_signal=5)
        ts = datetime.now()

        # Increment counter
        alert._bars_since_signal = 3
        result = alert.check(state, signals_with_data, ts)

        assert result is None
        assert alert._bars_since_signal == 0

    def test_warns_after_threshold(self, state: StrategyState, empty_signals: Signals) -> None:
        alert = NoSignalsAlert(max_bars_without_signal=5)
        ts = datetime.now()

        # Increment to threshold
        for _ in range(4):
            alert.check(state, empty_signals, ts)

        result = alert.check(state, empty_signals, ts)
        assert result is not None
        assert result.level == "warning"
        assert result.alert_name == "no_signals"
        assert result.details["bars_since_signal"] == 5

    def test_no_alert_within_threshold(self, state: StrategyState, empty_signals: Signals) -> None:
        alert = NoSignalsAlert(max_bars_without_signal=10)
        ts = datetime.now()

        for _ in range(5):
            result = alert.check(state, empty_signals, ts)
            assert result is None

    def test_counter_increments(self, state: StrategyState, empty_signals: Signals) -> None:
        alert = NoSignalsAlert(max_bars_without_signal=100)
        ts = datetime.now()

        for i in range(1, 6):
            alert.check(state, empty_signals, ts)
            assert alert._bars_since_signal == i


class TestStuckPositionAlert:
    def test_no_alert_for_recent(self, state: StrategyState, empty_signals: Signals) -> None:
        alert = StuckPositionAlert(max_duration=timedelta(days=7))
        ts = datetime(2024, 1, 2, 12, 0, 0)

        # Add recent position
        position = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_time=datetime(2024, 1, 2, 11, 0, 0),  # 1 hour ago
            entry_price=45000.0,
            qty=0.1,
        )
        state.portfolio.positions[position.id] = position

        result = alert.check(state, empty_signals, ts)
        assert result is None

    def test_warns_for_old_position(self, state: StrategyState, empty_signals: Signals) -> None:
        alert = StuckPositionAlert(max_duration=timedelta(days=7))
        ts = datetime(2024, 1, 10, 12, 0, 0)

        # Add old position
        position = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_time=datetime(2024, 1, 1, 12, 0, 0),  # 9 days ago
            entry_price=45000.0,
            qty=0.1,
        )
        state.portfolio.positions[position.id] = position

        result = alert.check(state, empty_signals, ts)
        assert result is not None
        assert result.level == "warning"
        assert result.alert_name == "stuck_position"
        assert result.details["position_id"] == "pos1"
        assert result.details["pair"] == "BTCUSDT"

    def test_no_alert_no_positions(self, state: StrategyState, empty_signals: Signals) -> None:
        alert = StuckPositionAlert(max_duration=timedelta(days=7))
        ts = datetime.now()

        result = alert.check(state, empty_signals, ts)
        assert result is None

    def test_closed_positions_ignored(self, state: StrategyState, empty_signals: Signals) -> None:
        alert = StuckPositionAlert(max_duration=timedelta(days=1))
        ts = datetime(2024, 1, 10, 12, 0, 0)

        # Add closed old position
        position = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            entry_price=45000.0,
            qty=0.1,
            is_closed=True,
        )
        state.portfolio.positions[position.id] = position

        result = alert.check(state, empty_signals, ts)
        assert result is None


class TestAlertManager:
    def test_runs_all_checks(self, state: StrategyState, empty_signals: Signals) -> None:
        alert1 = MaxDrawdownAlert(warning_threshold=0.05)
        alert2 = NoSignalsAlert(max_bars_without_signal=1)

        state.metrics = {"max_drawdown": 0.06}

        manager = AlertManager(alerts=[alert1, alert2])
        ts = datetime.now()

        results = manager.check_all(state, empty_signals, ts)

        assert len(results) == 2
        assert results[0].alert_name == "max_drawdown"
        assert results[1].alert_name == "no_signals"

    def test_alert_history_accumulated(self, state: StrategyState, empty_signals: Signals) -> None:
        alert = MaxDrawdownAlert(warning_threshold=0.05)
        manager = AlertManager(alerts=[alert])

        state.metrics = {"max_drawdown": 0.06}
        manager.check_all(state, empty_signals, datetime.now())

        state.metrics = {"max_drawdown": 0.07}
        manager.check_all(state, empty_signals, datetime.now())

        assert len(manager.alert_history) == 2

    def test_exception_in_alert_handled(self, state: StrategyState, empty_signals: Signals) -> None:
        class BrokenAlert(MaxDrawdownAlert):
            def check(self, state, signals, ts):
                raise RuntimeError("boom")

        alert1 = BrokenAlert(warning_threshold=0.05)
        alert2 = NoSignalsAlert(max_bars_without_signal=1)
        manager = AlertManager(alerts=[alert1, alert2])

        state.metrics = {"max_drawdown": 0.06}
        results = manager.check_all(state, empty_signals, datetime.now())

        # Only alert2 should have triggered
        assert len(results) == 1
        assert results[0].alert_name == "no_signals"

    def test_disabled_alert_skipped(self, state: StrategyState, empty_signals: Signals) -> None:
        alert1 = MaxDrawdownAlert(warning_threshold=0.05, enabled=False)
        alert2 = NoSignalsAlert(max_bars_without_signal=1, enabled=True)

        manager = AlertManager(alerts=[alert1, alert2])
        state.metrics = {"max_drawdown": 0.10}

        results = manager.check_all(state, empty_signals, datetime.now())

        assert len(results) == 1
        assert results[0].alert_name == "no_signals"

    def test_no_alerts_triggered(self, state: StrategyState, signals_with_data: Signals) -> None:
        alert = NoSignalsAlert(max_bars_without_signal=10)
        manager = AlertManager(alerts=[alert])

        results = manager.check_all(state, signals_with_data, datetime.now())

        assert len(results) == 0
        assert len(manager.alert_history) == 0
