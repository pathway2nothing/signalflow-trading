"""Tests for signalflow.core.containers.strategy_state.StrategyState."""

from datetime import datetime

import pytest

from signalflow.core.containers.strategy_state import StrategyState


class TestStrategyStateCreation:
    def test_defaults(self):
        state = StrategyState(strategy_id="s1")
        assert state.strategy_id == "s1"
        assert state.last_ts is None
        assert state.last_event_id is None
        assert state.portfolio.cash == 0.0
        assert state.runtime == {}
        assert state.metrics == {}
        assert state.metrics_phase_done == set()

    def test_portfolio_is_independent_per_instance(self):
        s1 = StrategyState(strategy_id="a")
        s2 = StrategyState(strategy_id="b")
        s1.portfolio.cash = 999.0
        assert s2.portfolio.cash == 0.0


class TestStrategyStateTouch:
    def test_touch_updates_last_ts(self):
        state = StrategyState(strategy_id="t")
        ts = datetime(2024, 6, 1, 12, 0)
        state.touch(ts)
        assert state.last_ts == ts

    def test_touch_updates_event_id(self):
        state = StrategyState(strategy_id="t")
        state.touch(datetime(2024, 1, 1), event_id="evt_1")
        assert state.last_event_id == "evt_1"

    def test_touch_without_event_id_keeps_old(self):
        state = StrategyState(strategy_id="t")
        state.touch(datetime(2024, 1, 1), event_id="evt_1")
        state.touch(datetime(2024, 1, 2))
        assert state.last_event_id == "evt_1"
        assert state.last_ts == datetime(2024, 1, 2)

    def test_touch_multiple_times(self):
        state = StrategyState(strategy_id="t")
        for i in range(5):
            state.touch(datetime(2024, 1, 1 + i), event_id=f"e{i}")
        assert state.last_ts == datetime(2024, 1, 5)
        assert state.last_event_id == "e4"


class TestStrategyStateResetTickCache:
    def test_reset_clears_phase_done(self):
        state = StrategyState(strategy_id="t")
        state.metrics_phase_done.add("returns")
        state.metrics_phase_done.add("drawdown")
        state.reset_tick_cache()
        assert state.metrics_phase_done == set()

    def test_reset_on_empty_set(self):
        state = StrategyState(strategy_id="t")
        state.reset_tick_cache()
        assert state.metrics_phase_done == set()


class TestStrategyStateAttributes:
    def test_runtime_dict_access(self):
        state = StrategyState(strategy_id="t")
        state.runtime["cooldown"] = 5
        assert state.runtime["cooldown"] == 5

    def test_metrics_dict_access(self):
        state = StrategyState(strategy_id="t")
        state.metrics["total_return"] = 0.05
        assert state.metrics["total_return"] == pytest.approx(0.05)

    def test_portfolio_open_positions(self, strategy_state, long_position):
        strategy_state.portfolio.positions[long_position.id] = long_position
        assert len(strategy_state.portfolio.open_positions()) == 1
