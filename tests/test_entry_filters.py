"""Tests for entry filters."""

from datetime import datetime

import pytest

from signalflow.core import Position, PositionType, StrategyState
from signalflow.strategy.component.entry.filters import (
    CompositeEntryFilter,
    CorrelationFilter,
    DrawdownFilter,
    PriceDistanceFilter,
    RegimeFilter,
    SignalAccuracyFilter,
    TimeOfDayFilter,
    VolatilityFilter,
)
from signalflow.strategy.component.sizing.base import SignalContext


@pytest.fixture
def signal_ctx():
    """Standard signal context for testing."""
    return SignalContext(
        pair="BTCUSDT",
        signal_type="rise",
        probability=0.8,
        price=50000.0,
        timestamp=datetime(2024, 1, 1, 10, 0),
    )


@pytest.fixture
def state():
    """Strategy state with 10k cash."""
    state = StrategyState(strategy_id="test")
    state.portfolio.cash = 10000.0
    return state


# ── RegimeFilter ─────────────────────────────────────────────────────────────


class TestRegimeFilter:
    def test_allows_rise_in_trend_up(self, signal_ctx, state):
        state.runtime["regime"] = {"BTCUSDT": "trend_up"}
        f = RegimeFilter()
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed
        assert reason == ""

    def test_allows_rise_in_oversold(self, signal_ctx, state):
        state.runtime["regime"] = {"BTCUSDT": "mean_reversion_oversold"}
        f = RegimeFilter()
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed

    def test_rejects_rise_in_trend_down(self, signal_ctx, state):
        state.runtime["regime"] = {"BTCUSDT": "trend_down"}
        f = RegimeFilter()
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert not allowed
        assert "trend_down" in reason

    def test_allows_fall_in_trend_down(self, state):
        ctx = SignalContext(
            pair="BTCUSDT",
            signal_type="fall",
            probability=0.8,
            price=50000.0,
            timestamp=datetime(2024, 1, 1, 10, 0),
        )
        state.runtime["regime"] = {"BTCUSDT": "trend_down"}
        f = RegimeFilter()
        allowed, reason = f.allow_entry(ctx, state, {})
        assert allowed

    def test_allows_when_no_regime_data(self, signal_ctx, state):
        f = RegimeFilter()
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed

    def test_uses_global_regime_as_fallback(self, signal_ctx, state):
        state.runtime["regime"] = {"global": "trend_up"}
        f = RegimeFilter()
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed


# ── VolatilityFilter ─────────────────────────────────────────────────────────


class TestVolatilityFilter:
    def test_allows_within_range(self, signal_ctx, state):
        state.runtime["atr"] = {"BTCUSDT": 500.0}  # 1% of price
        f = VolatilityFilter(min_volatility=0.005, max_volatility=0.02)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed

    def test_rejects_too_low_volatility(self, signal_ctx, state):
        state.runtime["atr"] = {"BTCUSDT": 100.0}  # 0.2% of price
        f = VolatilityFilter(min_volatility=0.01)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert not allowed
        assert "min=" in reason

    def test_rejects_too_high_volatility(self, signal_ctx, state):
        state.runtime["atr"] = {"BTCUSDT": 2500.0}  # 5% of price
        f = VolatilityFilter(max_volatility=0.02)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert not allowed
        assert "max=" in reason

    def test_allows_when_no_volatility_data(self, signal_ctx, state):
        f = VolatilityFilter(max_volatility=0.02)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed


# ── DrawdownFilter ───────────────────────────────────────────────────────────


class TestDrawdownFilter:
    def test_allows_when_below_max_drawdown(self, signal_ctx, state):
        state.metrics["current_drawdown"] = 0.05
        f = DrawdownFilter(max_drawdown=0.10)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed

    def test_pauses_on_max_drawdown(self, signal_ctx, state):
        state.metrics["current_drawdown"] = 0.15
        f = DrawdownFilter(max_drawdown=0.10)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert not allowed
        assert "15.00%" in reason

    def test_remains_paused_above_recovery(self, signal_ctx, state):
        f = DrawdownFilter(max_drawdown=0.10, recovery_threshold=0.05)

        # Hit drawdown limit
        state.metrics["current_drawdown"] = 0.12
        f.allow_entry(signal_ctx, state, {})

        # Still above recovery threshold
        state.metrics["current_drawdown"] = 0.07
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert not allowed
        assert "paused" in reason

    def test_resumes_after_recovery(self, signal_ctx, state):
        f = DrawdownFilter(max_drawdown=0.10, recovery_threshold=0.05)

        # Hit drawdown limit
        state.metrics["current_drawdown"] = 0.12
        f.allow_entry(signal_ctx, state, {})

        # Recovered below threshold
        state.metrics["current_drawdown"] = 0.04
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed


# ── CorrelationFilter ────────────────────────────────────────────────────────


class TestCorrelationFilter:
    def test_allows_when_no_correlations(self, signal_ctx, state):
        f = CorrelationFilter()
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed

    def test_allows_uncorrelated_position(self, signal_ctx, state):
        pos = Position(pair="ETHUSDT", position_type=PositionType.LONG, entry_price=3000.0, qty=1.0)
        state.portfolio.positions[pos.id] = pos
        state.runtime["correlations"] = {("BTCUSDT", "ETHUSDT"): 0.5}

        f = CorrelationFilter(max_correlation=0.7)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed

    def test_rejects_correlated_positions(self, signal_ctx, state):
        # Add two correlated positions
        for pair in ["ETHUSDT", "SOLUSDT"]:
            pos = Position(pair=pair, position_type=PositionType.LONG, entry_price=100.0, qty=1.0)
            state.portfolio.positions[f"pos_{pair}"] = pos

        state.runtime["correlations"] = {
            ("BTCUSDT", "ETHUSDT"): 0.85,
            ("BTCUSDT", "SOLUSDT"): 0.80,
        }

        f = CorrelationFilter(max_correlation=0.7, max_correlated_positions=2)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert not allowed
        assert "2 correlated positions" in reason


# ── TimeOfDayFilter ──────────────────────────────────────────────────────────


class TestTimeOfDayFilter:
    def test_allows_in_allowed_hours(self, signal_ctx, state):
        f = TimeOfDayFilter(allowed_hours=[9, 10, 11])
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed  # timestamp is hour 10

    def test_rejects_outside_allowed_hours(self, signal_ctx, state):
        f = TimeOfDayFilter(allowed_hours=[12, 13, 14])
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert not allowed
        assert "hour=10" in reason

    def test_rejects_blocked_hours(self, signal_ctx, state):
        f = TimeOfDayFilter(blocked_hours=[10, 11])
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert not allowed
        assert "blocked" in reason

    def test_allows_when_no_timestamp(self, state):
        ctx = SignalContext(
            pair="BTCUSDT", signal_type="rise", probability=0.8, price=50000.0, timestamp=None
        )
        f = TimeOfDayFilter(allowed_hours=[12, 13])
        allowed, reason = f.allow_entry(ctx, state, {})
        assert allowed


# ── PriceDistanceFilter ──────────────────────────────────────────────────────


class TestPriceDistanceFilter:
    def test_allows_first_entry(self, signal_ctx, state):
        f = PriceDistanceFilter(min_distance_pct=0.02)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed

    def test_allows_when_price_dropped_enough_long(self, signal_ctx, state):
        # Existing position at 50000
        pos = Position(
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            qty=0.01,
            entry_time=datetime(2024, 1, 1, 9, 0),
        )
        state.portfolio.positions[pos.id] = pos

        # New signal at 48500 (3% drop)
        signal_ctx.price = 48500.0
        f = PriceDistanceFilter(min_distance_pct=0.02, direction_aware=True)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed

    def test_rejects_when_price_too_close_long(self, signal_ctx, state):
        # Existing position at 50000
        pos = Position(
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            qty=0.01,
            entry_time=datetime(2024, 1, 1, 9, 0),
        )
        state.portfolio.positions[pos.id] = pos

        # New signal at 49500 (1% drop, less than 2% threshold)
        signal_ctx.price = 49500.0
        f = PriceDistanceFilter(min_distance_pct=0.02, direction_aware=True)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert not allowed
        assert "too close" in reason

    def test_direction_aware_short(self, state):
        # SHORT signal - should want price to go UP
        ctx = SignalContext(
            pair="BTCUSDT",
            signal_type="fall",
            probability=0.8,
            price=51000.0,  # 2% above entry
            timestamp=datetime(2024, 1, 1, 10, 0),
        )

        pos = Position(
            pair="BTCUSDT",
            position_type=PositionType.SHORT,
            entry_price=50000.0,
            qty=0.01,
            entry_time=datetime(2024, 1, 1, 9, 0),
        )
        state.portfolio.positions[pos.id] = pos

        f = PriceDistanceFilter(min_distance_pct=0.02, direction_aware=True)
        allowed, reason = f.allow_entry(ctx, state, {})
        assert allowed

    def test_absolute_distance_mode(self, signal_ctx, state):
        pos = Position(
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            qty=0.01,
            entry_time=datetime(2024, 1, 1, 9, 0),
        )
        state.portfolio.positions[pos.id] = pos

        # Price at 50500 (1% up)
        signal_ctx.price = 50500.0
        f = PriceDistanceFilter(min_distance_pct=0.02, direction_aware=False)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert not allowed  # Only 1% away, need 2%


# ── SignalAccuracyFilter ─────────────────────────────────────────────────────


class TestSignalAccuracyFilter:
    def test_allows_when_no_data(self, signal_ctx, state):
        f = SignalAccuracyFilter(min_accuracy=0.50)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed

    def test_allows_above_min_accuracy(self, signal_ctx, state):
        state.runtime["signal_accuracy"] = {"BTCUSDT": {"overall": 0.55, "samples": 50}}
        f = SignalAccuracyFilter(min_accuracy=0.50, min_samples=20)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed

    def test_rejects_below_min_accuracy(self, signal_ctx, state):
        state.runtime["signal_accuracy"] = {"BTCUSDT": {"overall": 0.40, "samples": 50}}
        f = SignalAccuracyFilter(min_accuracy=0.50, min_samples=20)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert not allowed
        assert "40.00%" in reason

    def test_allows_when_insufficient_samples(self, signal_ctx, state):
        state.runtime["signal_accuracy"] = {"BTCUSDT": {"overall": 0.30, "samples": 10}}
        f = SignalAccuracyFilter(min_accuracy=0.50, min_samples=20)
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed  # Not enough samples


# ── CompositeEntryFilter ─────────────────────────────────────────────────────


class TestCompositeEntryFilter:
    def test_empty_filters_allows(self, signal_ctx, state):
        f = CompositeEntryFilter(filters=[])
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed

    def test_require_all_passes_all_pass(self, signal_ctx, state):
        state.runtime["regime"] = {"BTCUSDT": "trend_up"}
        state.metrics["current_drawdown"] = 0.05

        f = CompositeEntryFilter(
            filters=[
                RegimeFilter(),
                DrawdownFilter(max_drawdown=0.10),
            ],
            require_all=True,
        )
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed

    def test_require_all_fails_if_any_fails(self, signal_ctx, state):
        state.runtime["regime"] = {"BTCUSDT": "trend_down"}  # Will fail
        state.metrics["current_drawdown"] = 0.05  # Would pass

        f = CompositeEntryFilter(
            filters=[
                RegimeFilter(),
                DrawdownFilter(max_drawdown=0.10),
            ],
            require_all=True,
        )
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert not allowed
        assert "RegimeFilter" in reason

    def test_require_any_passes_if_one_passes(self, signal_ctx, state):
        state.runtime["regime"] = {"BTCUSDT": "trend_down"}  # Will fail
        state.metrics["current_drawdown"] = 0.05  # Will pass

        f = CompositeEntryFilter(
            filters=[
                RegimeFilter(),
                DrawdownFilter(max_drawdown=0.10),
            ],
            require_all=False,
        )
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert allowed

    def test_require_any_fails_if_all_fail(self, signal_ctx, state):
        state.runtime["regime"] = {"BTCUSDT": "trend_down"}
        state.metrics["current_drawdown"] = 0.15

        f = CompositeEntryFilter(
            filters=[
                RegimeFilter(),
                DrawdownFilter(max_drawdown=0.10),
            ],
            require_all=False,
        )
        allowed, reason = f.allow_entry(signal_ctx, state, {})
        assert not allowed
        assert "RegimeFilter" in reason
        assert "DrawdownFilter" in reason
