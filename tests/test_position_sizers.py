"""Tests for position sizing strategies."""

from datetime import datetime

import pytest

from signalflow.core import Position, PositionType, StrategyState
from signalflow.strategy.component.sizing import (
    FixedFractionSizer,
    KellyCriterionSizer,
    MartingaleSizer,
    RiskParitySizer,
    SignalContext,
    SignalStrengthSizer,
    VolatilityTargetSizer,
)


@pytest.fixture
def signal_ctx():
    """Standard signal context for testing."""
    return SignalContext(
        pair="BTCUSDT",
        signal_type="rise",
        probability=0.8,
        price=50000.0,
        timestamp=datetime(2024, 1, 1),
    )


@pytest.fixture
def state_10k():
    """Strategy state with 10k cash."""
    state = StrategyState(strategy_id="test")
    state.portfolio.cash = 10000.0
    return state


# ── FixedFractionSizer ───────────────────────────────────────────────────────


class TestFixedFractionSizer:
    def test_computes_fraction_of_equity(self, signal_ctx, state_10k):
        sizer = FixedFractionSizer(fraction=0.02)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})
        assert size == pytest.approx(200.0)  # 10000 * 0.02

    def test_respects_min_notional(self, signal_ctx, state_10k):
        sizer = FixedFractionSizer(fraction=0.0001, min_notional=100.0)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})
        assert size == 0.0  # 1.0 < 100.0

    def test_caps_at_max_notional(self, signal_ctx, state_10k):
        sizer = FixedFractionSizer(fraction=0.5, max_notional=1000.0)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})
        assert size == 1000.0

    def test_uses_portfolio_equity(self, signal_ctx, state_10k):
        # Add an open position
        pos = Position(
            pair="ETHUSDT",
            position_type=PositionType.LONG,
            entry_price=1000.0,
            qty=2.0,
        )
        state_10k.portfolio.positions[pos.id] = pos
        # Equity = 10000 cash + 2*1000 = 12000
        sizer = FixedFractionSizer(fraction=0.02)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0, "ETHUSDT": 1000.0})
        assert size == pytest.approx(240.0)  # 12000 * 0.02


# ── SignalStrengthSizer ──────────────────────────────────────────────────────


class TestSignalStrengthSizer:
    def test_scales_by_probability(self, state_10k):
        ctx_high = SignalContext(pair="BTCUSDT", signal_type="rise", probability=0.9, price=100.0, timestamp=None)
        ctx_low = SignalContext(pair="BTCUSDT", signal_type="rise", probability=0.6, price=100.0, timestamp=None)

        sizer = SignalStrengthSizer(base_size=100.0, min_probability=0.5)

        size_high = sizer.compute_size(ctx_high, state_10k, {"BTCUSDT": 100.0})
        size_low = sizer.compute_size(ctx_low, state_10k, {"BTCUSDT": 100.0})

        assert size_high > size_low
        assert size_high == pytest.approx(90.0)
        assert size_low == pytest.approx(60.0)

    def test_below_min_probability_returns_zero(self, state_10k):
        ctx = SignalContext(pair="BTCUSDT", signal_type="rise", probability=0.4, price=100.0, timestamp=None)
        sizer = SignalStrengthSizer(base_size=100.0, min_probability=0.5)
        size = sizer.compute_size(ctx, state_10k, {"BTCUSDT": 100.0})
        assert size == 0.0

    def test_caps_at_max_notional(self, signal_ctx, state_10k):
        sizer = SignalStrengthSizer(base_size=1000.0, max_notional=500.0)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})
        assert size == 500.0


# ── KellyCriterionSizer ──────────────────────────────────────────────────────


class TestKellyCriterionSizer:
    def test_uses_signal_probability(self, signal_ctx, state_10k):
        sizer = KellyCriterionSizer(
            use_signal_probability=True,
            default_payoff_ratio=1.5,
            kelly_fraction=0.5,
        )
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})
        # Kelly: f = (0.8 * 1.5 - 0.2) / 1.5 = 0.667
        # Half Kelly: 0.333
        # Size = 10000 * 0.333 = 3333 (capped at max_fraction=0.25 -> 2500)
        assert size > 0
        assert size <= 10000 * 0.25

    def test_uses_historical_stats_when_available(self, signal_ctx, state_10k):
        # Add 35 closed positions with 60% win rate
        for i in range(35):
            pos = Position(
                pair="BTCUSDT",
                position_type=PositionType.LONG,
                entry_price=100.0,
                qty=1.0,
                is_closed=True,
                realized_pnl=10.0 if i < 21 else -5.0,  # 60% win rate, 2:1 payoff
            )
            state_10k.portfolio.positions[f"pos_{i}"] = pos

        sizer = KellyCriterionSizer(min_trades_for_stats=30, kelly_fraction=0.5)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})
        assert size > 0

    def test_returns_zero_for_negative_kelly(self, state_10k):
        # Low probability signal should result in negative Kelly -> 0
        ctx = SignalContext(pair="BTCUSDT", signal_type="rise", probability=0.3, price=100.0, timestamp=None)
        sizer = KellyCriterionSizer(
            use_signal_probability=True,
            default_payoff_ratio=1.0,
        )
        size = sizer.compute_size(ctx, state_10k, {"BTCUSDT": 100.0})
        # Kelly f = (0.3 * 1.0 - 0.7) / 1.0 = -0.4 -> clamped to 0
        assert size == 0.0


# ── VolatilityTargetSizer ────────────────────────────────────────────────────


class TestVolatilityTargetSizer:
    def test_sizes_inversely_to_volatility(self, signal_ctx, state_10k):
        state_10k.runtime["atr"] = {"BTCUSDT": 500.0}  # 1% of price

        sizer = VolatilityTargetSizer(target_volatility=0.01)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})

        # vol_pct = 500/50000 = 0.01
        # notional = (0.01 * 10000) / 0.01 = 10000
        # Capped at max_fraction=0.20 -> 2000
        assert size == pytest.approx(2000.0)

    def test_uses_default_volatility_when_no_atr(self, signal_ctx, state_10k):
        sizer = VolatilityTargetSizer(target_volatility=0.01, default_volatility_pct=0.02)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})
        # notional = (0.01 * 10000) / 0.02 = 5000
        # Capped at max_fraction=0.20 -> 2000
        assert size == pytest.approx(2000.0)

    def test_high_volatility_reduces_size(self, signal_ctx, state_10k):
        state_10k.runtime["atr"] = {"BTCUSDT": 2500.0}  # 5% of price

        sizer = VolatilityTargetSizer(target_volatility=0.01, max_fraction=1.0)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})

        # vol_pct = 2500/50000 = 0.05
        # notional = (0.01 * 10000) / 0.05 = 2000
        assert size == pytest.approx(2000.0)


# ── RiskParitySizer ──────────────────────────────────────────────────────────


class TestRiskParitySizer:
    def test_equal_risk_budget(self, signal_ctx, state_10k):
        state_10k.runtime["atr"] = {"BTCUSDT": 500.0}

        sizer = RiskParitySizer(target_positions=10)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})

        # risk_budget = 1/10 = 0.1
        # vol_pct = 500/50000 = 0.01
        # notional = (0.1 * 10000) / 0.01 = 100000
        # Capped at equity/target_positions = 1000
        assert size == pytest.approx(1000.0)

    def test_uses_default_volatility(self, signal_ctx, state_10k):
        sizer = RiskParitySizer(target_positions=10, default_volatility_pct=0.02)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})
        # notional = (0.1 * 10000) / 0.02 = 50000
        # Capped at 1000
        assert size == pytest.approx(1000.0)


# ── MartingaleSizer ──────────────────────────────────────────────────────────


class TestMartingaleSizer:
    def test_base_size_first_level(self, signal_ctx, state_10k):
        sizer = MartingaleSizer(base_size=100.0, multiplier=1.5)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})
        assert size == pytest.approx(100.0)  # Level 0: base

    def test_increases_size_with_grid_levels(self, signal_ctx, state_10k):
        # Add one existing position in BTCUSDT
        pos = Position(
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            qty=0.002,
        )
        state_10k.portfolio.positions[pos.id] = pos

        sizer = MartingaleSizer(base_size=100.0, multiplier=1.5)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})
        assert size == pytest.approx(150.0)  # Level 1: 100 * 1.5

    def test_multiple_grid_levels(self, signal_ctx, state_10k):
        # Add two existing positions
        for i in range(2):
            pos = Position(
                pair="BTCUSDT",
                position_type=PositionType.LONG,
                entry_price=50000.0 - i * 1000,
                qty=0.002,
            )
            state_10k.portfolio.positions[f"pos_{i}"] = pos

        sizer = MartingaleSizer(base_size=100.0, multiplier=1.5)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})
        assert size == pytest.approx(225.0)  # Level 2: 100 * 1.5^2

    def test_max_grid_levels(self, signal_ctx, state_10k):
        # Add max_grid_levels positions
        for i in range(5):
            pos = Position(
                pair="BTCUSDT",
                position_type=PositionType.LONG,
                entry_price=50000.0 - i * 1000,
                qty=0.002,
            )
            state_10k.portfolio.positions[f"pos_{i}"] = pos

        sizer = MartingaleSizer(base_size=100.0, multiplier=1.5, max_grid_levels=5)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})
        assert size == 0.0  # Max levels reached

    def test_different_pair_not_counted(self, signal_ctx, state_10k):
        # Add position in different pair
        pos = Position(
            pair="ETHUSDT",
            position_type=PositionType.LONG,
            entry_price=3000.0,
            qty=1.0,
        )
        state_10k.portfolio.positions[pos.id] = pos

        sizer = MartingaleSizer(base_size=100.0, multiplier=1.5)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0})
        assert size == pytest.approx(100.0)  # Level 0 for BTCUSDT

    def test_caps_at_max_notional(self, signal_ctx, state_10k):
        # Add multiple positions to increase size
        for i in range(3):
            pos = Position(
                pair="BTCUSDT",
                position_type=PositionType.LONG,
                entry_price=50000.0,
                qty=0.002,
            )
            state_10k.portfolio.positions[f"pos_{i}"] = pos

        sizer = MartingaleSizer(base_size=100.0, multiplier=2.0, max_notional=500.0)
        size = sizer.compute_size(signal_ctx, state_10k, {"BTCUSDT": 50000.0})
        # Level 3: 100 * 2^3 = 800, capped at 500
        assert size == pytest.approx(500.0)
