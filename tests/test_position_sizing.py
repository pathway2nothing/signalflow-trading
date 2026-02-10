"""Tests for position sizing strategies."""

import pytest

from signalflow.core.containers.position import Position
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import PositionType
from signalflow.strategy.component.sizing import (
    FixedFractionSizer,
    KellyCriterionSizer,
    MartingaleSizer,
    RiskParitySizer,
    SignalContext,
    SignalStrengthSizer,
    VolatilityTargetSizer,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def signal() -> SignalContext:
    """Basic signal context."""
    return SignalContext(
        pair="BTCUSDT",
        signal_type="rise",
        probability=0.7,
        price=50000.0,
    )


@pytest.fixture
def state() -> StrategyState:
    """Basic strategy state with cash."""
    s = StrategyState(strategy_id="test")
    s.portfolio.cash = 10000.0
    return s


@pytest.fixture
def prices() -> dict[str, float]:
    """Price map."""
    return {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}


# ── SignalContext ─────────────────────────────────────────────────────────


class TestSignalContext:
    def test_creation(self):
        ctx = SignalContext(
            pair="BTCUSDT",
            signal_type="rise",
            probability=0.8,
            price=50000.0,
        )
        assert ctx.pair == "BTCUSDT"
        assert ctx.probability == 0.8

    def test_defaults(self):
        ctx = SignalContext(
            pair="BTCUSDT",
            signal_type="rise",
            probability=0.5,
            price=100.0,
        )
        assert ctx.timestamp is None
        assert ctx.meta == {}


# ── FixedFractionSizer ────────────────────────────────────────────────────


class TestFixedFractionSizer:
    def test_basic_sizing(self, signal, state, prices):
        sizer = FixedFractionSizer(fraction=0.02)
        notional = sizer.compute_size(signal, state, prices)
        # 2% of 10000 = 200
        assert notional == 200.0

    def test_different_fraction(self, signal, state, prices):
        sizer = FixedFractionSizer(fraction=0.10)
        notional = sizer.compute_size(signal, state, prices)
        # 10% of 10000 = 1000
        assert notional == 1000.0

    def test_min_notional_filter(self, signal, state, prices):
        state.portfolio.cash = 100.0  # Small balance
        sizer = FixedFractionSizer(fraction=0.01, min_notional=10.0)
        notional = sizer.compute_size(signal, state, prices)
        # 1% of 100 = 1, below min_notional
        assert notional == 0.0

    def test_max_notional_cap(self, signal, state, prices):
        state.portfolio.cash = 100000.0
        sizer = FixedFractionSizer(fraction=0.10, max_notional=5000.0)
        notional = sizer.compute_size(signal, state, prices)
        # 10% of 100000 = 10000, but capped at 5000
        assert notional == 5000.0


# ── SignalStrengthSizer ───────────────────────────────────────────────────


class TestSignalStrengthSizer:
    def test_probability_scaling(self, signal, state, prices):
        sizer = SignalStrengthSizer(base_size=100.0)
        notional = sizer.compute_size(signal, state, prices)
        # 100 * 0.7 probability = 70
        assert notional == 70.0

    def test_high_probability(self, state, prices):
        signal = SignalContext(
            pair="BTCUSDT",
            signal_type="rise",
            probability=1.0,
            price=50000.0,
        )
        sizer = SignalStrengthSizer(base_size=100.0)
        notional = sizer.compute_size(signal, state, prices)
        assert notional == 100.0

    def test_low_probability_filtered(self, state, prices):
        signal = SignalContext(
            pair="BTCUSDT",
            signal_type="rise",
            probability=0.3,  # Below default 0.5 threshold
            price=50000.0,
        )
        sizer = SignalStrengthSizer(base_size=100.0, min_probability=0.5)
        notional = sizer.compute_size(signal, state, prices)
        assert notional == 0.0

    def test_scale_factor(self, signal, state, prices):
        sizer = SignalStrengthSizer(base_size=100.0, scale_factor=2.0)
        notional = sizer.compute_size(signal, state, prices)
        # 100 * 0.7 * 2.0 = 140
        assert notional == 140.0

    def test_max_notional(self, signal, state, prices):
        sizer = SignalStrengthSizer(base_size=1000.0, max_notional=500.0)
        notional = sizer.compute_size(signal, state, prices)
        # 1000 * 0.7 = 700, capped at 500
        assert notional == 500.0


# ── KellyCriterionSizer ───────────────────────────────────────────────────


class TestKellyCriterionSizer:
    def test_uses_signal_probability(self, signal, state, prices):
        sizer = KellyCriterionSizer(
            kelly_fraction=1.0,  # Full Kelly for testing
            use_signal_probability=True,
            default_payoff_ratio=1.5,
        )
        notional = sizer.compute_size(signal, state, prices)
        # win_rate=0.7, payoff=1.5
        # f* = (0.7 * 1.5 - 0.3) / 1.5 = 0.5
        # notional = 10000 * 0.5 = 5000 (but capped at 25%)
        assert notional > 0

    def test_half_kelly(self, signal, state, prices):
        sizer_full = KellyCriterionSizer(kelly_fraction=1.0)
        sizer_half = KellyCriterionSizer(kelly_fraction=0.5)

        notional_full = sizer_full.compute_size(signal, state, prices)
        notional_half = sizer_half.compute_size(signal, state, prices)

        # Half Kelly should be half of full Kelly
        assert notional_half < notional_full

    def test_negative_edge_returns_zero(self, state, prices):
        # Low probability signal with 1:1 payoff = negative edge
        signal = SignalContext(
            pair="BTCUSDT",
            signal_type="rise",
            probability=0.3,
            price=50000.0,
        )
        sizer = KellyCriterionSizer(
            default_payoff_ratio=1.0,
            use_signal_probability=True,
        )
        notional = sizer.compute_size(signal, state, prices)
        # f* = (0.3 * 1.0 - 0.7) / 1.0 = -0.4 -> 0
        assert notional == 0.0

    def test_max_fraction_cap(self, signal, state, prices):
        sizer = KellyCriterionSizer(
            kelly_fraction=1.0,
            max_fraction=0.10,  # Cap at 10%
            default_payoff_ratio=2.0,  # High payoff -> high Kelly
        )
        notional = sizer.compute_size(signal, state, prices)
        # Should be capped at 10% of equity = 1000
        assert notional <= 1000.0


# ── VolatilityTargetSizer ─────────────────────────────────────────────────


class TestVolatilityTargetSizer:
    def test_with_atr_data(self, signal, state, prices):
        # ATR = 1000, price = 50000 -> vol_pct = 2%
        state.runtime["atr"] = {"BTCUSDT": 1000.0}
        sizer = VolatilityTargetSizer(target_volatility=0.01)
        notional = sizer.compute_size(signal, state, prices)
        # target_vol=1%, asset_vol=2% -> notional = (0.01 * 10000) / 0.02 = 5000
        assert notional > 0

    def test_high_vol_smaller_position(self, signal, state, prices):
        # Use high max_fraction to avoid capping
        sizer = VolatilityTargetSizer(target_volatility=0.01, max_fraction=0.90)

        # High volatility
        state.runtime["atr"] = {"BTCUSDT": 2000.0}  # 4% vol
        notional_high_vol = sizer.compute_size(signal, state, prices)

        # Low volatility
        state.runtime["atr"] = {"BTCUSDT": 500.0}  # 1% vol
        notional_low_vol = sizer.compute_size(signal, state, prices)

        # Higher vol -> smaller position
        assert notional_high_vol < notional_low_vol

    def test_default_volatility_when_missing(self, signal, state, prices):
        sizer = VolatilityTargetSizer(
            target_volatility=0.01,
            default_volatility_pct=0.02,
        )
        # No ATR data in state.runtime
        notional = sizer.compute_size(signal, state, prices)
        # Uses default 2% vol
        assert notional > 0

    def test_max_fraction_cap(self, signal, state, prices):
        state.runtime["atr"] = {"BTCUSDT": 100.0}  # Very low vol -> huge position
        sizer = VolatilityTargetSizer(
            target_volatility=0.01,
            max_fraction=0.20,
        )
        notional = sizer.compute_size(signal, state, prices)
        # Should be capped at 20% of equity = 2000
        assert notional <= 2000.0


# ── RiskParitySizer ───────────────────────────────────────────────────────


class TestRiskParitySizer:
    def test_equal_risk_budget(self, signal, state, prices):
        state.runtime["atr"] = {"BTCUSDT": 1000.0}
        sizer = RiskParitySizer(target_positions=10)
        notional = sizer.compute_size(signal, state, prices)
        # 10% risk budget, 2% vol -> notional = (0.1 * 10000) / 0.02 = 50000
        # But capped at equity/10 = 1000
        assert notional > 0
        assert notional <= state.portfolio.cash / 10

    def test_default_volatility(self, signal, state, prices):
        sizer = RiskParitySizer(
            target_positions=10,
            default_volatility_pct=0.02,
        )
        # No ATR data
        notional = sizer.compute_size(signal, state, prices)
        assert notional > 0


# ── MartingaleSizer ───────────────────────────────────────────────────────


class TestMartingaleSizer:
    def test_first_level(self, signal, state, prices):
        sizer = MartingaleSizer(base_size=100.0, multiplier=1.5)
        notional = sizer.compute_size(signal, state, prices)
        # Level 0: base_size * 1.5^0 = 100
        assert notional == 100.0

    def test_second_level(self, signal, state, prices):
        # Add existing position
        pos = Position(
            id="pos1",
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            qty=0.001,
            entry_price=49000.0,
        )
        state.portfolio.positions["pos1"] = pos

        sizer = MartingaleSizer(base_size=100.0, multiplier=1.5)
        notional = sizer.compute_size(signal, state, prices)
        # Level 1: base_size * 1.5^1 = 150
        assert notional == 150.0

    def test_third_level(self, signal, state, prices):
        # Add two existing positions
        for i in range(2):
            pos = Position(
                id=f"pos{i}",
                pair="BTCUSDT",
                position_type=PositionType.LONG,
                qty=0.001,
                entry_price=49000.0 - i * 1000,
            )
            state.portfolio.positions[f"pos{i}"] = pos

        sizer = MartingaleSizer(base_size=100.0, multiplier=1.5)
        notional = sizer.compute_size(signal, state, prices)
        # Level 2: base_size * 1.5^2 = 225
        assert notional == 225.0

    def test_max_grid_levels(self, signal, state, prices):
        # Add max_grid_levels positions
        for i in range(5):
            pos = Position(
                id=f"pos{i}",
                pair="BTCUSDT",
                position_type=PositionType.LONG,
                qty=0.001,
                entry_price=49000.0 - i * 1000,
            )
            state.portfolio.positions[f"pos{i}"] = pos

        sizer = MartingaleSizer(base_size=100.0, max_grid_levels=5)
        notional = sizer.compute_size(signal, state, prices)
        # Already at max levels
        assert notional == 0.0

    def test_different_pair_not_counted(self, signal, state, prices):
        # Add position in different pair
        pos = Position(
            id="pos_eth",
            pair="ETHUSDT",
            position_type=PositionType.LONG,
            qty=1.0,
            entry_price=3000.0,
        )
        state.portfolio.positions["pos_eth"] = pos

        sizer = MartingaleSizer(base_size=100.0, multiplier=1.5)
        notional = sizer.compute_size(signal, state, prices)
        # ETHUSDT position shouldn't affect BTCUSDT grid level
        assert notional == 100.0  # Still level 0

    def test_max_notional_cap(self, signal, state, prices):
        # Add many positions to get high multiplier
        for i in range(4):
            pos = Position(
                id=f"pos{i}",
                pair="BTCUSDT",
                position_type=PositionType.LONG,
                qty=0.001,
                entry_price=49000.0,
            )
            state.portfolio.positions[f"pos{i}"] = pos

        sizer = MartingaleSizer(
            base_size=100.0,
            multiplier=2.0,
            max_grid_levels=10,
            max_notional=500.0,
        )
        notional = sizer.compute_size(signal, state, prices)
        # Level 4: 100 * 2^4 = 1600, but capped at 500
        assert notional == 500.0


# ── Edge Cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_equity(self, signal, prices):
        state = StrategyState(strategy_id="test")
        state.portfolio.cash = 0.0

        sizer = FixedFractionSizer(fraction=0.02)
        notional = sizer.compute_size(signal, state, prices)
        assert notional == 0.0

    def test_zero_price(self, state, prices):
        signal = SignalContext(
            pair="BTCUSDT",
            signal_type="rise",
            probability=0.7,
            price=0.0,
        )
        sizer = VolatilityTargetSizer()
        state.runtime["atr"] = {"BTCUSDT": 1000.0}
        notional = sizer.compute_size(signal, state, prices)
        # Should use default volatility
        assert notional >= 0

    def test_negative_probability(self, state, prices):
        signal = SignalContext(
            pair="BTCUSDT",
            signal_type="rise",
            probability=-0.5,
            price=50000.0,
        )
        sizer = SignalStrengthSizer(base_size=100.0, min_probability=0.0)
        notional = sizer.compute_size(signal, state, prices)
        # Negative probability -> 0 or negative, should be filtered
        assert notional <= 0


# ── Interface Compliance ──────────────────────────────────────────────────


class TestInterface:
    def test_all_sizers_have_compute_size(self):
        sizers = [
            FixedFractionSizer(),
            SignalStrengthSizer(),
            KellyCriterionSizer(),
            VolatilityTargetSizer(),
            RiskParitySizer(),
            MartingaleSizer(),
        ]
        for sizer in sizers:
            assert hasattr(sizer, "compute_size")
            assert callable(sizer.compute_size)

    def test_all_sizers_return_float(self, signal, state, prices):
        sizers = [
            FixedFractionSizer(),
            SignalStrengthSizer(),
            KellyCriterionSizer(),
            VolatilityTargetSizer(),
            RiskParitySizer(),
            MartingaleSizer(),
        ]
        for sizer in sizers:
            result = sizer.compute_size(signal, state, prices)
            assert isinstance(result, float)
