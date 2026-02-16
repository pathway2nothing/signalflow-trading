"""Tests for strategy model decision types."""

from datetime import datetime

import polars as pl
import pytest

from signalflow.core.containers.position import Position
from signalflow.core.containers.signals import Signals
from signalflow.core.enums import PositionType
from signalflow.strategy.model.context import ModelContext
from signalflow.strategy.model.decision import StrategyAction, StrategyDecision

TS = datetime(2024, 1, 1)


# ── StrategyDecision validation ─────────────────────────────────────────────


class TestStrategyDecision:
    def test_valid_enter_decision(self):
        decision = StrategyDecision(
            action=StrategyAction.ENTER,
            pair="BTCUSDT",
            size_multiplier=1.5,
            confidence=0.8,
        )
        assert decision.action == StrategyAction.ENTER
        assert decision.size_multiplier == 1.5
        assert decision.confidence == 0.8
        assert decision.position_id is None

    def test_enter_requires_positive_size_multiplier(self):
        with pytest.raises(ValueError, match="size_multiplier must be positive"):
            StrategyDecision(
                action=StrategyAction.ENTER,
                pair="BTCUSDT",
                size_multiplier=0,
            )

    def test_enter_negative_size_multiplier_fails(self):
        with pytest.raises(ValueError, match="size_multiplier must be positive"):
            StrategyDecision(
                action=StrategyAction.ENTER,
                pair="BTCUSDT",
                size_multiplier=-0.5,
            )

    def test_close_requires_position_id(self):
        with pytest.raises(ValueError, match="CLOSE action requires position_id"):
            StrategyDecision(
                action=StrategyAction.CLOSE,
                pair="BTCUSDT",
            )

    def test_valid_close_decision(self):
        decision = StrategyDecision(
            action=StrategyAction.CLOSE,
            pair="BTCUSDT",
            position_id="pos_123",
            confidence=0.9,
        )
        assert decision.action == StrategyAction.CLOSE
        assert decision.position_id == "pos_123"

    def test_close_all_no_position_id_required(self):
        decision = StrategyDecision(
            action=StrategyAction.CLOSE_ALL,
            pair="BTCUSDT",
        )
        assert decision.position_id is None
        assert decision.action == StrategyAction.CLOSE_ALL

    def test_skip_decision(self):
        decision = StrategyDecision(
            action=StrategyAction.SKIP,
            pair="BTCUSDT",
        )
        assert decision.action == StrategyAction.SKIP

    def test_hold_decision(self):
        decision = StrategyDecision(
            action=StrategyAction.HOLD,
            pair="BTCUSDT",
        )
        assert decision.action == StrategyAction.HOLD

    def test_decision_with_meta(self):
        decision = StrategyDecision(
            action=StrategyAction.ENTER,
            pair="BTCUSDT",
            meta={"model": "rf_v2", "reason": "high_confidence"},
        )
        assert decision.meta["model"] == "rf_v2"
        assert decision.meta["reason"] == "high_confidence"

    def test_decision_is_frozen(self):
        decision = StrategyDecision(
            action=StrategyAction.ENTER,
            pair="BTCUSDT",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            decision.pair = "ETHUSDT"


# ── StrategyAction enum ─────────────────────────────────────────────────────


class TestStrategyAction:
    def test_action_values(self):
        assert StrategyAction.ENTER.value == "enter"
        assert StrategyAction.SKIP.value == "skip"
        assert StrategyAction.CLOSE.value == "close"
        assert StrategyAction.CLOSE_ALL.value == "close_all"
        assert StrategyAction.HOLD.value == "hold"

    def test_action_is_str(self):
        # StrategyAction inherits from str
        assert isinstance(StrategyAction.ENTER, str)
        assert StrategyAction.ENTER == "enter"


# ── ModelContext ────────────────────────────────────────────────────────────


class TestModelContext:
    def test_context_creation(self):
        signals = Signals(
            pl.DataFrame(
                {
                    "pair": ["BTCUSDT"],
                    "timestamp": [TS],
                    "signal_type": ["rise"],
                    "signal": [1],
                    "probability": [0.8],
                }
            )
        )
        positions = [
            Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=50000.0, qty=0.1),
        ]
        context = ModelContext(
            timestamp=TS,
            signals=signals,
            prices={"BTCUSDT": 51000.0},
            positions=positions,
            metrics={"equity": 10000.0, "max_drawdown": 0.05},
            runtime={"regime": "trend_up"},
        )

        assert context.timestamp == TS
        assert context.signals.value.height == 1
        assert context.prices["BTCUSDT"] == 51000.0
        assert len(context.positions) == 1
        assert context.metrics["equity"] == 10000.0
        assert context.runtime["regime"] == "trend_up"

    def test_context_is_frozen(self):
        signals = Signals(pl.DataFrame())
        context = ModelContext(
            timestamp=TS,
            signals=signals,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            context.timestamp = datetime.now()

    def test_context_with_empty_signals(self):
        signals = Signals(pl.DataFrame())
        context = ModelContext(
            timestamp=TS,
            signals=signals,
        )
        assert context.signals.value.height == 0
