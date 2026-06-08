"""Tests for §4.1 (pinned strategy/RL models) and §4.2 (warmup contract)."""

import pytest

from signalflow.core import assert_warmup_consistency, required_warmup_bars, warmup_bars_of
from signalflow.models.model_ref import ModelRef
from signalflow.strategy.model import StrategyAction, StrategyDecision, resolve_strategy_model

# ── §4.1 pinned strategy models ─────────────────────────────────────────────


class _FakeModel:
    def decide(self, context):  # satisfies StrategyModel protocol
        return [StrategyDecision(action=StrategyAction.SKIP, pair="BTCUSDT")]


class _FakeRegistry:
    def __init__(self, model):
        self._model = model
        self.requested: list[ModelRef] = []

    def get(self, ref):
        self.requested.append(ref)
        return self._model

    def has(self, ref):
        return True


class TestPinnedStrategyModel:
    def test_explicit_model_wins(self):
        m = _FakeModel()
        assert resolve_strategy_model(m, None, None) is m

    def test_resolves_via_registry_by_ref(self):
        m = _FakeModel()
        reg = _FakeRegistry(m)
        ref = ModelRef(name="rl_exit", version="3")
        resolved = resolve_strategy_model(None, ref, reg)
        assert resolved is m
        assert reg.requested == [ref]

    def test_ref_without_registry_raises(self):
        ref = ModelRef(name="rl_exit", version="3")
        with pytest.raises(ValueError, match="without a ModelRegistry"):
            resolve_strategy_model(None, ref, None)

    def test_resolved_non_model_raises(self):
        class NotAModel:
            pass

        reg = _FakeRegistry(NotAModel())
        ref = ModelRef(name="rl_exit", version="3")
        with pytest.raises(TypeError, match="does not implement StrategyModel"):
            resolve_strategy_model(None, ref, reg)

    def test_floating_version_forbidden(self):
        # The pinned invariant is inherited from ModelRef itself.
        with pytest.raises(ValueError, match="latest"):
            ModelRef(name="rl_exit", version="latest")

    def test_none_everywhere_returns_none(self):
        assert resolve_strategy_model(None, None, None) is None


# ── §4.2 warmup contract ────────────────────────────────────────────────────


class _WarmupBars:
    warmup_bars = 50


class _WarmupProp:
    @property
    def warmup(self):
        return 30


class _WarmupMethod:
    def warmup_bars(self):
        return 70


class TestWarmupContract:
    def test_warmup_bars_attr(self):
        assert warmup_bars_of(_WarmupBars()) == 50

    def test_warmup_property(self):
        assert warmup_bars_of(_WarmupProp()) == 30

    def test_warmup_callable(self):
        assert warmup_bars_of(_WarmupMethod()) == 70

    def test_none_is_zero(self):
        assert warmup_bars_of(None) == 0

    def test_required_is_max_with_floor(self):
        comps = [_WarmupBars(), _WarmupProp()]  # 50, 30
        assert required_warmup_bars(comps, floor=10) == 50
        assert required_warmup_bars(comps, floor=100) == 100  # floor wins

    def test_required_flattens_and_handles_scalars(self):
        assert required_warmup_bars(_WarmupMethod(), [_WarmupBars()]) == 70

    def test_assert_consistency_ok(self):
        assert_warmup_consistency(100, 100)  # no raise

    def test_assert_consistency_mismatch_raises(self):
        with pytest.raises(ValueError, match="Warmup window mismatch"):
            assert_warmup_consistency(100, 120)
