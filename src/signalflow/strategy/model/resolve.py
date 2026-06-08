"""Pinned resolution of strategy/RL models (REFACTOR_PLAN.md §4.1).

Forecast models already travel through an immutable ``ModelRef`` resolved via a
``ModelRegistry`` (with floating ``latest`` versions forbidden). Strategy/RL
models used by ``ModelEntryRule`` / ``ModelExitRule`` historically took an
arbitrary already-loaded object (``load_model(path)``), bypassing the
pinned/fit-free invariant. This helper lets those rules resolve their model the
same way — by pinned ``ModelRef`` — so backtest and live load the identical,
version-locked artifact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from signalflow.strategy.model.protocol import StrategyModel

if TYPE_CHECKING:
    from signalflow.models.model_ref import ModelRef
    from signalflow.models.registry import ModelRegistry


def resolve_strategy_model(
    model: StrategyModel | None,
    model_ref: ModelRef | None,
    registry: ModelRegistry | None,
) -> StrategyModel | None:
    """Return the strategy model, resolving a pinned ``ModelRef`` if needed.

    Precedence: an explicitly-injected ``model`` wins; otherwise the model is
    resolved from ``registry.get(model_ref)``. The ``ModelRef`` itself forbids
    floating versions (``latest``) at construction, so parity is preserved.

    Raises:
        ValueError: if ``model_ref`` is given without a ``registry``.
        TypeError: if the resolved object does not satisfy ``StrategyModel``.
    """
    if model is not None:
        return model
    if model_ref is None:
        return None
    if registry is None:
        raise ValueError("model_ref given without a ModelRegistry to resolve it")
    resolved = registry.get(model_ref)
    if not isinstance(resolved, StrategyModel):
        raise TypeError(f"Model resolved from {model_ref.uri!r} does not implement StrategyModel.decide()")
    return resolved
