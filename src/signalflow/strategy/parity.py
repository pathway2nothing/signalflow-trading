"""ParityCheck: research-to-production parity as a verifiable component (§2).

``FeatureSpec.feature_hash`` catches drift in the *recipe* of features
(train vs serve). ``ParityCheck`` catches drift in *execution* (backtest vs
paper vs live): the same config + the same data, run in two modes, must produce
the same SEQUENCE of events (trades) — not merely the same final metrics.

Each component (a trade field, or the presence of a trade) is classified:
    - EXACT        — must match byte/value-for-value (side, pair, qty, type);
    - APPROXIMATE  — must match within a declared tolerance (price, fee);
    - OUT_OF_SCOPE — deliberately not compared (e.g. live-only latency).

``ParityCheck`` fails when a component leaves its declared class — exactly as a
``feature_hash`` mismatch refuses to load a model. Folding the event log (see
``core.eventlog``) gives an ordered sequence, so the FIRST divergence is
localised precisely (a snapshot-only comparison could only see the endpoints).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from signalflow.core import RawData, Signals, StrategyState, Trade


class ComponentClass(StrEnum):
    """Declared parity class of a component."""

    EXACT = "exact"
    APPROXIMATE = "approximate"
    OUT_OF_SCOPE = "out_of_scope"


# Trade fields each component name maps to. "type" reads ``Trade.meta['type']``
# and "presence" is the synthetic component for "a trade exists at this index".
_PRESENCE = "presence"


@dataclass(frozen=True, slots=True)
class ParitySpec:
    """Declares the parity class (and tolerance) of each compared component.

    Attributes:
        components: component name -> ComponentClass. Component names are trade
            fields (``side``/``pair``/``qty``/``price``/``fee``/``type``) plus
            the synthetic ``presence``.
        tolerances: component name -> absolute tolerance (for APPROXIMATE).
    """

    components: dict[str, ComponentClass]
    tolerances: dict[str, float] = field(default_factory=dict)

    def parity_hash(self) -> str:
        """Stable hash of the declaration (drift-protects the spec itself)."""
        payload = {
            "components": {k: str(v) for k, v in sorted(self.components.items())},
            "tolerances": {k: round(v, 12) for k, v in sorted(self.tolerances.items())},
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def class_of(self, component: str) -> ComponentClass:
        return self.components.get(component, ComponentClass.EXACT)


def default_parity_spec(*, price_tol: float = 1e-9, fee_tol: float = 1e-9) -> ParitySpec:
    """The conventional spec: structure exact, price/fee approximate."""
    return ParitySpec(
        components={
            _PRESENCE: ComponentClass.EXACT,
            "side": ComponentClass.EXACT,
            "pair": ComponentClass.EXACT,
            "qty": ComponentClass.EXACT,
            "type": ComponentClass.EXACT,
            "price": ComponentClass.APPROXIMATE,
            "fee": ComponentClass.APPROXIMATE,
        },
        tolerances={"price": price_tol, "fee": fee_tol},
    )


@dataclass(frozen=True, slots=True)
class ParityDivergence:
    """The first point where the two runs left a component's declared class."""

    index: int
    ts: datetime | None
    component: str
    component_class: ComponentClass
    value_a: Any
    value_b: Any


@dataclass(slots=True)
class ParityResult:
    """Outcome of a parity run."""

    diverged: bool
    first_divergence: ParityDivergence | None
    n_events_a: int
    n_events_b: int

    def assert_in_class(self) -> None:
        """Raise if any component left its declared class (analog of verify_hash)."""
        if not self.diverged:
            return
        d = self.first_divergence
        assert d is not None
        raise RuntimeError(
            f"Parity violation: component '{d.component}' (declared {d.component_class}) "
            f"diverged at event {d.index} (ts={d.ts}): {d.value_a!r} != {d.value_b!r}"
        )


def _trade_field(trade: Trade, component: str) -> Any:
    if component == "type":
        return trade.meta.get("type")
    return getattr(trade, component)


class ParityCheck:
    """Run one config in two modes and verify the event sequences agree."""

    def __init__(self, spec: ParitySpec | None = None) -> None:
        self.spec = spec or default_parity_spec()

    @staticmethod
    def _extract_trades(runner: Any, raw_data: RawData, signals: Signals, state: StrategyState | None) -> list[Trade]:
        runner.run(raw_data, signals, state)
        trades = getattr(runner, "trades", None)
        if trades is None:
            trades = getattr(runner, "_trades", [])
        return list(trades)

    def compare(self, trades_a: list[Trade], trades_b: list[Trade]) -> ParityResult:
        """Diff two trade sequences event-by-event; report the FIRST divergence."""
        n = max(len(trades_a), len(trades_b))
        for i in range(n):
            ta = trades_a[i] if i < len(trades_a) else None
            tb = trades_b[i] if i < len(trades_b) else None

            # Presence: does a trade exist at this index on both sides?
            if (ta is None) != (tb is None):
                cls = self.spec.class_of(_PRESENCE)
                ts = (ta or tb).ts if (ta or tb) else None
                if cls is not ComponentClass.OUT_OF_SCOPE:
                    return ParityResult(
                        True,
                        ParityDivergence(i, ts, _PRESENCE, cls, ta is not None, tb is not None),
                        len(trades_a),
                        len(trades_b),
                    )
                continue

            assert ta is not None and tb is not None
            for component, cls in self.spec.components.items():
                if component == _PRESENCE or cls is ComponentClass.OUT_OF_SCOPE:
                    continue
                va, vb = _trade_field(ta, component), _trade_field(tb, component)
                if cls is ComponentClass.APPROXIMATE:
                    tol = self.spec.tolerances.get(component, 0.0)
                    if abs(float(va) - float(vb)) > tol:
                        return ParityResult(
                            True, ParityDivergence(i, ta.ts, component, cls, va, vb), len(trades_a), len(trades_b)
                        )
                else:  # EXACT
                    if va != vb:
                        return ParityResult(
                            True, ParityDivergence(i, ta.ts, component, cls, va, vb), len(trades_a), len(trades_b)
                        )

        return ParityResult(False, None, len(trades_a), len(trades_b))

    def run(
        self,
        *,
        runner_a: Any,
        runner_b: Any,
        raw_data: RawData,
        signals: Signals,
        initial_state: StrategyState | None = None,
    ) -> ParityResult:
        """Run both runners on the same data and compare their event logs."""
        trades_a = self._extract_trades(runner_a, raw_data, signals, initial_state)
        trades_b = self._extract_trades(runner_b, raw_data, signals, initial_state)
        return self.compare(trades_a, trades_b)

    def assert_parity(
        self,
        *,
        runner_a: Any,
        runner_b: Any,
        raw_data: RawData,
        signals: Signals,
        initial_state: StrategyState | None = None,
    ) -> ParityResult:
        """Run and raise if parity is violated; return the result otherwise."""
        result = self.run(
            runner_a=runner_a,
            runner_b=runner_b,
            raw_data=raw_data,
            signals=signals,
            initial_state=initial_state,
        )
        result.assert_in_class()
        return result
