"""Reconciliation port: internal event log vs exchange event log (§3).

In live trading the exchange is the source of truth about fills. Without
reconciliation the internal portfolio silently drifts from reality, which by
definition breaks parity. Reconciliation is therefore constitutive of "live
mode exists", not an optional add-on.

Following the port/adapter convention of ``signalflow.models`` (``Resolver`` /
``ModelRegistry``): a minimal ``Reconciler`` Protocol plus a concrete adapter.
Crucially it operates over the SINGLE canonical model — ``core.Trade`` event
logs folded via ``core.eventlog`` — not a parallel position model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from signalflow.core import Trade

# Numeric fields compared when the same trade id appears on both sides.
_COMPARED_FIELDS = ("pair", "side", "price", "qty", "fee")


class RecoveryMode(StrEnum):
    """How to recover internal state after a restart."""

    SYNC = "sync"  # reconcile internal state against the exchange
    RESTORE = "restore"  # restore from persistence only
    CLOSE_ALL = "close_all"  # close all positions
    MANUAL = "manual"  # require manual intervention


class OrphanPositionAction(StrEnum):
    """What to do with positions found on the exchange but not internally."""

    CLOSE = "close"
    ADOPT = "adopt"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class TradeMismatch:
    """A trade present on both sides but with differing fields."""

    trade_id: str
    internal: Trade
    exchange: Trade
    fields: tuple[str, ...]


@dataclass(slots=True)
class ReconcileResult:
    """Outcome of merging the internal and exchange logs.

    Attributes:
        matched: Trade ids present and equal on both sides.
        only_internal: Trades the internal log has that the exchange does not
            (e.g. an order we think filled but the exchange never confirmed).
        only_exchange: Trades the exchange reports that we have not recorded
            (orphan fills — the dangerous case for silent drift).
        mismatched: Trades present on both sides but with differing fields.
    """

    matched: list[str] = field(default_factory=list)
    only_internal: list[Trade] = field(default_factory=list)
    only_exchange: list[Trade] = field(default_factory=list)
    mismatched: list[TradeMismatch] = field(default_factory=list)

    @property
    def in_sync(self) -> bool:
        """True when the two logs agree completely."""
        return not (self.only_internal or self.only_exchange or self.mismatched)


@runtime_checkable
class Reconciler(Protocol):
    """Port: reconcile an internal trade log against an exchange trade log."""

    def reconcile(self, internal: list[Trade], exchange: list[Trade]) -> ReconcileResult:
        """Return the discrepancies between the two event logs."""
        ...


def _trades_differ(a: Trade, b: Trade, *, tol: float) -> tuple[str, ...]:
    diffs: list[str] = []
    for f in _COMPARED_FIELDS:
        av, bv = getattr(a, f), getattr(b, f)
        if isinstance(av, float) or isinstance(bv, float):
            if abs(float(av) - float(bv)) > tol:
                diffs.append(f)
        elif av != bv:
            diffs.append(f)
    return tuple(diffs)


@dataclass(slots=True)
class LogMergeReconciler:
    """Adapter: reconcile by merging both logs on ``Trade.id``.

    Offline / paper-grade implementation: matches trades by id, then classifies
    each as matched / only-internal / only-exchange / mismatched. Numeric fields
    are compared within ``tol`` (exchange fees/prices may round differently).
    """

    tol: float = 1e-9

    def reconcile(self, internal: list[Trade], exchange: list[Trade]) -> ReconcileResult:
        internal_by_id = {t.id: t for t in internal}
        exchange_by_id = {t.id: t for t in exchange}

        result = ReconcileResult()

        for tid, itrade in internal_by_id.items():
            etrade = exchange_by_id.get(tid)
            if etrade is None:
                result.only_internal.append(itrade)
                continue
            diffs = _trades_differ(itrade, etrade, tol=self.tol)
            if diffs:
                result.mismatched.append(TradeMismatch(tid, itrade, etrade, diffs))
            else:
                result.matched.append(tid)

        for tid, etrade in exchange_by_id.items():
            if tid not in internal_by_id:
                result.only_exchange.append(etrade)

        return result
