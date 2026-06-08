"""Live-trading work-in-progress (not yet wired into execution).

This package parks components that only become load-bearing once live trading
is implemented: durable state persistence (``state.StateManager`` and its
Redis/DuckDB backends) and exchange ``reconciliation``.

Status / REFACTOR_PLAN.md notes:
    - ``state.py`` is orphaned (no callers). It still defines its OWN
      ``Position`` model (side/size/sl/tp), distinct from the canonical
      ``signalflow.core.Position``. This duplicate is the architectural flaw
      called out in §3 and must be migrated to ``core.Position`` when live
      trading is wired. Until then it is quarantined here and intentionally
      NOT re-exported from ``signalflow.strategy``.
    - ``reconciliation.py`` is the §3 port: it verifies the internal event log
      against the exchange event log over the canonical ``core`` model.
"""

from signalflow.strategy.live.reconciliation import (
    LogMergeReconciler,
    OrphanPositionAction,
    Reconciler,
    ReconcileResult,
    RecoveryMode,
)

__all__ = [
    "LogMergeReconciler",
    "OrphanPositionAction",
    "ReconcileResult",
    "Reconciler",
    "RecoveryMode",
]
