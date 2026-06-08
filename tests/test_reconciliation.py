"""Tests for the live reconciliation port (strategy/live/reconciliation.py)."""

from datetime import datetime

from signalflow.core import Trade
from signalflow.strategy.live import LogMergeReconciler, Reconciler

TS = datetime(2024, 1, 1, 10)


def _trade(tid, price=100.0, qty=1.0, fee=0.1, side="BUY"):
    return Trade(id=tid, position_id="p1", pair="BTCUSDT", side=side, ts=TS, price=price, qty=qty, fee=fee)


def test_adapter_satisfies_port():
    assert isinstance(LogMergeReconciler(), Reconciler)


def test_identical_logs_in_sync():
    rec = LogMergeReconciler()
    internal = [_trade("t1"), _trade("t2", side="SELL")]
    exchange = [_trade("t1"), _trade("t2", side="SELL")]
    result = rec.reconcile(internal, exchange)
    assert result.in_sync
    assert sorted(result.matched) == ["t1", "t2"]


def test_orphan_exchange_fill_detected():
    rec = LogMergeReconciler()
    internal = [_trade("t1")]
    exchange = [_trade("t1"), _trade("t2")]  # exchange knows a fill we don't
    result = rec.reconcile(internal, exchange)
    assert not result.in_sync
    assert [t.id for t in result.only_exchange] == ["t2"]


def test_missing_exchange_confirmation_detected():
    rec = LogMergeReconciler()
    internal = [_trade("t1"), _trade("t2")]
    exchange = [_trade("t1")]  # we think t2 filled; exchange disagrees
    result = rec.reconcile(internal, exchange)
    assert [t.id for t in result.only_internal] == ["t2"]


def test_field_mismatch_localized():
    rec = LogMergeReconciler()
    internal = [_trade("t1", price=100.0, fee=0.1)]
    exchange = [_trade("t1", price=100.5, fee=0.2)]  # exchange filled at a different price/fee
    result = rec.reconcile(internal, exchange)
    assert not result.in_sync
    assert len(result.mismatched) == 1
    mm = result.mismatched[0]
    assert mm.trade_id == "t1"
    assert set(mm.fields) == {"price", "fee"}


def test_fee_within_tolerance_is_match():
    rec = LogMergeReconciler(tol=1e-6)
    internal = [_trade("t1", fee=0.10000000001)]
    exchange = [_trade("t1", fee=0.1)]
    result = rec.reconcile(internal, exchange)
    assert result.in_sync
