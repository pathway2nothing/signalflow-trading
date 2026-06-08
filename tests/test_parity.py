"""Tests for ParityCheck (strategy/parity.py)."""

from datetime import datetime

import polars as pl
import pytest

from signalflow.core import RawData, Signals, Trade
from signalflow.strategy import ComponentClass, ParityCheck, ParitySpec, default_parity_spec
from signalflow.strategy.parity import _PRESENCE

TS = datetime(2024, 1, 1)


def _trade(tid, side="BUY", price=100.0, qty=1.0, fee=0.1, ttype="entry", ts=TS):
    return Trade(
        id=tid,
        position_id="p1",
        pair="BTCUSDT",
        side=side,
        ts=ts,
        price=price,
        qty=qty,
        fee=fee,
        meta={"type": ttype},
    )


class _FakeRunner:
    """Minimal runner stub exposing a preset trade log via ``.trades``."""

    def __init__(self, trades):
        self._preset = trades
        self.trades = []

    def run(self, raw_data, signals, state):
        self.trades = list(self._preset)
        return state


def _empty_raw():
    return RawData(datetime_start=TS, datetime_end=TS, pairs=["BTCUSDT"], data={"spot": pl.DataFrame()})


def _empty_signals():
    return Signals(pl.DataFrame())


# ── spec ──────────────────────────────────────────────────────────────────


def test_parity_hash_stable_and_order_independent():
    a = ParitySpec(components={"side": ComponentClass.EXACT, "price": ComponentClass.APPROXIMATE})
    b = ParitySpec(components={"price": ComponentClass.APPROXIMATE, "side": ComponentClass.EXACT})
    assert a.parity_hash() == b.parity_hash()


def test_parity_hash_changes_with_class():
    a = ParitySpec(components={"price": ComponentClass.APPROXIMATE})
    b = ParitySpec(components={"price": ComponentClass.EXACT})
    assert a.parity_hash() != b.parity_hash()


# ── compare ───────────────────────────────────────────────────────────────


class TestCompare:
    def test_identical_sequences_no_divergence(self):
        check = ParityCheck()
        seq = [_trade("t1"), _trade("t2", side="SELL", ttype="exit")]
        result = check.compare(seq, [_trade("t1"), _trade("t2", side="SELL", ttype="exit")])
        assert not result.diverged
        assert result.first_divergence is None

    def test_price_within_tolerance_ok(self):
        check = ParityCheck(default_parity_spec(price_tol=0.01))
        a = [_trade("t1", price=100.0)]
        b = [_trade("t1", price=100.005)]  # within tol
        assert not check.compare(a, b).diverged

    def test_price_beyond_tolerance_diverges(self):
        check = ParityCheck(default_parity_spec(price_tol=0.001))
        a = [_trade("t1", price=100.0)]
        b = [_trade("t1", price=100.5)]
        result = check.compare(a, b)
        assert result.diverged
        assert result.first_divergence.component == "price"
        assert result.first_divergence.component_class == ComponentClass.APPROXIMATE

    def test_exact_side_mismatch_diverges(self):
        check = ParityCheck()
        a = [_trade("t1", side="BUY")]
        b = [_trade("t1", side="SELL")]
        result = check.compare(a, b)
        assert result.diverged
        assert result.first_divergence.component == "side"

    def test_presence_divergence_localized(self):
        check = ParityCheck()
        a = [_trade("t1"), _trade("t2")]
        b = [_trade("t1")]  # missing second trade
        result = check.compare(a, b)
        assert result.diverged
        assert result.first_divergence.index == 1
        assert result.first_divergence.component == _PRESENCE

    def test_first_divergence_is_earliest(self):
        check = ParityCheck()
        a = [_trade("t1", side="BUY"), _trade("t2", side="BUY")]
        b = [_trade("t1", side="SELL"), _trade("t2", side="SELL")]
        result = check.compare(a, b)
        assert result.first_divergence.index == 0

    def test_out_of_scope_component_ignored(self):
        spec = ParitySpec(
            components={"side": ComponentClass.EXACT, "fee": ComponentClass.OUT_OF_SCOPE},
        )
        check = ParityCheck(spec)
        a = [_trade("t1", fee=0.1)]
        b = [_trade("t1", fee=99.0)]  # fee differs wildly but is out of scope
        assert not check.compare(a, b).diverged


# ── run / assert ──────────────────────────────────────────────────────────


class TestRun:
    def test_identical_runners_pass(self):
        seq = [_trade("t1"), _trade("t2", ttype="exit", side="SELL")]
        check = ParityCheck()
        result = check.run(
            runner_a=_FakeRunner(seq),
            runner_b=_FakeRunner(list(seq)),
            raw_data=_empty_raw(),
            signals=_empty_signals(),
        )
        assert not result.diverged

    def test_assert_parity_raises_on_divergence(self):
        check = ParityCheck()
        with pytest.raises(RuntimeError, match="Parity violation"):
            check.assert_parity(
                runner_a=_FakeRunner([_trade("t1", side="BUY")]),
                runner_b=_FakeRunner([_trade("t1", side="SELL")]),
                raw_data=_empty_raw(),
                signals=_empty_signals(),
            )

    def test_assert_parity_returns_result_when_ok(self):
        seq = [_trade("t1")]
        check = ParityCheck()
        result = check.assert_parity(
            runner_a=_FakeRunner(seq),
            runner_b=_FakeRunner(list(seq)),
            raw_data=_empty_raw(),
            signals=_empty_signals(),
        )
        assert result.diverged is False
