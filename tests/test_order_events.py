"""Order-lifecycle events in the live loop and venue reconciliation on resume."""

from dataclasses import dataclass

import pytest

import signalflow as sf
from signalflow.engine.types import OrderEvent
from signalflow.enums import Side
from signalflow.flow.live import load_state, run_live_loop, save_state


@pytest.fixture(scope="module")
def small_ds():
    return sf.data("memory", pairs=["BTCUSDT"], start="2024-01-01", end="2024-01-08", interval="1h")


def _flow():
    return sf.Flow(
        name="oe",
        detectors=[sf.SmaCrossDetector(fast=3, slow=8)],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.9), exit=sf.Exit(tp=0.005, sl=0.005)),
    )


@dataclass
class _FakeVenue(sf.BinanceBroker):
    """Never fills during the run; query_order reports a completed fill."""

    def execute(self, orders, bar):
        return []

    def query_order(self, pair, client_order_id):
        return {"executedQty": "1.0", "cummulativeQuoteQty": "100.0", "fills": []}


def test_order_log_placed_result_pairs(small_ds):
    captured = {}

    def on_bar(engine, bar, fills, latency):
        captured["engine"] = engine

    run = run_live_loop(_flow(), sf.ReplayFeed(small_ds), 10_000.0, sf.SimBroker(), on_bar=on_bar)
    engine = captured["engine"]
    placed = [e for e in engine.order_log if e.kind == "placed"]
    results = [e for e in engine.order_log if e.kind == "result"]
    assert placed
    assert [e.client_order_id for e in placed] == [e.client_order_id for e in results]

    last_ts = small_ds.frame.get_column("ts").max()
    fold_eq = sf.Engine.fold(run.fills, 10_000.0).equity(small_ds.prices_at(last_ts))
    assert run.final_equity == pytest.approx(fold_eq, abs=1e-6)


def test_state_preserves_order_log(tmp_path):
    eng = sf.Engine(10_000.0)
    eng.record_order(OrderEvent("sf-abc", "BTCUSDT", Side.BUY, 1.0, "2024-01-01T00:00:00", "placed"))
    eng.record_order(OrderEvent("sf-abc", "BTCUSDT", Side.BUY, 1.0, "2024-01-01T00:00:00", "result", "filled"))
    path = str(tmp_path / "book.json")
    save_state(eng, path)
    restored = sf.Engine(10_000.0)
    load_state(restored, path)
    assert len(restored.order_log) == 2
    assert restored.order_log[0].client_order_id == "sf-abc"
    assert restored.order_log[1].status == "filled"


def _dangling_state(tmp_path):
    order = sf.Order("BTCUSDT", Side.BUY, 1.0, ts="2024-01-01T00:00:00")
    cid = sf.BinanceBroker.client_order_id(order)
    eng = sf.Engine(10_000.0)
    eng.record_order(OrderEvent(cid, "BTCUSDT", Side.BUY, 1.0, "2024-01-01T00:00:00", "placed"))
    path = str(tmp_path / "book.json")
    save_state(eng, path)
    return path, cid


def test_reconciliation_synthesizes_missing_fill(small_ds, tmp_path):
    path, cid = _dangling_state(tmp_path)
    captured = {}

    def on_bar(engine, bar, fills, latency):
        captured["engine"] = engine

    run = run_live_loop(
        _flow(),
        sf.ReplayFeed(small_ds),
        10_000.0,
        _FakeVenue(api_key="k", api_secret="s"),
        armed=True,
        state_path=path,
        on_bar=on_bar,
        max_bars=1,
    )
    engine = captured["engine"]
    resolved = [e for e in engine.order_log if e.kind == "result" and e.client_order_id == cid]
    assert resolved and resolved[0].status == "filled"
    assert any(f.pair == "BTCUSDT" and f.qty == pytest.approx(1.0) for f in run.fills)


def test_reconciliation_venue_error_still_starts(small_ds, tmp_path):
    path, _ = _dangling_state(tmp_path)

    @dataclass
    class _BadVenue(_FakeVenue):
        def query_order(self, pair, client_order_id):
            raise RuntimeError("venue unreachable")

    run = run_live_loop(
        _flow(),
        sf.ReplayFeed(small_ds),
        10_000.0,
        _BadVenue(api_key="k", api_secret="s"),
        armed=True,
        state_path=path,
        max_bars=1,
    )
    assert run.equity_curve.height > 0
