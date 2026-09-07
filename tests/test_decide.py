"""Stateless per-bar decisions (`Flow.decide`) and the trailing-window buffer."""

import polars as pl
import pytest

import signalflow as sf


def _flow():
    return sf.Flow(
        name="dec",
        detectors=[sf.SmaCrossDetector(fast=3, slow=20)],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.5), exit=sf.Exit(tp=0.01, sl=0.01)),
    )


@pytest.fixture(scope="module")
def ds():
    return sf.dataset("synthetic", pairs=["BTCUSDT", "ETHUSDT"], start="2024-01-01", end="2024-01-20", interval="1h")


def test_buffer_keeps_the_last_window_bars(ds):
    buf = sf.Buffer(window=10).push(ds.frame)
    assert buf.bars == 10
    last_ts = ds.frame.get_column("ts").unique().sort()[-10:]
    assert buf.frame.get_column("ts").unique().sort().to_list() == last_ts.to_list()
    assert buf.frame.height == 20  # two pairs per bar
    assert buf.ts == last_ts[-1]


def test_buffer_pushes_one_bar_at_a_time(ds):
    buf = sf.Buffer(window=5)
    for bar in ds.iter_bars():
        buf.push(bar.frame)
    assert buf.bars == 5 and buf.frame.height == 10
    assert buf.ts == ds.frame.get_column("ts").max()


def test_flow_buffer_is_sized_from_required_warmup():
    flow = _flow()
    assert flow.buffer().window == flow.required_warmup + 1
    assert flow.buffer(window=7).window == 7


def test_decide_matches_the_live_loop(ds):
    """Buffer + decide + SimBroker + Engine, driven by hand, reproduce simulate()."""
    flow = _flow()
    sim = flow.simulate(ds, capital=10_000)

    feed = sf.ReplayFeed(ds, warmup_bars=flow.required_warmup)
    broker, engine = sf.SimBroker(quote="USDT"), sf.Engine(10_000, target="USDT")
    buf = flow.buffer()
    for bar in feed.warmup().iter_bars():
        buf.push(bar.frame)
    peak, fills_total = 0.0, 0
    for bar in feed.stream():
        buf.push(bar.frame)
        snap = engine.snapshot(bar.ts, bar.prices)
        peak = max(peak, snap.equity)
        d = flow.decide(buf, snap, bar.ts, peak=peak)
        fills = broker.execute(d.orders, bar)
        engine.apply(fills)
        fills_total += len(fills)

    assert fills_total == len(sim.fills) > 0
    assert engine.equity(bar.prices) == pytest.approx(sim.final_equity)


def test_decide_is_pure_and_accepts_a_dataset(ds):
    flow = _flow()
    engine = sf.Engine(10_000, target="USDT")
    ts = ds.frame.get_column("ts").max()
    prices = dict(ds.frame.filter(pl.col("ts") == ts).select(["pair", "close"]).iter_rows())
    snap = engine.snapshot(ts, prices)

    first = flow.decide(ds, snap)
    second = flow.decide(ds, snap)
    assert first.ts == ts == second.ts
    assert first.signals.equals(second.signals)
    assert [(o.pair, o.side, o.qty) for o in first.orders] == [(o.pair, o.side, o.qty) for o in second.orders]
    assert engine.snapshot(ts, prices).equity == snap.equity  # nothing was applied


def test_decide_signals_are_the_bar_slice(ds):
    flow = _flow()
    engine = sf.Engine(10_000, target="USDT")
    buf = flow.buffer().push(ds.frame)
    d = flow.decide(buf, engine.snapshot(buf.ts, {}), prices=None)
    assert set(d.signals.get_column("ts").to_list()) <= {buf.ts}
    assert {"pair", "ts", "signal"} <= set(d.signals.columns)


def test_decide_rejects_empty_history():
    flow = _flow()
    with pytest.raises(ValueError):
        flow.decide(sf.Buffer(5), None)


def test_sim_broker_fills_at_external_prices(ds):
    bar = next(iter(ds.iter_bars()))
    broker = sf.SimBroker(quote="USDT", slippage=0.001, fee_rate=0.0)
    order = sf.Order("BTCUSDT", sf.Side.BUY, 0.5, ts=bar.ts)
    fills = broker.execute([order], bar, prices={"BTCUSDT": 200_000.0})
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(200_000.0 * 1.001)
    assert fills[0].qty == 0.5
    assert broker.execute([order], bar)[0].price == pytest.approx(bar.prices["BTCUSDT"] * 1.001)
