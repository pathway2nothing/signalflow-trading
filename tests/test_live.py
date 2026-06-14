"""Real-time loop: ReplayFeed drives the same decision core; state persists."""

import pytest

import signalflow as sf
from signalflow.engine.engine import Engine
from signalflow.flow.live import load_state, run_live_loop, save_state


@pytest.fixture(scope="module")
def small_ds():
    return sf.data("memory", pairs=["BTCUSDT"], start="2024-01-01", end="2024-01-08", interval="1h")


def _flow():
    return sf.Flow(
        name="live",
        detectors=[sf.SmaCrossDetector(fast=3, slow=8)],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.5), exit=sf.Exit(tp=0.01, sl=0.01)),
    )


def test_live_replay_runs_and_reconciles(small_ds):
    run = _flow().live(small_ds, capital=10_000)
    assert run.mode == "live"
    assert run.promotable is False
    assert run.equity_curve.height > 0
    last_ts = small_ds.frame.get_column("ts").max()
    fold_eq = Engine.fold(run.fills, 10_000.0).equity(small_ds.prices_at(last_ts))
    assert run.final_equity == pytest.approx(fold_eq, abs=1e-6)


def test_live_matches_backtest_on_same_data(small_ds):
    """Replayed live and backtest agree on final equity over identical data."""
    bt = _flow().backtest(small_ds, capital=10_000)
    live = _flow().live(small_ds, capital=10_000)
    assert live.final_equity == pytest.approx(bt.final_equity, abs=1e-6)


def test_state_persists_and_resumes(small_ds, tmp_path):
    path = str(tmp_path / "book.json")
    eng = Engine(10_000.0)
    eng.apply([f for f in _flow().backtest(small_ds, 10_000).fills[:5]])
    save_state(eng, path)
    restored = Engine(10_000.0)
    assert load_state(restored, path)
    assert restored.balances == eng.balances
    assert set(restored.positions) == set(eng.positions)


def test_binance_broker_requires_keys():
    b = sf.BinanceBroker()
    with pytest.raises(ValueError):
        b.execute([sf.Order(pair="BTCUSDT", side=sf.Side.BUY, qty=1.0)], bar=None)


def test_max_bars_caps_run(small_ds):
    feed = sf.ReplayFeed(small_ds)
    run = run_live_loop(_flow(), feed, 10_000.0, sf.SimBroker(), max_bars=10)
    assert run.equity_curve.height <= 11


def test_simulate_matches_backtest(small_ds):
    """Incremental live simulation reproduces the vectorized backtest (no look-ahead)."""
    bt = _flow().backtest(small_ds, capital=10_000)
    sim = _flow().simulate(small_ds, capital=10_000, warmup=0)
    assert sim.final_equity == pytest.approx(bt.final_equity, abs=1e-6)
    assert len(sim.fills) == len(bt.fills)


def test_replayfeed_warmup_split(small_ds):
    """warmup() holds the leading window; stream() yields the rest, no overlap."""
    feed = sf.ReplayFeed(small_ds, warmup_bars=20)
    warm_ts = feed.warmup().frame.get_column("ts").n_unique()
    live_ts = sum(1 for _ in feed.stream())
    assert warm_ts == 20
    assert warm_ts + live_ts == small_ds.frame.get_column("ts").n_unique()


def test_simulate_warmup_window_runs(small_ds):
    run = _flow().simulate(small_ds, capital=10_000, warmup=30)
    assert run.promotable is False
    assert run.equity_curve.height > 0
