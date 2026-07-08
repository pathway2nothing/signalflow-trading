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
    assert len(restored.event_log) == len(eng.event_log)
    for k, pos in eng.positions.items():
        assert restored.positions[k].opened_ts == pos.opened_ts


def test_armed_live_halts_on_tripped_kill_switch(small_ds, tmp_path):
    ks = tmp_path / "kill.txt"
    ks.write_text("tripped")
    flow = sf.Flow(
        name="armed",
        detectors=[sf.SmaCrossDetector(fast=3, slow=8)],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.5), exit=sf.Exit(tp=0.01, sl=0.01)),
        risk=sf.Risk(kill_switch_path=str(ks)),
    )
    with pytest.raises(sf.KillSwitchTripped):
        run_live_loop(flow, sf.ReplayFeed(small_ds), 10_000.0, sf.ExchangeBroker(), max_bars=5)


def test_backtest_kill_switch_stays_silent(small_ds, tmp_path):
    ks = tmp_path / "kill.txt"
    ks.write_text("tripped")
    flow = sf.Flow(
        name="silent",
        detectors=[sf.SmaCrossDetector(fast=3, slow=8)],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.5), exit=sf.Exit(tp=0.01, sl=0.01)),
        risk=sf.Risk(kill_switch_path=str(ks)),
    )
    run = flow.backtest(small_ds, capital=10_000)
    assert all(f.side != sf.Side.BUY for f in run.fills)


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


def _warmup60_flow():
    return sf.Flow(
        name="warm",
        detectors=[sf.SmaCrossDetector(fast=3, slow=59)],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.5), exit=sf.Exit(tp=0.01, sl=0.01)),
    )


def test_simulate_default_warmup_consumes_declared(small_ds):
    flow = _warmup60_flow()
    assert flow.required_warmup == 60

    default = flow.simulate(small_ds, capital=10_000)
    explicit = flow.simulate(small_ds, capital=10_000, warmup=60)

    assert default.final_equity == pytest.approx(explicit.final_equity, abs=1e-9)
    assert len(default.fills) == len(explicit.fills)

    first_60 = small_ds.frame.get_column("ts").unique().sort().to_list()[:60]
    assert all(f.ts not in set(first_60) for f in default.fills)


def test_required_warmup_combines_feature_model_and_detector(small_ds):
    model = sf.ForecastModel(
        target=sf.FixedHorizon(bars=12),
        features=sf.FeaturePipe(sf.SMA(40)),
        output="p_rise",
        n_folds=3,
    ).fit(small_ds)
    flow = sf.Flow(
        name="combo",
        forecasts={"m": model},
        detectors=[sf.SmaCrossDetector(fast=3, slow=59)],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.5)),
    )
    assert flow.required_warmup == 60


def test_polling_feed_warmup_resolves_from_flow():
    from signalflow.flow.live import resolve_warmup_bars

    feed = sf.PollingFeed(source=sf.MemorySource(), pairs=["BTCUSDT"], interval="1h")
    assert feed.warmup_bars is None

    flow = _warmup60_flow()
    required = resolve_warmup_bars(flow, feed)
    assert required == 60
    assert feed.warmup_bars == 60
