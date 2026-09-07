"""The kill switch works while a loop runs, survives restarts, and Flow.live passes the loop knobs through."""

import polars as pl
import pytest

import signalflow as sf
from signalflow.flow.live import load_state, save_state


class _Dense(sf.SignalDetector):
    """RISE whenever close is above its 5-bar mean - fires on most bars."""

    @property
    def warmup(self) -> int:
        return 5

    def detect(self, df):
        sma = pl.col("close").rolling_mean(5).over("pair")
        return df.with_columns(
            pl.when(pl.col("close") > sma).then(pl.lit(sf.RISE)).otherwise(pl.lit(sf.NONE)).alias("signal")
        )


def _flow(risk: sf.Risk | None = None) -> sf.Flow:
    return sf.Flow(
        name="ks",
        detectors=[_Dense()],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.2), exit=sf.Exit(tp=0.002, sl=0.002)),
        risk=risk or sf.Risk(),
    )


@pytest.fixture(scope="module")
def ds():
    return sf.dataset("synthetic", pairs=["BTCUSDT"], start="2024-01-01", end="2024-01-08", interval="1h")


def test_kill_switch_file_is_reread_on_every_check(tmp_path):
    ks = tmp_path / "kill"
    risk = sf.Risk(kill_switch_path=str(ks))
    assert risk.tripped is False
    ks.write_text("operator")
    assert risk.tripped is True
    ks.unlink()
    assert risk.tripped is False
    risk.trip("drawdown")
    assert ks.exists() and risk.tripped and risk.reason == "drawdown"
    risk.reset()
    assert not ks.exists() and risk.tripped is False


def test_creating_and_removing_the_file_gates_entries_mid_run(ds, tmp_path):
    ks = tmp_path / "kill"
    flow = _flow(sf.Risk(kill_switch_path=str(ks)))
    ts = ds.frame.get_column("ts").unique().sort().to_list()
    engage_at, release_at = ts[40], ts[90]

    def on_bar(engine, bar, fills, latency):
        if bar.ts == engage_at:
            ks.write_text("stop")
        if bar.ts == release_at:
            ks.unlink()

    run = flow.simulate(ds, capital=10_000, on_bar=on_bar)
    buys = [f.ts for f in run.fills if f.side == sf.Side.BUY]
    assert any(t < engage_at for t in buys), "entries happened before the switch"
    assert not [t for t in buys if engage_at < t <= release_at], "no entries while the file existed"
    assert any(t > release_at for t in buys), "entries resumed after the file was removed"


def test_risk_state_persists_and_restores(tmp_path):
    engine = sf.Engine(10_000, target="USDT")
    risk = sf.Risk()
    risk.trip("drawdown 0.3 >= 0.25")
    path = str(tmp_path / "book.json")
    save_state(engine, path, peak=10_000.0, risk=risk)

    fresh = sf.Risk()
    state = load_state(sf.Engine(10_000, target="USDT"), path)
    assert state["risk"] == {"tripped": True, "reason": "drawdown 0.3 >= 0.25"}
    fresh.restore(state["risk"])
    assert fresh.tripped and fresh.reason == "drawdown 0.3 >= 0.25"


def test_restart_resumes_tripped(ds, tmp_path):
    path = str(tmp_path / "book.json")
    first = _flow(sf.Risk(max_drawdown=0.0))  # trips on the first bar
    run1 = first.simulate(ds, capital=10_000, state_path=path)
    assert first.risk.tripped and not [f for f in run1.fills if f.side == sf.Side.BUY]

    second = _flow(sf.Risk())  # no limit of its own, but resumes the saved trip
    run2 = second.simulate(ds, capital=10_000, state_path=path)
    assert second.risk.tripped
    assert not [f for f in run2.fills if f.side == sf.Side.BUY]


def test_flow_live_passes_loop_knobs_through(ds):
    seen: dict = {"bars": 0, "mandates": set()}

    class _Spy(sf.RulesStrategy):
        def decide(self, obs):
            seen["mandates"].add(tuple(sorted(obs.mandate.items())))
            return super().decide(obs)

    flow = sf.Flow(name="spy", detectors=[_Dense()], strategy=_Spy())

    def on_bar(engine, bar, fills, latency):
        seen["bars"] += 1

    run = flow.live(
        ds, capital=10_000, mandate={"max_notional": 1_000}, on_bar=on_bar, late_bar_policy="skip", max_latency_s=1.0
    )
    assert run.equity_curve.height > 0
    assert seen["bars"] == ds.frame.get_column("ts").n_unique()
    assert (("max_notional", 1_000),) in seen["mandates"]
    with pytest.raises(ValueError):
        flow.live(ds, capital=10_000, late_bar_policy="bogus")
