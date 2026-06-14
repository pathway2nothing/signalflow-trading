"""LLM strategy model tests."""

import os
import warnings

import polars as pl
import pytest

warnings.filterwarnings("ignore", message="X does not have valid feature names")
pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names")

import signalflow as sf
from signalflow.engine.engine import Engine
from signalflow.flow.loop import enriched_signals
from signalflow.strategy.llm import Decisions, LLMStrategy, OpenAICompatClient
from signalflow.strategy.observation import Observation


@pytest.fixture(scope="module")
def flow_data():
    data = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", interval="1h")
    model = sf.ForecastModel(
        backend="lightgbm",
        target=sf.FixedHorizon(12),
        features=sf.FeaturePipe(sf.SMA(20), sf.SMA(10)),
        n_folds=3,
    )
    model.fit(data)
    flow = sf.Flow(
        name="llm_test",
        forecasts={"m": model},
        detectors=[sf.ThresholdDetector(forecast="m", p_min=0.5)],
        strategy=sf.RulesStrategy(),
        risk=sf.Risk(),
    )
    return flow, data


def _observation(flow, data, *, want_rise):
    signals = enriched_signals(flow, data)
    engine = Engine(10_000, target=data.quote, quote=data.quote)
    last = None
    for bar in data.iter_bars():
        snap = engine.snapshot(bar.ts, bar.prices)
        sdf = signals.filter(pl.col("ts") == bar.ts) if signals.height else signals
        last = Observation(bar.ts, sdf, snap, {})
        if not want_rise:
            return last
        if "signal" in sdf.columns and (sdf.get_column("signal") == "rise").sum() > 0:
            return last
    return last


class FakeClient:
    """Canned client: open the first RISE pair it sees, else hold."""

    def __init__(self):
        self.calls = 0

    def decide(self, context, schema):
        self.calls += 1
        decisions = [
            {"pair": row["pair"], "action": "open", "size_pct": 0.1}
            for row in context.get("signals", [])
            if row.get("signal") == "rise"
        ]
        return {"decisions": decisions}


def test_llm_decide_returns_intents(flow_data):
    obs = _observation(*flow_data, want_rise=True)
    llm = LLMStrategy(client=FakeClient(), mandate="long-only momentum, max 1 position")
    intents = llm.decide(obs)
    assert all(isinstance(i, sf.Intent) for i in intents)
    assert intents and intents[0].kind == sf.IntentKind.OPEN


def test_llm_caches_per_bar(flow_data):
    obs = _observation(*flow_data, want_rise=False)
    client = FakeClient()
    llm = LLMStrategy(client=client, mandate="m")
    llm.decide(obs)
    llm.decide(obs)
    assert client.calls == 1


def test_llm_fallback_on_client_failure(flow_data):
    obs = _observation(*flow_data, want_rise=False)

    class BoomClient:
        def decide(self, context, schema):
            raise RuntimeError("boom")

    llm = LLMStrategy(client=BoomClient(), mandate="m", fallback=sf.RulesStrategy())
    assert isinstance(llm.decide(obs), list)


def test_llm_backtest_runs(flow_data):
    flow, data = flow_data
    llm = LLMStrategy(client=FakeClient(), mandate="long-only momentum")
    run = flow.replace(strategy=llm).backtest(data, capital=10_000)
    assert run.equity_curve.height > 0


@pytest.mark.skipif(not os.environ.get("SIGNALFLOW_LLM_BASE_URL"), reason="no SIGNALFLOW_LLM_BASE_URL set")
def test_openai_compat_client_live(flow_data):
    obs = _observation(*flow_data, want_rise=True)
    client = OpenAICompatClient()
    llm = LLMStrategy(client=client, mandate="long-only; open at most one position on the strongest RISE")
    assert isinstance(llm.decide(obs), list)
    raw = client.decide({**obs.to_prompt_context(), "mandate": llm.mandate}, Decisions.model_json_schema())
    assert raw is None or "decisions" in raw
