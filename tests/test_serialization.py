"""Declarative round-trip: to_config <-> build_transform for pipes and detectors."""

import cloudpickle
import polars as pl
import pytest
import yaml

import signalflow as sf
from signalflow._version import __version__
from signalflow.transform.base import build_transform
from signalflow.transform.encode.woe import Binning, WoE


def _ds():
    return sf.data("memory", pairs=["BTCUSDT"], start="2024-01-01", end="2024-01-05", interval="1h")


def _observation():
    from signalflow.engine.types import PortfolioSnapshot, Position
    from signalflow.strategy.observation import Observation

    snap = PortfolioSnapshot(
        ts="2024-01-01T00:00:00",
        target="USDT",
        balances={"USDT": 100.0},
        positions={"BTCUSDT": Position(pair="BTCUSDT", qty=1.0, avg_price=90.0)},
        equity=100.0,
        prices={"BTCUSDT": 100.0},
    )
    signals = pl.DataFrame({"pair": ["BTCUSDT"], "signal": ["rise"]})
    return Observation("2024-01-01T00:00:00", signals, snap, {"note": "x"})


def test_observation_dict_round_trip():
    obs = _observation()
    rebuilt = type(obs).from_dict(obs.to_dict())
    assert rebuilt.schema_version == obs.schema_version
    assert rebuilt.mandate == obs.mandate
    assert "BTCUSDT" in rebuilt.portfolio.positions


def test_observation_schema_mismatch_raises():
    from signalflow.strategy.observation import Observation

    payload = _observation().to_dict()
    payload["schema_version"] = 999
    with pytest.raises(sf.SchemaVersionError):
        Observation.from_dict(payload)


def test_riskviolation_not_exported():
    from signalflow import errors

    assert not hasattr(sf, "RiskViolation")
    assert not hasattr(errors, "RiskViolation")


def test_feature_pipe_load_rejects_non_pipe(tmp_path):
    path = str(tmp_path / "not_a_pipe.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(sf.SMA(10).to_config(), fh)
    with pytest.raises(sf.PipeError):
        sf.FeaturePipe.load(path)


def test_feature_pipe_config_round_trip():
    pipe = sf.FeaturePipe(sf.SMA(10), sf.SMA(20))
    rebuilt = build_transform(pipe.to_config())
    assert isinstance(rebuilt, sf.FeaturePipe)
    assert rebuilt.outputs == pipe.outputs
    ds = _ds()
    assert rebuilt.compute(ds.frame).equals(pipe.compute(ds.frame))


def test_nested_feature_pipe_round_trip():
    pipe = sf.FeaturePipe(sf.FeaturePipe(sf.SMA(5)), sf.SMA(30))
    rebuilt = build_transform(pipe.to_config())
    assert rebuilt.outputs == pipe.outputs == ["sma_5", "sma_30"]
    inner = rebuilt.transforms[0]
    assert isinstance(inner, sf.FeaturePipe)


def test_feature_pipe_save_load(tmp_path):
    pipe = sf.FeaturePipe(sf.SMA(10), sf.SMA(50))
    path = str(tmp_path / "pipe.yaml")
    pipe.save(path)
    loaded = sf.FeaturePipe.load(path)
    assert loaded.outputs == pipe.outputs


def test_detector_config_round_trip():
    det = sf.SmaCrossDetector(fast=5, slow=20)
    rebuilt = build_transform(det.to_config())
    assert rebuilt.to_config() == det.to_config()


def test_flow_yaml_detector_round_trip(tmp_path):
    flow = sf.Flow(name="ser", detectors=[sf.SmaCrossDetector(fast=3, slow=8)])
    path = str(tmp_path / "flow.yaml")
    flow.save(path)
    loaded = sf.Flow.load(path)
    assert [d.to_config() for d in loaded.detectors] == [d.to_config() for d in flow.detectors]


def test_woe_recipe_round_trip():
    woe = WoE(binning=Binning("quantile", 8), smoothing=0.7, columns=["a", "b"])
    rebuilt = build_transform(woe.to_config())
    assert isinstance(rebuilt, WoE)
    assert isinstance(rebuilt.binning, Binning)
    assert rebuilt.binning.method == "quantile" and rebuilt.binning.max_bins == 8
    assert rebuilt.smoothing == 0.7 and rebuilt.columns == ["a", "b"]
    assert rebuilt.to_config() == woe.to_config()


def test_rules_strategy_config_round_trip():
    from signalflow.strategy.base import build_strategy

    strat = sf.RulesStrategy(entry=sf.Entry(size_pct=0.2, min_p_success=0.4), exit=sf.Exit(tp=0.05, sl=0.02))
    rebuilt = build_strategy(strat.to_config())
    assert isinstance(rebuilt, sf.RulesStrategy)
    assert rebuilt.entry.size_pct == 0.2 and rebuilt.entry.min_p_success == 0.4
    assert rebuilt.exit.tp == 0.05 and rebuilt.exit.sl == 0.02


def test_vote_validator_config_threshold_round_trip():
    from signalflow.flow.yaml import _build_validator, _validator_cfg
    from signalflow.model import VoteValidator

    v = VoteValidator([], threshold=0.75)
    assert v.to_config() == {"threshold": 0.75}
    cfg = _validator_cfg(v, model_dir=None)
    assert cfg["params"] == {"threshold": 0.75}
    rebuilt = _build_validator(cfg)
    assert isinstance(rebuilt, VoteValidator) and rebuilt.threshold == 0.75


def test_flow_yaml_stamps_version(tmp_path):
    flow = sf.Flow(name="ver", detectors=[sf.SmaCrossDetector(fast=3, slow=8)])
    path = str(tmp_path / "flow.yaml")
    flow.save(path)
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    assert doc["signalflow_version"] == __version__


def test_flow_yaml_version_mismatch_warns():
    from loguru import logger

    from signalflow.flow.yaml import _warn_version_mismatch

    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        _warn_version_mismatch("0.0")
        _warn_version_mismatch(__version__)
    finally:
        logger.remove(sink)
    assert sum("mismatch" in m for m in msgs) == 1


def test_llm_strategy_config_round_trip():
    pytest.importorskip("pydantic")
    from signalflow.strategy.base import build_strategy
    from signalflow.strategy.llm import LLMStrategy, OpenAICompatClient

    llm = LLMStrategy(
        client=OpenAICompatClient(base_url="http://host/v1", model="m1", api_key="secret"),
        mandate="be long",
    )
    cfg = llm.to_config()
    assert "api_key" not in cfg["params"]["client"]
    rebuilt = build_strategy(cfg)
    assert isinstance(rebuilt, LLMStrategy)
    assert rebuilt.client.base_url == "http://host/v1" and rebuilt.client.model == "m1"
    assert rebuilt.mandate == "be long"
    assert isinstance(rebuilt.fallback, sf.RulesStrategy)


def test_llm_flow_save_load_decide(tmp_path):
    pytest.importorskip("pydantic")
    from signalflow.engine.types import PortfolioSnapshot
    from signalflow.strategy.llm import LLMStrategy, OpenAICompatClient
    from signalflow.strategy.observation import Observation

    llm = LLMStrategy(client=OpenAICompatClient(base_url="http://127.0.0.1:1/v1", model="m1", timeout=1.0), mandate="m")
    flow = sf.Flow(name="llm_ser", detectors=[sf.SmaCrossDetector(fast=3, slow=8)], strategy=llm)
    path = str(tmp_path / "flow.yaml")
    flow.save(path)
    loaded = sf.Flow.load(path)
    assert isinstance(loaded.strategy, LLMStrategy)
    snap = PortfolioSnapshot(ts=0, target="USDT", balances={"USDT": 10_000.0}, positions={}, equity=10_000.0, prices={})
    obs = Observation(0, pl.DataFrame({"pair": [], "signal": []}), snap, {})
    assert isinstance(loaded.strategy.decide(obs), list)


def test_flow_save_unregistered_detector_raises(tmp_path):
    from dataclasses import dataclass

    @dataclass
    class UnregisteredDetector(sf.SignalDetector):
        n: int = 3

        def detect(self, df):
            return df.with_columns(pl.lit(sf.NONE).alias("signal"))

    flow = sf.Flow(name="unreg", detectors=[UnregisteredDetector()])
    with pytest.raises(sf.ArtifactError) as exc:
        flow.save(str(tmp_path / "flow.yaml"))
    assert "UnregisteredDetector" in str(exc.value)


def test_non_dataclass_transform_registration_rejected():
    from signalflow.decorators import detector

    with pytest.raises(TypeError, match="dataclass"):

        @detector("plain_reject_test")
        class PlainReject(sf.SignalDetector):
            def __init__(self, lookback=12):
                self.lookback = lookback

            def detect(self, df):
                return df.with_columns(pl.lit(sf.NONE).alias("signal"))


def test_non_dataclass_to_config_raises():
    class PlainNoReg(sf.SignalDetector):
        def __init__(self, lookback=12):
            self.lookback = lookback

        def detect(self, df):
            return df.with_columns(pl.lit(sf.NONE).alias("signal"))

    with pytest.raises(sf.PipeError):
        PlainNoReg(5).to_config()


def test_woe_fitted_state_survives_pickle():
    n = 300
    df = pl.DataFrame({"f1": [float(i % 17) for i in range(n)], "f2": [float(i % 23) for i in range(n)]})
    target = pl.Series([1.0 if i % 3 == 0 else -1.0 for i in range(n)])
    woe = WoE(binning=Binning("quantile", 5), columns=["f1", "f2"]).fit(df, target)
    back = cloudpickle.loads(cloudpickle.dumps(woe))
    assert back.outputs == woe.outputs
    assert back.compute(df).equals(woe.compute(df))
