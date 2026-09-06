"""FeaturePipeline.from_names: registry-name -> validated pipeline tests."""

import polars as pl
import pytest
from loguru import logger

from signalflow.data import dataset
from signalflow.decorators import feature
from signalflow.errors import PipelineError, UnknownComponentError
from signalflow.transform import FeaturePipeline
from signalflow.transform.base import Feature


@feature("boom_feature")
class _BoomFeature(Feature):
    """Deliberately-broken feature that raises on compute (for drop/raise tests)."""

    @property
    def outputs(self) -> list[str]:
        return ["boom"]

    def exprs(self) -> list[pl.Expr]:
        raise RuntimeError("boom feature cannot compute")


@pytest.fixture(scope="module")
def sample():
    return dataset("synthetic", pairs=["BTCUSDT"], start="2023-01-01", interval="1h")


def test_builds_from_sma():
    pipe = FeaturePipeline.from_names(["sma"])
    assert isinstance(pipe, FeaturePipeline)
    assert "sma_20" in pipe.outputs


def test_unknown_name_raises():
    with pytest.raises(UnknownComponentError):
        FeaturePipeline.from_names(["definitely_not_a_feature"])


def test_broken_transform_raises_by_default(sample):
    with pytest.raises(PipelineError):
        FeaturePipeline.from_names(["sma", "boom_feature"], data=sample)


def test_broken_transform_dropped_with_warning(sample):
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        pipe = FeaturePipeline.from_names(["sma", "boom_feature"], data=sample, on_error="drop")
    finally:
        logger.remove(sink_id)
    assert "sma_20" in pipe.outputs
    assert "boom" not in pipe.outputs
    assert any("boom_feature" in m for m in messages)


def test_pipeline_split_and_outputs_with_stateful_tail():
    import signalflow as sf

    pipe = sf.FeaturePipeline(sf.SMA(10), sf.SMA(20), sf.WoE(), sf.IVSelector(min_iv=0.0))
    prefix, tail = pipe.split()
    assert prefix.outputs == ["sma_10", "sma_20"]
    assert [t.name for t in tail.transforms] == ["woe", "iv_selector"]
    assert pipe.outputs == ["sma_10", "sma_20"]  # tail outputs unknown before fit
    assert not pipe.is_fitted and prefix.is_fitted

    ds = sf.dataset("synthetic", pairs=["BTCUSDT"], start="2024-01-01", end="2024-02-01", interval="1h")
    frame = prefix.compute(ds.frame).drop_nulls(subset=["sma_10", "sma_20"])
    y = (frame.get_column("close").shift(-1) > frame.get_column("close")).cast(pl.Int64).fill_null(0)
    fitted = tail.clone().fit(frame, y)
    assert fitted.is_fitted
    assert fitted.outputs == ["sma_10__woe", "sma_20__woe"]  # WoE replaced its inputs; selector kept both
    out = fitted.compute(frame)
    assert "sma_10" not in out.columns and "sma_10__woe" in out.columns


def test_model_with_encoder_in_pipeline_round_trips(tmp_path):
    import signalflow as sf

    ds = sf.dataset("synthetic", pairs=["BTCUSDT"], start="2024-01-01", end="2024-03-01", interval="1h")
    model = sf.ForecastModel(
        target=sf.FixedHorizon(bars=6),
        features=sf.FeaturePipeline(sf.SMA(5), sf.SMA(10), sf.WoE(), sf.IVSelector()),
        cv=sf.KFold(3),
    ).fit(ds)
    assert model.tail is not None and model.tail_.is_fitted
    assert model.model_._sf_cols and all(c.endswith("__woe") for c in model.model_._sf_cols)
    assert model.woe_history(), "each fold records its WoE state"
    assert model.feature_signature["model_columns"] == model.model_._sf_cols
    cfg = model.features.to_config()
    assert [t["transform"] for t in cfg["params"]["transforms"]] == ["sma", "sma", "woe", "iv_selector"]
    rebuilt = sf.FeaturePipeline.from_config(cfg)
    assert [t.name for t in rebuilt.transforms] == [t.name for t in model.features.transforms]
    uri = model.save(f"file://{tmp_path / 'm'}")
    same = sf.ForecastModel.load(uri)
    assert same.predict(ds).equals(model.predict(ds))
