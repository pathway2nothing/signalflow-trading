"""Persistence layer round-trip tests."""

import os
import warnings

import numpy as np
import pytest

import signalflow as sf
from signalflow.data import dataset
from signalflow.model import ForecastModel
from signalflow.model.cv import KFold
from signalflow.target import FixedHorizon
from signalflow.transform import SMA, FeaturePipeline

warnings.filterwarnings("ignore", message="X does not have valid feature names")

pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names")


@pytest.fixture(scope="module")
def fitted():
    ds = dataset("synthetic", pairs=["BTCUSDT"], start="2023-01-01", interval="1h")
    model = ForecastModel(
        backend="lightgbm",
        target=FixedHorizon(bars=12),
        features=FeaturePipeline(SMA(20), SMA(10)),
        cv=KFold(3),
    )
    model.fit(ds)
    return model, ds


def _assert_round_trip(loaded, model, ds):
    assert loaded.is_fitted
    assert loaded.fingerprint == model.fingerprint
    got = loaded.predict(ds)[model.output].to_numpy()
    want = model.predict(ds)[model.output].to_numpy()
    assert np.allclose(got, want, equal_nan=True)
    assert loaded.oos_.shape == model.oos_.shape


def test_file_round_trip(fitted, tmp_path):
    model, ds = fitted
    uri = (tmp_path / "model_dir").as_posix()
    returned = model.save(uri)
    assert returned.startswith("file://")
    loaded = ForecastModel.load(uri)
    _assert_round_trip(loaded, model, ds)


def test_mlflow_round_trip(fitted, tmp_path, monkeypatch):
    import mlflow

    tracking = (tmp_path / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)

    model, ds = fitted
    uri = model.save("mlflow://models/sf_test_model")
    assert uri.startswith("mlflow://")
    loaded = ForecastModel.load(uri)
    _assert_round_trip(loaded, model, ds)


def test_mlflow_save_inside_active_run(fitted, tmp_path):
    import mlflow

    tracking = (tmp_path / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking)

    model, ds = fitted
    mlflow.set_experiment("caller_experiment")
    caller_exp = mlflow.get_experiment_by_name("caller_experiment")

    with mlflow.start_run() as parent:
        parent_id = parent.info.run_id
        uri = model.save("mlflow://models/sf_nested_model")

        active = mlflow.active_run()
        assert active is not None
        assert active.info.run_id == parent_id
        assert active.info.experiment_id == caller_exp.experiment_id

        children = mlflow.search_runs(
            experiment_ids=[caller_exp.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
        )
        assert len(children) >= 1

    assert uri.startswith("mlflow://")
    loaded = ForecastModel.load(uri)
    _assert_round_trip(loaded, model, ds)


def test_mlflow_uri_pins_version_on_registration(fitted, tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from types import SimpleNamespace

    import mlflow

    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri((tmp_path / "mlruns").resolve().as_uri())
    monkeypatch.setattr(mlflow, "register_model", lambda uri, name: SimpleNamespace(version="7"))

    model, _ = fitted
    uri = model.save("mlflow://models/d17_pin_test")
    assert uri == "mlflow://models/d17_pin_test@7"


def test_mlflow_uri_unversioned_on_registration_failure(fitted, tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    import mlflow

    def _boom(uri, name):
        raise RuntimeError("registry unavailable")

    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri((tmp_path / "mlruns").resolve().as_uri())
    monkeypatch.setattr(mlflow, "register_model", _boom)

    model, _ = fitted
    uri = model.save("mlflow://models/d17_unver_test")
    assert uri == "mlflow://models/d17_unver_test"


def test_mlflow_load_warns_without_version(fitted, tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    import mlflow
    from loguru import logger

    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri((tmp_path / "mlruns").resolve().as_uri())
    monkeypatch.setattr(mlflow, "register_model", lambda uri, name: None)

    model, _ = fitted
    uri = model.save("mlflow://models/d17_warn_test")
    assert "@" not in uri

    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        ForecastModel.load(uri)
    finally:
        logger.remove(sink)
    assert any("no @version" in m for m in msgs)


@pytest.mark.skipif(
    not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    reason="no HF_TOKEN set",
)
def test_hf_round_trip(fitted):
    from huggingface_hub import whoami

    model, ds = fitted
    user = whoami(token=os.environ.get("HF_TOKEN"))["name"]
    repo_id = f"{user}/sf-persistence-test"
    model.save(f"hf://{repo_id}")
    loaded = ForecastModel.load(f"hf://{repo_id}")
    _assert_round_trip(loaded, model, ds)


def test_artifact_records_environment_and_warns_on_mismatch(tmp_path, ds, fitted_forecast):
    import json

    from signalflow.model.store._layout import ENV_JSON, check_environment

    uri = fitted_forecast.save(f"file://{tmp_path / 'env_model'}")
    env = json.loads((tmp_path / "env_model" / ENV_JSON).read_text())
    assert env["signalflow"] == sf.__version__ and "polars" in env and "lightgbm" in env
    assert check_environment(env) == []
    assert check_environment({**env, "polars": "0.1.0"}) == [f"polars 0.1.0 (artifact) vs {env['polars']} (now)"]
    assert sf.ForecastModel.load(uri).predict(ds).equals(fitted_forecast.predict(ds))


def test_hub_artifacts_need_trust_remote():
    from signalflow.model.store import load_model

    with pytest.raises(sf.ArtifactError, match="trust_remote"):
        load_model("hf://someone/some-model")
    with pytest.raises(sf.ArtifactError, match="trust_remote"):
        sf.ForecastModel.load("hf://someone/some-model")
