"""Persistence layer round-trip tests."""


import os
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore", message="X does not have valid feature names")

pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names"
)

from signalflow.data import data
from signalflow.model import ForecastModel
from signalflow.target import FixedHorizon
from signalflow.transform import SMA, FeaturePipe


@pytest.fixture(scope="module")
def fitted():
    ds = data("memory", pairs=["BTCUSDT"], start="2023-01-01", interval="1h")
    model = ForecastModel(
        backend="lightgbm",
        target=FixedHorizon(bars=12),
        features=FeaturePipe(SMA(20), SMA(10)),
        n_folds=3,
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
