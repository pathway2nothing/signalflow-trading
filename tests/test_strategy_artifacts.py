"""Strategies with trained artifacts pin them through flow.save(model_dir=) like forecast models."""

from dataclasses import dataclass

import pytest

import signalflow as sf
from signalflow.errors import ArtifactError
from signalflow.strategy.base import Strategy


@sf.register_strategy("_artifact_probe")
@dataclass
class _ArtifactProbe(Strategy):
    """Test-only strategy that must persist a blob before its config is complete."""

    blob: str = "trained"
    uri: str | None = None

    def save_artifacts(self, model_dir):
        if self.uri:
            return
        if not model_dir:
            raise ArtifactError("probe needs model_dir")
        import os

        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "probe.txt")
        with open(path, "w") as fh:
            fh.write(self.blob)
        self.uri = path

    def to_config(self):
        return {"name": self.name, "params": {"uri": self.uri}}

    @classmethod
    def from_config(cls, cfg):
        uri = (cfg.get("params") or {})["uri"]
        with open(uri) as fh:
            return cls(blob=fh.read(), uri=uri)

    def decide(self, obs):
        return []


def _flow():
    return sf.Flow(name="probe", detectors=[sf.SmaCrossDetector()], strategy=_ArtifactProbe(blob="weights-v1"))


def test_save_persists_strategy_artifact_and_load_restores_it(tmp_path):
    flow = _flow()
    path = tmp_path / "flow.yaml"
    flow.save(str(path), model_dir=str(tmp_path / "models"))
    assert flow.strategy.uri and (tmp_path / "models" / "probe.txt").read_text() == "weights-v1"

    loaded = sf.Flow.load(str(path))
    assert isinstance(loaded.strategy, _ArtifactProbe)
    assert loaded.strategy.blob == "weights-v1"


def test_save_without_model_dir_raises_for_unpinned_artifact(tmp_path):
    with pytest.raises(ArtifactError, match="model_dir"):
        _flow().save(str(tmp_path / "flow.yaml"))


def test_save_is_idempotent_once_pinned(tmp_path):
    flow = _flow()
    flow.save(str(tmp_path / "a.yaml"), model_dir=str(tmp_path / "models"))
    first = flow.strategy.uri
    flow.save(str(tmp_path / "b.yaml"))  # no model_dir needed: already pinned
    assert flow.strategy.uri == first
