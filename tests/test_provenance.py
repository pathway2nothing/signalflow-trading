"""Every tracked run records the code that produced it."""

import re

import pytest

import signalflow as sf
from signalflow.experiment import git_sha, log_config, provenance


def test_provenance_has_the_pinning_fields():
    tags = provenance(seed=7)
    for key in ("signalflow_trading_version", "signalflow_trading_git", "git_sha", "python", "platform", "polars"):
        assert key in tags and isinstance(tags[key], str) and tags[key]
    assert tags["seed"] == "7"
    assert "seed" not in provenance()
    assert re.fullmatch(r"[0-9a-f]{12}(\+dirty)?|unknown", tags["git_sha"])


def test_git_sha_outside_a_repository_is_unknown(tmp_path):
    assert git_sha(tmp_path) == "unknown"


def test_log_config_without_an_active_run_is_a_noop(tmp_path):
    cfg = tmp_path / "experiment.yaml"
    cfg.write_text("kind: experiment\n", encoding="utf-8")
    assert log_config(cfg) is False


def test_experiment_run_tags_provenance_and_config(tmp_path):
    pytest.importorskip("mlflow")
    import mlflow

    cfg = tmp_path / "experiment.yaml"
    cfg.write_text("kind: experiment\nseed: 3\n", encoding="utf-8")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    with sf.experiment_run("prov_exp", params={"a": 1}, tags={"platform": "override"}, tracking_uri=uri, seed=3):
        assert log_config(cfg) is True
        run_id = mlflow.active_run().info.run_id
    mlflow.set_tracking_uri(uri)
    tags = mlflow.get_run(run_id).data.tags
    assert tags["seed"] == "3"
    assert tags["signalflow_trading_version"] == sf.__version__
    assert re.fullmatch(r"[0-9a-f]{12}(\+dirty)?|unknown", tags["git_sha"])
    assert tags["platform"] == "override"
    artifacts = [a.path for a in mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path="config")]
    assert "config/experiment.yaml" in artifacts

    with sf.experiment_run("prov_exp", tracking_uri=uri, record_provenance=False):
        run_id = mlflow.active_run().info.run_id
    assert "git_sha" not in mlflow.get_run(run_id).data.tags
