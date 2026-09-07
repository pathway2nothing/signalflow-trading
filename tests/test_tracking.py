"""Pluggable trackers: registration, fan-out, provenance tags, config artifacts, optional backends."""

import pytest

import signalflow as sf
from signalflow.experiment import active_tracker, available_trackers, get_tracker, log_config
from signalflow.experiment.spec import tracking_target
from signalflow.experiment.tracking import MultiTracker, NullTracker


@sf.register_tracker("memory")
class _MemoryTracker(sf.BaseTracker):
    def __init__(self, label: str = "") -> None:
        self.label = label
        self.events: list = []

    def start(self, experiment, run_name=None):
        self.events.append(("start", experiment, run_name))

    def log_params(self, params):
        self.events.append(("params", dict(params)))

    def set_tags(self, tags):
        self.events.append(("tags", dict(tags)))

    def log_metrics(self, metrics, step=None):
        self.events.append(("metrics", dict(metrics), step))

    def log_artifact(self, path, artifact_path=None):
        self.events.append(("artifact", path, artifact_path))

    def end(self):
        self.events.append(("end",))


def test_registered_tracker_receives_the_whole_run(tmp_path):
    cfg = tmp_path / "experiment.yaml"
    cfg.write_text("kind: experiment\n", encoding="utf-8")
    assert "memory" in available_trackers()
    with sf.experiment_run("exp", params={"a": 1}, tags={"who": "me"}, seed=5, tracker="memory", label="x") as t:
        assert isinstance(t, _MemoryTracker) and t.label == "x"
        assert active_tracker() is t
        t.log_metrics({"auc": 0.7}, step=2)
        assert log_config(cfg) is True
    assert active_tracker() is None
    kinds = [e[0] for e in t.events]
    assert kinds == ["start", "params", "tags", "metrics", "artifact", "end"]
    assert t.events[0] == ("start", "exp", None)
    tags = t.events[2][1]
    assert tags["who"] == "me" and tags["seed"] == "5" and "git_sha" in tags
    assert t.events[4] == ("artifact", str(cfg), "config")


def test_instance_and_list_fan_out():
    a, b = _MemoryTracker("a"), _MemoryTracker("b")
    with sf.experiment_run("exp", tracker=[a, b], record_provenance=False) as t:
        assert isinstance(t, MultiTracker)
        t.log_metrics({"x": 1.0})
    assert [e[0] for e in a.events] == ["start", "metrics", "end"]
    assert [e[0] for e in b.events] == ["start", "metrics", "end"]


def test_null_and_unknown_trackers():
    with sf.experiment_run("exp", tracker=None) as t:
        assert isinstance(t, NullTracker)
    with pytest.raises(sf.UnknownComponentError):
        get_tracker("no-such-tracker")


def test_missing_backend_disables_tracking_with_a_warning(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no wandb here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with sf.experiment_run("exp", tracker="wandb") as t:
        assert t is None
    assert log_config("pyproject.toml") is False


def test_spec_tracking_block_forms():
    assert tracking_target({"mlflow": "exp1"}) == ("mlflow", "exp1", {})
    assert tracking_target({"tracker": "wandb", "experiment": "exp2", "options": {"mode": "offline"}}) == (
        "wandb",
        "exp2",
        {"mode": "offline"},
    )
    assert tracking_target({"tracker": ["mlflow", "litlogger"], "experiment": "exp3"}) == (
        ["mlflow", "litlogger"],
        "exp3",
        {},
    )
    assert tracking_target({}) == (None, None, {})


def test_wandb_offline_run_if_installed(tmp_path):
    wandb = pytest.importorskip("wandb")
    with sf.experiment_run(
        "sf-test", params={"a": 1}, tracker="wandb", mode="offline", dir=str(tmp_path), record_provenance=False
    ) as t:
        assert t is not None and t._run is not None
        t.log_metrics({"auc": 0.5}, step=1)
    assert wandb.run is None


def test_litlogger_run_if_installed():
    pytest.importorskip("litlogger")
    with sf.experiment_run("sf-test", params={"a": 1}, tracker="litlogger", record_provenance=False) as t:
        assert t is not None
        t.log_metrics({"auc": 0.5}, step=1)
