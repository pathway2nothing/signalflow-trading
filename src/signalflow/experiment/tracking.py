"""Experiment tracking - pluggable run loggers.

A :class:`Tracker` is the small surface a run needs: start, params, tags,
metrics, artifacts, end. MLflow, Weights & Biases and Lightning AI's LitLogger
ship as adapters that import their package only when used, so none is a required
dependency;
any package can add its own through the ``signalflow.trackers`` entry point or
:func:`register_tracker`. :func:`experiment_run` opens a run on the chosen
tracker(s), tags it with provenance, and yields the tracker.
"""

import contextlib
import contextvars
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from signalflow.errors import UnknownComponentError
from signalflow.experiment.provenance import provenance

_ENTRY_POINT_GROUP = "signalflow.trackers"
_TRACKERS: dict[str, type] = {}
_ACTIVE: contextvars.ContextVar["Tracker | None"] = contextvars.ContextVar("sf_active_tracker", default=None)


@runtime_checkable
class Tracker(Protocol):
    """What a run logger must offer. Every method may be a no-op."""

    def start(self, experiment: str, run_name: "str | None" = None) -> None: ...

    def log_params(self, params: dict) -> None: ...

    def set_tags(self, tags: dict) -> None: ...

    def log_metrics(self, metrics: dict, step: "int | None" = None) -> None: ...

    def log_artifact(self, path: str, artifact_path: "str | None" = None) -> None: ...

    def end(self) -> None: ...


class BaseTracker:
    """No-op implementation to subclass; override what the backend supports."""

    name: str = "base"

    def start(self, experiment: str, run_name: "str | None" = None) -> None:
        return None

    def log_params(self, params: dict) -> None:
        return None

    def set_tags(self, tags: dict) -> None:
        return None

    def log_metrics(self, metrics: dict, step: "int | None" = None) -> None:
        return None

    def log_artifact(self, path: str, artifact_path: "str | None" = None) -> None:
        return None

    def end(self) -> None:
        return None


def register_tracker(name: str):
    """Class decorator: make a tracker available as ``experiment_run(tracker=name)``."""

    def deco(cls: type) -> type:
        cls.name = name
        _TRACKERS[name] = cls
        return cls

    return deco


def _entry_point_trackers() -> dict[str, Any]:
    return {ep.name: ep for ep in entry_points(group=_ENTRY_POINT_GROUP)}


def available_trackers() -> list[str]:
    """Built-in and entry-point tracker names."""
    return sorted({*_TRACKERS, *_entry_point_trackers()})


def get_tracker(name: str, **options: Any) -> Tracker:
    """Instantiate a registered tracker by name (``options`` go to its constructor)."""
    cls = _TRACKERS.get(name)
    if cls is None:
        ep = _entry_point_trackers().get(name)
        if ep is None:
            raise UnknownComponentError(f"unknown tracker {name!r}; available: {available_trackers()}")
        cls = ep.load()
        _TRACKERS[name] = cls
    return cls(**options)


@register_tracker("null")
class NullTracker(BaseTracker):
    """Logs nothing; the default when no backend is wanted."""


@register_tracker("mlflow")
class MLflowTracker(BaseTracker):
    """MLflow run (``pip install mlflow``, part of the ``[live]`` extra)."""

    def __init__(self, tracking_uri: "str | None" = None) -> None:
        self.tracking_uri = tracking_uri
        self.mlflow: Any = None
        self.run_id: str | None = None

    def start(self, experiment: str, run_name: "str | None" = None) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError("mlflow is not installed (pip install signalflow-trading[live])") from exc
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_name=run_name)
        self.mlflow, self.run_id = mlflow, run.info.run_id

    def log_params(self, params: dict) -> None:
        self.mlflow.log_params(params)

    def set_tags(self, tags: dict) -> None:
        self.mlflow.set_tags(tags)

    def log_metrics(self, metrics: dict, step: "int | None" = None) -> None:
        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, artifact_path: "str | None" = None) -> None:
        self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def end(self) -> None:
        if self.mlflow is not None:
            self.mlflow.end_run()


@register_tracker("litlogger")
class LitLoggerTracker(BaseTracker):
    """Lightning AI LitLogger (``pip install litlogger``); constructor ``options`` go to ``LitLogger(...)``.

    A best-effort adapter over the LitLogger surface: hyperparameters for params
    and tags, ``log_metrics`` for metrics, ``log_artifact``/``log_file`` for files
    when the installed version offers them, ``finalize`` at the end. Missing calls
    are reported once and skipped rather than failing the run.
    """

    def __init__(self, **options: Any) -> None:
        self.options = options
        self._logger: Any = None
        self._warned: set[str] = set()

    def start(self, experiment: str, run_name: "str | None" = None) -> None:
        try:
            import litlogger
        except ImportError as exc:
            raise ImportError("litlogger is not installed (pip install litlogger)") from exc
        cls = getattr(litlogger, "LitLogger", None)
        if cls is None:
            raise ImportError("litlogger is installed but exposes no LitLogger class")
        self._logger = cls(name=run_name or experiment, **self.options)

    def _call(self, names: "tuple[str, ...]", *args: Any, **kwargs: Any) -> bool:
        for name in names:
            fn = getattr(self._logger, name, None)
            if callable(fn):
                try:
                    fn(*args, **kwargs)
                except TypeError:
                    fn(*args)
                return True
        if names[0] not in self._warned:
            self._warned.add(names[0])
            logger.warning(f"LitLoggerTracker: the installed litlogger has none of {names}; skipping")
        return False

    def log_params(self, params: dict) -> None:
        self._call(("log_hyperparams", "log_params"), params)

    def set_tags(self, tags: dict) -> None:
        self._call(("log_hyperparams", "log_params"), {f"tag.{k}": v for k, v in tags.items()})

    def log_metrics(self, metrics: dict, step: "int | None" = None) -> None:
        if not self._call(("log_metrics",), metrics, step=step):
            for key, value in metrics.items():
                self._call(("log",), key, value, step=step)

    def log_artifact(self, path: str, artifact_path: "str | None" = None) -> None:
        self._call(("log_artifact", "log_file"), str(path))

    def end(self) -> None:
        if self._logger is not None:
            self._call(("finalize", "finish", "close"))


@register_tracker("wandb")
class WandbTracker(BaseTracker):
    """Weights & Biases run (``pip install wandb``); constructor ``options`` go to ``wandb.init(...)``.

    The experiment name becomes the W&B ``project`` unless ``options`` set one;
    params go to ``config``, tags to ``config`` under ``tag.<name>`` (W&B tags are
    bare strings), metrics to ``run.log``, files to a logged artifact typed by
    ``artifact_path``.
    """

    def __init__(self, **options: Any) -> None:
        self.options = options
        self._wandb: Any = None
        self._run: Any = None

    def start(self, experiment: str, run_name: "str | None" = None) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb is not installed (pip install wandb)") from exc
        init = {"project": experiment, "name": run_name, **self.options}
        self._wandb = wandb
        self._run = wandb.init(**init)

    def log_params(self, params: dict) -> None:
        self._run.config.update(params, allow_val_change=True)

    def set_tags(self, tags: dict) -> None:
        self._run.config.update({f"tag.{k}": v for k, v in tags.items()}, allow_val_change=True)

    def log_metrics(self, metrics: dict, step: "int | None" = None) -> None:
        self._run.log(dict(metrics), step=step)

    def log_artifact(self, path: str, artifact_path: "str | None" = None) -> None:
        artifact = self._wandb.Artifact(name=Path(path).stem, type=artifact_path or "artifact")
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def end(self) -> None:
        if self._run is not None:
            self._run.finish()


class MultiTracker(BaseTracker):
    """Fan every call out to several trackers."""

    name = "multi"

    def __init__(self, trackers: "list[Tracker]") -> None:
        self.trackers = list(trackers)

    def start(self, experiment: str, run_name: "str | None" = None) -> None:
        for t in self.trackers:
            t.start(experiment, run_name)

    def log_params(self, params: dict) -> None:
        for t in self.trackers:
            t.log_params(params)

    def set_tags(self, tags: dict) -> None:
        for t in self.trackers:
            t.set_tags(tags)

    def log_metrics(self, metrics: dict, step: "int | None" = None) -> None:
        for t in self.trackers:
            t.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, artifact_path: "str | None" = None) -> None:
        for t in self.trackers:
            t.log_artifact(path, artifact_path)

    def end(self) -> None:
        for t in self.trackers:
            t.end()


def resolve_tracker(spec: Any, **options: Any) -> Tracker:
    """A tracker from a name, an instance, a list of either, or ``None`` (-> ``NullTracker``)."""
    if spec is None:
        return NullTracker()
    if isinstance(spec, str):
        return get_tracker(spec, **options)
    if isinstance(spec, (list, tuple)):
        return MultiTracker([resolve_tracker(s, **options) for s in spec])
    return spec


def active_tracker() -> "Tracker | None":
    """The tracker of the enclosing :func:`experiment_run`, if any."""
    return _ACTIVE.get()


@contextlib.contextmanager
def experiment_run(
    name: str,
    params: "dict | None" = None,
    tags: "dict | None" = None,
    tracking_uri: "str | None" = None,
    seed: "int | None" = None,
    run_name: "str | None" = None,
    record_provenance: bool = True,
    tracker: Any = "mlflow",
    **options: Any,
):
    """Open a run on ``tracker`` (a name, an instance, or a list), log ``params``/``tags``, yield the tracker.

    With ``record_provenance`` (default) the run is tagged with what produced it:
    signalflow / ta / labs versions and editable-checkout commits, the working
    directory's commit (``+dirty`` when modified), python, platform, polars and
    ``seed`` - see :func:`signalflow.experiment.provenance.provenance`. Explicit
    ``tags`` override the automatic ones. ``options`` (and ``tracking_uri`` for
    MLflow) go to the tracker's constructor. A backend whose package is not
    installed disables tracking with a warning and yields ``None``.
    """
    if tracking_uri is not None:
        options["tracking_uri"] = tracking_uri
    try:
        handle = resolve_tracker(tracker, **options)
        handle.start(name, run_name)
    except ImportError as exc:
        logger.warning(f"experiment_run: {exc}; tracking disabled")
        yield None
        return
    token = _ACTIVE.set(handle)
    try:
        if params:
            handle.log_params(params)
        all_tags = {**(provenance(seed=seed) if record_provenance else {}), **(tags or {})}
        if all_tags:
            handle.set_tags(all_tags)
        yield handle
    finally:
        _ACTIVE.reset(token)
        handle.end()


def log_config(path: "str | Path", artifact_path: str = "config") -> bool:
    """Attach a config file (the run's full knob set) to the active run; ``False`` when there is none."""
    handle = active_tracker()
    if handle is None:
        return False
    handle.log_artifact(str(path), artifact_path)
    return True


__all__ = [
    "BaseTracker",
    "LitLoggerTracker",
    "MLflowTracker",
    "MultiTracker",
    "NullTracker",
    "Tracker",
    "WandbTracker",
    "active_tracker",
    "available_trackers",
    "experiment_run",
    "get_tracker",
    "log_config",
    "register_tracker",
    "resolve_tracker",
]
