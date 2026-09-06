"""Flow serialization - deployment is data, not code."""

from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger

from signalflow._version import __version__
from signalflow.errors import ArtifactError, UnknownComponentError
from signalflow.strategy.base import build_strategy
from signalflow.strategy.risk import Risk
from signalflow.transform.base import build_transform

if TYPE_CHECKING:
    from signalflow.flow.flow import Flow


def _model_uri(model: Any, slot: str, model_dir: str | None) -> str:
    uri = getattr(model, "_uri", None)
    if uri:
        return str(uri)
    if model_dir:
        return str(model.save(f"file://{model_dir}/{slot}"))
    raise ArtifactError(f"forecast {slot!r} has no pinned URI; save it first or pass model_dir=")


def _strategy_cfg(strategy: Any, model_dir: str | None = None, *, persist: bool = False) -> dict:
    """Declarative strategy config; with ``persist`` first let the strategy pin its artifacts.

    A strategy carrying a trained artifact (an RL policy, say) exposes
    ``save_artifacts(model_dir)``, mirroring how forecast models are pinned to a
    URI: it writes under ``model_dir`` and records the URI its ``to_config`` emits.
    """
    if persist:
        save_artifacts = getattr(strategy, "save_artifacts", None)
        if callable(save_artifacts):
            save_artifacts(model_dir)
    to_config = getattr(strategy, "to_config", None)
    if callable(to_config):
        return dict(to_config())
    return {"name": getattr(strategy, "_sf_name", type(strategy).__name__), "params": {}}


def _build_strategy(cfg: dict) -> Any:
    return build_strategy(cfg)


def _validator_cfg(validator: Any, model_dir: str | None) -> dict | None:
    if validator is None:
        return None
    if hasattr(validator, "children"):
        params = validator.to_config() if hasattr(validator, "to_config") else {}
        return {
            "combinator": type(validator).__name__,
            "children": [_model_uri(c, f"val{i}", model_dir) for i, c in enumerate(validator.children)],
            "params": params,
        }
    return {"uri": _model_uri(validator, "validator", model_dir)}


def _build_validator(cfg: dict | None, trust_remote: bool = False) -> Any:
    if cfg is None:
        return None
    from signalflow.model import ForecastModel, MaxValidator, MeanValidator, VoteValidator

    if "uri" in cfg:
        return ForecastModel.load(cfg["uri"], trust_remote=trust_remote)
    combos = {"MeanValidator": MeanValidator, "MaxValidator": MaxValidator, "VoteValidator": VoteValidator}
    children = [ForecastModel.load(u, trust_remote=trust_remote) for u in cfg["children"]]
    params = cfg.get("params") or {}
    return combos[cfg["combinator"]](children, **params)


def _warn_version_mismatch(saved: "str | None") -> None:
    if not saved:
        return
    if str(saved).split(".")[:2] != __version__.split(".")[:2]:
        logger.warning(
            f"flow.yaml was written by signalflow {saved}; running {__version__} "
            "(major/minor mismatch - behavior may differ)"
        )


def _validate_registered(flow: Any) -> None:
    from signalflow.enums import ComponentType
    from signalflow.registry import registry

    for det in flow.detectors:
        cfg = det.to_config()
        try:
            registry.get(ComponentType.TRANSFORM, cfg["transform"])
        except UnknownComponentError as e:
            raise ArtifactError(
                f"flow.save: detector {cfg['transform']!r} ({type(det).__qualname__}) is not registered; "
                f"register it with @detector(...) so load can rebuild it"
            ) from e
    strategy_name = _strategy_cfg(flow.strategy).get("name")
    if strategy_name:
        try:
            registry.get(ComponentType.STRATEGY, strategy_name)
        except UnknownComponentError as e:
            raise ArtifactError(
                f"flow.save: strategy {strategy_name!r} ({type(flow.strategy).__qualname__}) is not registered; "
                f"register it with @strategy(...) so load can rebuild it"
            ) from e


def save_flow(flow: Any, path: str, model_dir: str | None = None, run: Any = None) -> str:
    _validate_registered(flow)
    doc = {
        "signalflow_version": __version__,
        "name": flow.name,
        "quote": flow.quote,
        "forecasts": {slot: _model_uri(m, slot, model_dir) for slot, m in flow.forecasts.items()},
        "detectors": [d.to_config() for d in flow.detectors],
        "validator": _validator_cfg(flow.validator, model_dir),
        "strategy": _strategy_cfg(flow.strategy, model_dir, persist=True),
        "risk": {
            "max_drawdown": flow.risk.max_drawdown,
            "max_positions": flow.risk.max_positions,
            "max_notional_per_pair": flow.risk.max_notional_per_pair,
            "kill_switch_path": flow.risk.kill_switch_path,
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)
    logger.debug(
        f"Flow.save: {flow.name!r} -> {path} (forecasts={list(doc['forecasts'])}, "
        f"detectors={[d['transform'] for d in doc['detectors']]}, strategy={doc['strategy']['name']!r}, "
        f"model_dir={model_dir!r})"
    )
    if run is not None:
        import json
        import os

        with open(os.path.join(os.path.dirname(path) or ".", "scorecard.json"), "w", encoding="utf-8") as fh:
            json.dump(run.scorecard(), fh, indent=2)
    return path


def load_flow(path: str, trust_remote: bool = False) -> "Flow":
    """Rebuild a Flow from ``path``; ``trust_remote`` is needed for ``hf://`` model URIs."""
    from signalflow.flow.flow import Flow
    from signalflow.model import ForecastModel

    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    _warn_version_mismatch(doc.get("signalflow_version"))
    logger.debug(f"Flow.load: {doc.get('name')!r} from {path} (signalflow {doc.get('signalflow_version')})")
    forecasts = {
        slot: ForecastModel.load(uri, trust_remote=trust_remote) for slot, uri in (doc.get("forecasts") or {}).items()
    }
    detectors: list[Any] = [build_transform(d) for d in (doc.get("detectors") or [])]
    return Flow(
        name=doc["name"],
        forecasts=forecasts,
        detectors=detectors,
        validator=_build_validator(doc.get("validator"), trust_remote=trust_remote),
        strategy=_build_strategy(doc.get("strategy") or {"name": "rules", "params": {}}),
        risk=Risk(**(doc.get("risk") or {})),
        quote=doc.get("quote", "USDT"),
    )
