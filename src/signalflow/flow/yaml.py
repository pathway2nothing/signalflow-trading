"""Flow serialization - deployment is data, not code."""

import yaml

from signalflow.enums import ComponentType
from signalflow.errors import ArtifactError
from signalflow.registry import registry
from signalflow.strategy.risk import Risk
from signalflow.strategy.rules import Entry, Exit, RulesStrategy
from signalflow.transform.base import build_transform


def _model_uri(model, slot: str, model_dir: str | None) -> str:
    uri = getattr(model, "_uri", None)
    if uri:
        return uri
    if model_dir:
        return model.save(f"file://{model_dir}/{slot}")
    raise ArtifactError(f"forecast {slot!r} has no pinned URI; save it first or pass model_dir=")


def _strategy_cfg(strategy) -> dict:
    if isinstance(strategy, RulesStrategy):
        return {"name": "rules", "entry": vars(strategy.entry), "exit": vars(strategy.exit)}
    return {"name": getattr(strategy, "_sf_name", type(strategy).__name__)}


def _build_strategy(cfg: dict):
    if cfg.get("name") == "rules":
        return RulesStrategy(entry=Entry(**cfg["entry"]), exit=Exit(**cfg["exit"]))
    return registry.create(ComponentType.STRATEGY, cfg["name"])


def _validator_cfg(validator, model_dir: str | None) -> dict | None:
    if validator is None:
        return None
    if hasattr(validator, "children"):
        return {
            "combinator": type(validator).__name__,
            "children": [_model_uri(c, f"val{i}", model_dir) for i, c in enumerate(validator.children)],
        }
    return {"uri": _model_uri(validator, "validator", model_dir)}


def _build_validator(cfg: dict | None):
    if cfg is None:
        return None
    from signalflow.model import ForecastModel, MaxValidator, MeanValidator, VoteValidator

    if "uri" in cfg:
        return ForecastModel.load(cfg["uri"])
    combos = {"MeanValidator": MeanValidator, "MaxValidator": MaxValidator, "VoteValidator": VoteValidator}
    children = [ForecastModel.load(u) for u in cfg["children"]]
    return combos[cfg["combinator"]](children)


def save_flow(flow, path: str, model_dir: str | None = None) -> str:
    doc = {
        "name": flow.name,
        "quote": flow.quote,
        "forecasts": {slot: _model_uri(m, slot, model_dir) for slot, m in flow.forecasts.items()},
        "detectors": [d.to_config() for d in flow.detectors],
        "validator": _validator_cfg(flow.validator, model_dir),
        "strategy": _strategy_cfg(flow.strategy),
        "risk": {
            "max_drawdown": flow.risk.max_drawdown,
            "max_positions": flow.risk.max_positions,
            "max_notional_per_pair": flow.risk.max_notional_per_pair,
            "kill_switch_path": flow.risk.kill_switch_path,
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)
    return path


def load_flow(path: str):
    from signalflow.flow.flow import Flow
    from signalflow.model import ForecastModel

    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    forecasts = {slot: ForecastModel.load(uri) for slot, uri in (doc.get("forecasts") or {}).items()}
    detectors = [build_transform(d) for d in (doc.get("detectors") or [])]
    return Flow(
        name=doc["name"],
        forecasts=forecasts,
        detectors=detectors,
        validator=_build_validator(doc.get("validator")),
        strategy=_build_strategy(doc.get("strategy") or {"name": "rules", "entry": {}, "exit": {}}),
        risk=Risk(**(doc.get("risk") or {})),
        quote=doc.get("quote", "USDT"),
    )
