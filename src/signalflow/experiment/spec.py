"""Declarative research: run an ``experiment.yaml`` end to end from the framework helpers.

A single spec pins the data, model, walk-forward (or single fit) scheme, metrics, an
optional backtest, and optional MLflow tracking - reproducible from the yaml + seed alone.
"""

import copy
import json
from pathlib import Path

import yaml

from signalflow.errors import FlowConfigError
from signalflow.experiment.seeding import seed_everything
from signalflow.experiment.tracking import experiment_run

_ALLOWED_TOP = {"kind", "name", "seed", "data", "model", "scheme", "metrics", "backtest", "tracking"}


def load_spec(path: "str | Path") -> dict:
    """Parse an experiment.yaml, rejecting unknown top-level keys."""
    with open(path, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    if not isinstance(spec, dict):
        raise FlowConfigError(f"experiment spec {path} is not a mapping")
    unknown = set(spec) - _ALLOWED_TOP
    if unknown:
        raise FlowConfigError(
            f"experiment spec has unknown top-level keys {sorted(unknown)}; allowed: {sorted(_ALLOWED_TOP)}"
        )
    return spec


def _build_model(mcfg: dict):
    from signalflow.model.forecast import ForecastModel
    from signalflow.target.base import make_target
    from signalflow.transform.base import build_transform
    from signalflow.transform.encode import IVSelector, WoE
    from signalflow.transform.pipe import FeaturePipe

    target = make_target(mcfg["target"]["name"], **(mcfg["target"].get("params") or {}))
    pipe = FeaturePipe(*[build_transform(f) for f in (mcfg.get("features") or [])])
    if mcfg.get("encode", "default") == "default":
        encode, select = WoE(), IVSelector()
    else:
        encode, select = None, None
    return ForecastModel(
        backend=mcfg.get("backend", "lightgbm"),
        target=target,
        features=pipe,
        encode=encode,
        select=select,
        backend_params=mcfg.get("backend_params") or {},
        output=mcfg.get("output", "p_rise"),
    )


def _fold_scores(result, metrics: list) -> list:
    combined = None
    for metric in metrics:
        frame = result.evaluate(metric).rename({"score": metric})
        combined = frame if combined is None else combined.join(frame.select(["fold", metric]), on="fold", how="left")
    return combined.to_dicts() if combined is not None else []


def _run_backtest(spec: dict, model, ds) -> dict:
    from signalflow.flow.flow import Flow
    from signalflow.strategy.base import build_strategy
    from signalflow.strategy.rules import RulesStrategy
    from signalflow.transform.base import build_transform

    bt = spec["backtest"]
    slot = bt.get("slot", "rise")
    detectors = [build_transform(d) for d in (bt.get("detectors") or [])]
    strategy = build_strategy(bt["strategy"]) if bt.get("strategy") else RulesStrategy()
    flow = Flow(name=spec.get("name", "exp"), forecasts={slot: model}, detectors=detectors, strategy=strategy)
    run = flow.backtest(ds, capital=bt.get("capital", 10_000), oos=bt.get("oos", False))
    return run.scorecard()


def _flatten(spec: dict) -> dict:
    flat: dict = {}
    for section, value in spec.items():
        if isinstance(value, dict):
            for key, inner in value.items():
                flat[f"{section}.{key}"] = str(inner)
        else:
            flat[section] = str(value)
    return flat


def run_experiment(path: "str | Path") -> dict:
    """Run an experiment.yaml and return ``{folds, model_scorecard, run_scorecard, spec}``.

    Writes the same dict to ``results.json`` next to the spec.
    """
    from signalflow.data.dataset import data as build_data
    from signalflow.model.metrics import classification_scorecard
    from signalflow.model.walkforward import walk_forward

    spec = load_spec(path)
    seed_everything(int(spec.get("seed", 0)))
    ds = build_data(**spec["data"])
    template = _build_model(spec["model"])
    metrics = spec.get("metrics") or ["auc"]
    scheme = spec.get("scheme") or {}

    folds = None
    if "walk_forward" in scheme:
        result = walk_forward(template, ds, **scheme["walk_forward"])
        folds = _fold_scores(result, metrics)

    model_scorecard = None
    fitted_full = None
    if "fit" in scheme or "backtest" in spec:
        fitted_full = copy.deepcopy(template).fit(ds)
        model_scorecard = classification_scorecard(fitted_full, ds)

    run_scorecard = None
    if "backtest" in spec:
        run_scorecard = _run_backtest(spec, fitted_full, ds)

    result_dict = {"folds": folds, "model_scorecard": model_scorecard, "run_scorecard": run_scorecard, "spec": spec}
    Path(path).with_name("results.json").write_text(json.dumps(result_dict, indent=2, default=str), encoding="utf-8")

    tracking = spec.get("tracking") or {}
    if tracking.get("mlflow"):
        with experiment_run(tracking["mlflow"], params=_flatten(spec)) as mlflow_mod:
            if mlflow_mod is not None:
                for card, prefix in ((model_scorecard, "model"), (run_scorecard, "run")):
                    if card:
                        mlflow_mod.log_metrics(
                            {f"{prefix}.{k}": float(v) for k, v in card.items() if isinstance(v, (int, float))}
                        )
    return result_dict
