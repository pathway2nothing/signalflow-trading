"""FlowBundle - a promotable flow directory with an evidence manifest."""

import json
from pathlib import Path

import yaml

from signalflow._version import __version__
from signalflow.errors import ArtifactError

MANIFEST_NAME = "manifest.json"
SCORECARD_NAME = "scorecard.json"
FLOW_NAME = "flow.yaml"
MODELS_DIR = "models"
MANIFEST_VERSION = 1
MIN_OOS_COVERAGE = 0.95


def _span(run) -> "list[str | None]":
    ec = run.equity_curve
    if ec.height == 0 or "ts" not in ec.columns:
        return [None, None]
    ts = ec.get_column("ts")
    return [str(ts.min()), str(ts.max())]


def write_bundle(flow, run, dir_path: "str | Path") -> str:
    """Write a promotable bundle: flow.yaml + models + scorecard.json + manifest.json."""
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    flow.save(str(d / FLOW_NAME), model_dir=str(d / MODELS_DIR), run=run)
    card = run.scorecard()
    models = {}
    for slot, model in flow.forecasts.items():
        fingerprint = getattr(model, "fingerprint", {}) or {}
        models[slot] = {"uri": getattr(model, "_uri", None), "fingerprint_id": fingerprint.get("id")}
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "name": flow.name,
        "signalflow_version": __version__,
        "models": models,
        "run": {
            "mode": card.get("mode"),
            "oos": card.get("oos"),
            "oos_coverage": card.get("oos_coverage"),
            "promotable": card.get("promotable"),
            "span": _span(run),
            "final_equity": card.get("final_equity"),
            "sharpe": card.get("sharpe"),
        },
    }
    (d / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(d)


def read_manifest(dir_path: "str | Path") -> dict:
    """Load a bundle's manifest.json; raise ``ArtifactError`` when absent or unparseable."""
    path = Path(dir_path) / MANIFEST_NAME
    if not path.exists():
        raise ArtifactError(f"no {MANIFEST_NAME} in bundle {dir_path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        raise ArtifactError(f"could not read {path}: {e}") from e


def validate_bundle(dir_path: "str | Path") -> "list[str]":
    """Return human-readable problems that block promotion (empty list means valid)."""
    d = Path(dir_path)
    problems: list[str] = []
    for name in (FLOW_NAME, SCORECARD_NAME, MANIFEST_NAME):
        if not (d / name).exists():
            problems.append(f"missing {name}")
    if problems:
        return problems
    manifest = read_manifest(d)
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        problems.append(f"manifest_version {manifest.get('manifest_version')} != {MANIFEST_VERSION}")
    run = manifest.get("run") or {}
    if not run.get("promotable"):
        problems.append("run.promotable is false")
    if not run.get("oos"):
        problems.append("run.oos is false")
    cov = run.get("oos_coverage")
    if cov is not None and cov < MIN_OOS_COVERAGE:
        problems.append(f"run.oos_coverage {cov} < {MIN_OOS_COVERAGE}")
    saved = str(manifest.get("signalflow_version", ""))
    if saved.split(".")[:2] != __version__.split(".")[:2]:
        problems.append(f"signalflow_version {saved} != running {__version__} (major/minor mismatch)")
    models = manifest.get("models") or {}
    try:
        doc = yaml.safe_load((d / FLOW_NAME).read_text(encoding="utf-8"))
        for slot in doc.get("forecasts") or {}:
            if slot not in models:
                problems.append(f"forecast slot {slot!r} in flow.yaml is absent from manifest models")
    except (OSError, ValueError) as e:
        problems.append(f"could not cross-check flow.yaml slots: {e}")
    return problems
