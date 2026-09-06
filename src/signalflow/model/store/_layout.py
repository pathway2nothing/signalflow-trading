"""
On-disk artifact layout.

A model directory contains::

model.pkl              cloudpickle of the entire fitted ForecastModel
    oos/predictions.parquet  model.oos_ (also queryable standalone)
    oos/fingerprint.json     model.fingerprint
    signature.json           model.feature_signature

``read_layout`` reconstructs the ForecastModel from ``model.pkl`` and, if the
parquet OOS file is present, re-attaches it as the authoritative ``oos_`` copy.
"""


import json
from pathlib import Path

import cloudpickle
import polars as pl
from loguru import logger

from signalflow.errors import ArtifactError

MODEL_PKL = "model.pkl"
OOS_DIR = "oos"
OOS_PARQUET = "predictions.parquet"
FINGERPRINT_JSON = "fingerprint.json"
SIGNATURE_JSON = "signature.json"
ENV_JSON = "env.json"

_ENV_LIBS = ("polars", "numpy", "sklearn", "lightgbm", "cloudpickle", "torch")


def environment() -> dict:
    """Versions the pickled model depends on (python, signalflow and the numeric stack)."""
    import importlib
    import platform

    from signalflow._version import __version__

    env = {"python": platform.python_version(), "signalflow": __version__}
    for lib in _ENV_LIBS:
        try:
            env[lib] = getattr(importlib.import_module(lib), "__version__", "?")
        except Exception:
            continue
    return env


def _major_minor(version: str) -> str:
    return ".".join(str(version).split(".")[:2])


def check_environment(saved: dict) -> list[str]:
    """Libraries whose major.minor differs from the saved artifact's (empty when compatible)."""
    current = environment()
    return [
        f"{lib} {saved[lib]} (artifact) vs {current[lib]} (now)"
        for lib in saved
        if lib in current and _major_minor(saved[lib]) != _major_minor(current[lib])
    ]


def write_layout(model, directory: str | Path) -> Path:
    """Write the full artifact layout for ``model`` under ``directory``."""
    if not getattr(model, "is_fitted", False):
        raise ArtifactError("cannot save an unfitted ForecastModel")

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    oos_dir = d / OOS_DIR
    oos_dir.mkdir(parents=True, exist_ok=True)


    with (d / MODEL_PKL).open("wb") as fh:
        cloudpickle.dump(model, fh)


    oos = getattr(model, "oos_", None)
    if oos is not None:
        oos.write_parquet(oos_dir / OOS_PARQUET)


    fingerprint = getattr(model, "fingerprint", None)
    if fingerprint is not None:
        (oos_dir / FINGERPRINT_JSON).write_text(
            json.dumps(fingerprint, indent=2, default=str), encoding="utf-8"
        )


    signature = getattr(model, "feature_signature", None)
    if signature is not None:
        (d / SIGNATURE_JSON).write_text(
            json.dumps(signature, indent=2, default=str), encoding="utf-8"
        )

    (d / ENV_JSON).write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    return d


def read_layout(directory: str | Path):
    """Reconstruct a fitted ForecastModel from an artifact layout directory."""
    d = Path(directory)
    pkl = d / MODEL_PKL
    if not pkl.is_file():
        raise ArtifactError(f"no {MODEL_PKL} found under {d}")
    env_file = d / ENV_JSON
    if env_file.is_file():
        try:
            mismatches = check_environment(json.loads(env_file.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            mismatches = []
        if mismatches:
            logger.warning(
                f"model artifact {d} was saved under a different environment: {'; '.join(mismatches)}; "
                f"unpickled behaviour may differ - retrain to be safe"
            )

    try:
        with pkl.open("rb") as fh:
            model = cloudpickle.load(fh)
    except Exception as exc:
        raise ArtifactError(f"failed to unpickle model from {pkl}: {exc}") from exc


    oos_parquet = d / OOS_DIR / OOS_PARQUET
    if oos_parquet.is_file():
        try:
            model.oos_ = pl.read_parquet(oos_parquet)
        except Exception as exc:
            raise ArtifactError(f"failed to read OOS parquet {oos_parquet}: {exc}") from exc

    model._fitted = True
    return model
