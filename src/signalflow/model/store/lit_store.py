"""
Lightning AI model store (``litmodels``), the artifact side of the Lightning platform.

URI forms (``lit://`` already stripped to ``location``)::

    models/<name>            upload/download the model named <name>
    models/<name>@<version>  download a specific version
    <name>                   shorthand for models/<name>

A name without a teamspace is prefixed with ``LIGHTNING_TEAMSPACE`` (or the
``teamspace`` an env var supplies), because litmodels addresses models as
``<teamspace>/<model>``. The artifact layout is the framework's own (model.pkl,
oos/, signature.json), uploaded as a directory and read back the same way, so a
model saved to ``lit://`` loads exactly like one saved to ``file://``.
"""

import os
import tempfile
from pathlib import Path

from signalflow.errors import ArtifactError
from signalflow.model.store._layout import read_layout, write_layout

_TEAMSPACE_ENV = ("LIGHTNING_TEAMSPACE", "LIGHTNING_ORG_TEAMSPACE")


def _teamspace() -> "str | None":
    for var in _TEAMSPACE_ENV:
        value = os.environ.get(var, "").strip()
        if value:
            return value.strip("/")
    return None


def _parse_location(location: str) -> "tuple[str, str | None]":
    """Return ``(model_name, version)``; the name is qualified with the teamspace when needed."""
    loc = location.strip().strip("/")
    if loc.startswith("models/"):
        loc = loc[len("models/") :]
    name, sep, version = loc.partition("@")
    if not name:
        raise ArtifactError(f"could not parse model name from lit location {location!r}")
    if "/" not in name:
        teamspace = _teamspace()
        if not teamspace:
            raise ArtifactError(
                f"lit://{location}: no teamspace in the name and no {_TEAMSPACE_ENV[0]} set; "
                f"use lit://models/<teamspace>/<model> or export {_TEAMSPACE_ENV[0]}"
            )
        name = f"{teamspace}/{name}"
    return name, (version if sep else None)


def _litmodels():
    try:
        import litmodels
    except ImportError as exc:
        raise ArtifactError("lit:// artifacts need litmodels (pip install litmodels)") from exc
    return litmodels


def save(model, location: str) -> str:
    """Upload the artifact layout of ``model`` to the Lightning model store."""
    litmodels = _litmodels()
    name, _ = _parse_location(location)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            write_layout(model, tmp)
            info = litmodels.upload_model_files(name=name, path=tmp, progress_bar=False, verbose=0)
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"lit save failed for {location!r}: {exc}") from exc
    version = getattr(info, "version", None)
    return f"lit://models/{name}@{version}" if version else f"lit://models/{name}"


def load(location: str):
    """Download the artifact layout from the Lightning model store and rebuild the model."""
    litmodels = _litmodels()
    name, version = _parse_location(location)
    ref = f"{name}:{version}" if version else name
    try:
        with tempfile.TemporaryDirectory() as tmp:
            litmodels.download_model(name=ref, download_dir=tmp, progress_bar=False)
            root = Path(tmp)
            found = root / "model.pkl"
            if not found.exists():
                found = next(iter(sorted(root.rglob("model.pkl"))), None)
            if found is None:
                raise ArtifactError(f"lit://{location}: downloaded artifact has no model.pkl")
            return read_layout(found.parent)
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"lit load failed for {location!r}: {exc}") from exc
