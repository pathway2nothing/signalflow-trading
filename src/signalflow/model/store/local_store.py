"""Local-file backend: the layout dir lives directly at the resolved path."""


from pathlib import Path

from signalflow.errors import ArtifactError
from signalflow.model.store._layout import read_layout, write_layout


def save(model, location: str) -> str:
    """Write the artifact layout at ``location`` and return ``file://<path>``."""
    try:
        d = write_layout(model, location)
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"failed to save model to {location!r}: {exc}") from exc
    return f"file://{d.resolve().as_posix()}"


def load(location: str):
    """Read a ForecastModel from the layout dir at ``location``."""
    d = Path(location)
    if not d.exists():
        raise ArtifactError(f"no artifact directory at {location!r}")
    return read_layout(d)
