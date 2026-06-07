"""ModelRef — declarative, versioned reference to a forecast model artifact.

A ModelRef is a *pinned* pointer to a model that lives in some external
registry (MLflow, HuggingFace, ...). It carries no weights — only enough
metadata to resolve the artifact later. This keeps the trading pipeline
decoupled from training: models are produced elsewhere and arrive as
versioned, reproducible artifacts.

Invariants:
    - ``version`` is mandatory (empty -> ValueError).
    - ``version == "latest"`` is forbidden unless ``SF_ALLOW_LATEST=1`` is set
      in the environment, because a floating ``latest`` silently breaks parity
      and reproducibility between training and live inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_VALID_SOURCES = frozenset({"mlflow", "hf"})

ALLOW_LATEST_ENV = "SF_ALLOW_LATEST"


def _latest_allowed() -> bool:
    """True only when SF_ALLOW_LATEST=1 (dev mode opt-in)."""
    return os.environ.get(ALLOW_LATEST_ENV) == "1"


@dataclass(frozen=True)
class ModelRef:
    """A pinned, versioned reference to a model artifact.

    Attributes:
        name: Registered model name (e.g. ``"revert"``). Non-empty.
        version: Mandatory version. Usually a numeric string/int (``"3"``),
            but may be ``"latest"`` only in dev mode (see module docstring).
        source: Backing registry, one of ``{"mlflow", "hf"}``. Default ``"mlflow"``.

    Construction is cheap and never touches the network or weights — use a
    :class:`~signalflow.models.resolver.Resolver` to actually load the model.

    Example:
        >>> ModelRef.parse("models:/revert/3")
        ModelRef(name='revert', version='3', source='mlflow')
        >>> ModelRef.parse("revert@3")
        ModelRef(name='revert', version='3', source='mlflow')
    """

    name: str
    version: str | int
    source: str = "mlflow"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ModelRef.name must be a non-empty string")

        version_str = str(self.version).strip()
        if not version_str:
            raise ValueError("ModelRef.version is mandatory and must be non-empty")

        if self.source not in _VALID_SOURCES:
            raise ValueError(f"ModelRef.source must be one of {sorted(_VALID_SOURCES)}, got {self.source!r}")

        if version_str.lower() == "latest" and not _latest_allowed():
            raise ValueError(
                "ModelRef.version='latest' is forbidden: a floating version breaks "
                "parity/reproducibility between training and live inference. "
                f"Pin an explicit version, or set {ALLOW_LATEST_ENV}=1 for dev only."
            )

    @classmethod
    def parse(cls, spec: str, *, source: str = "mlflow") -> ModelRef:
        """Parse a ModelRef from a compact string spec.

        Supported forms:
            - MLflow URI: ``"models:/<name>/<version>"`` (forces ``source="mlflow"``).
            - At-spec: ``"<name>@<version>"`` (uses the ``source`` argument).

        Args:
            spec: The string to parse.
            source: Source to use for non-URI specs. Default ``"mlflow"``.

        Returns:
            A validated ModelRef.

        Raises:
            ValueError: If the spec is malformed or violates ModelRef invariants.
        """
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("ModelRef spec must be a non-empty string")

        text = spec.strip()

        if text.startswith("models:/"):
            rest = text[len("models:/") :]
            parts = rest.split("/")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"invalid MLflow model URI {spec!r}; expected 'models:/<name>/<version>'")
            return cls(name=parts[0], version=parts[1], source="mlflow")

        if "@" in text:
            name, _, version = text.partition("@")
            if not name or not version:
                raise ValueError(f"invalid model spec {spec!r}; expected '<name>@<version>'")
            return cls(name=name, version=version, source=source)

        raise ValueError(f"unrecognized model spec {spec!r}; expected 'models:/<name>/<version>' or '<name>@<version>'")

    @property
    def uri(self) -> str:
        """MLflow-style URI for this ref: ``models:/<name>/<version>``."""
        return f"models:/{self.name}/{self.version}"
