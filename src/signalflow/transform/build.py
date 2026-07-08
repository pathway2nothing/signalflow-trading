"""Registry-name -> validated FeaturePipe builder."""

from collections.abc import Sequence

from loguru import logger

from signalflow.data.dataset import Dataset
from signalflow.enums import ComponentType
from signalflow.errors import PipeError
from signalflow.registry import registry
from signalflow.transform.base import Transform
from signalflow.transform.pipe import FeaturePipe

_PROBE_ROWS = 128


def build_pipe(
    names: Sequence[str],
    data: "Dataset | None" = None,
    max_warmup: "int | None" = None,
    on_error: str = "raise",
) -> FeaturePipe:
    """Build a FeaturePipe from registered transform names, validating each against ``data``.

    Unknown names raise ``UnknownComponentError``. With ``data`` given, each transform is
    probed on a small sample; ``on_error="raise"`` re-raises naming the feature, ``"drop"``
    drops it and logs a WARNING with the dropped names. ``max_warmup`` filters longer warmups.
    """
    if on_error not in ("raise", "drop"):
        raise ValueError(f"on_error must be 'raise' or 'drop', got {on_error!r}")

    sample = data.frame.head(_PROBE_ROWS) if data is not None else None
    kept: list[Transform] = []
    dropped: list[str] = []
    for name in names:
        cls = registry.get(ComponentType.TRANSFORM, name)
        transform = cls()
        if max_warmup is not None and transform.warmup > max_warmup:
            dropped.append(name)
            continue
        if sample is not None and not transform.requires_fit:
            try:
                transform.compute(sample)
            except Exception as exc:
                if on_error == "raise":
                    raise PipeError(f"build_pipe: transform {name!r} failed to compute: {exc}") from exc
                dropped.append(name)
                continue
        kept.append(transform)

    if dropped:
        logger.warning(f"build_pipe dropped {dropped}")
    return FeaturePipe(*kept)
