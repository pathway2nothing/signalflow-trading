"""FeaturePipeline - an ordered, nestable composition of Transforms."""

import time
from collections.abc import Sequence

import polars as pl
import yaml
from loguru import logger

from signalflow._logging import names
from signalflow.decorators import transform
from signalflow.enums import SIGNAL_COL
from signalflow.errors import PipelineError
from signalflow.transform.base import Feature, Transform, build_transform, ensure_sorted

_PROBE_ROWS = 128


def resolved_requires(t: Transform) -> "list[str] | None":
    """Concrete input columns of ``t``: ``required_cols()`` when it resolves templates, else ``requires``."""
    resolver = getattr(t, "required_cols", None)
    if callable(resolver):
        return list(resolver())
    reqs = getattr(t, "requires", None)
    return list(reqs) if isinstance(reqs, (list, tuple)) else None


def _is_plain_feature(t: Transform) -> bool:
    """A :class:`Feature` that relies on the base ``compute`` (pure ``exprs``) and can be fused lazily."""
    return isinstance(t, Feature) and type(t).compute is Feature.compute


@transform("feature_pipeline")
class FeaturePipeline(Transform):
    """Run child transforms in order; outputs are the union of their outputs.

    Accepts the children either unpacked (``FeaturePipeline(a, b)``) or as one
    list (``FeaturePipeline([a, b])``).
    """

    def __init__(self, *transforms: Transform | Sequence[Transform]):
        if len(transforms) == 1 and isinstance(transforms[0], (list, tuple)):
            transforms = tuple(transforms[0])
        for t in transforms:
            if SIGNAL_COL in t.outputs:
                raise PipelineError(
                    f"{t.name!r} outputs {SIGNAL_COL!r}; detectors go in detectors=, "
                    "not in a FeaturePipeline (signal-as-feature is a separate explicit step)"
                )
        self.transforms: tuple[Transform, ...] = tuple(transforms)

    @classmethod
    def from_names(
        cls,
        names: Sequence[str],
        data=None,
        max_warmup: "int | None" = None,
        on_error: str = "raise",
    ) -> "FeaturePipeline":
        """Build a pipeline from registered transform names, validating each against ``data``.

        Unknown names raise ``UnknownComponentError``. With ``data`` given, each stateless
        transform is probed on a small sample; ``on_error="raise"`` re-raises naming the
        feature, ``"drop"`` drops it and logs a WARNING with the dropped names.
        ``max_warmup`` filters transforms with a longer warmup.
        """
        from signalflow.enums import ComponentType
        from signalflow.registry import registry

        if on_error not in ("raise", "drop"):
            raise ValueError(f"on_error must be 'raise' or 'drop', got {on_error!r}")

        sample = data.frame.head(_PROBE_ROWS) if data is not None else None
        kept: list[Transform] = []
        dropped: list[str] = []
        for name in names:
            transform_cls = registry.get(ComponentType.TRANSFORM, name)
            t = transform_cls()
            if max_warmup is not None and t.warmup > max_warmup:
                dropped.append(name)
                continue
            if sample is not None and not t.requires_fit:
                try:
                    t.compute(sample)
                except Exception as exc:
                    if on_error == "raise":
                        raise PipelineError(
                            f"FeaturePipeline.from_names: transform {name!r} failed to compute: {exc}"
                        ) from exc
                    dropped.append(name)
                    continue
            kept.append(t)

        if dropped:
            logger.warning(f"FeaturePipeline.from_names dropped {dropped}")
        return cls(*kept)

    @property
    def warmup(self) -> int:
        """Bars the whole pipeline needs: the largest *effective* warmup of its steps."""
        return max(self.effective_warmups(), default=0)

    def effective_warmups(self) -> list[int]:
        """Per-step warmup with producers added in: a chain sums, independent steps take the max.

        A step that reads a column produced earlier in the pipeline needs its own
        ``warmup`` plus the effective warmup of that producer; raw dataset columns
        add nothing. A step whose ``requires`` is unknown (``None``) is charged the
        largest effective warmup so far, which is exact for encoders, selectors and
        scalers that consume every feature and conservative for anything else.
        """
        producer: dict[str, int] = {}
        effective: list[int] = []
        for t in self.transforms:
            reqs = resolved_requires(t)
            if reqs is None:
                base = max(producer.values(), default=0)
            else:
                base = max((producer.get(c, 0) for c in reqs), default=0)
            eff = int(t.warmup) + base
            effective.append(eff)
            if t.narrows:
                producer = {c: v for c, v in producer.items() if c in t.outputs}
            for c in t.removes:
                producer.pop(c, None)
            for c in t.outputs:
                producer[c] = eff
        return effective

    @property
    def outputs(self) -> list[str]:
        """Columns the pipeline leaves for the model: appended outputs minus what later steps remove.

        A ``narrows`` step (a selector) keeps only its ``outputs``; an unfitted stateful
        step whose outputs are not yet known leaves the running list unchanged.
        """
        cols: list[str] = []
        for t in self.transforms:
            produced = list(t.outputs)
            if t.requires_fit and not t.is_fitted and not produced:
                continue
            if t.narrows:
                cols = [c for c in cols if c in produced] + [c for c in produced if c not in cols]
                continue
            removed = set(t.removes)
            cols = [c for c in cols if c not in removed] + [c for c in produced if c not in cols]
        return cols

    @property
    def is_fitted(self) -> bool:
        return all(t.is_fitted for t in self.transforms)

    def split(self) -> "tuple[FeaturePipeline, FeaturePipeline]":
        """``(prefix, tail)``: the stateless steps before the first ``requires_fit`` step, and the rest.

        A model computes the prefix once over all data and refits a clone of the
        tail inside every fold. The tail is empty when nothing is stateful.
        """
        for i, t in enumerate(self.transforms):
            if t.requires_fit:
                return FeaturePipeline(*self.transforms[:i]), FeaturePipeline(*self.transforms[i:])
        return FeaturePipeline(*self.transforms), FeaturePipeline()

    def clone(self) -> "FeaturePipeline":
        return FeaturePipeline(*[t.clone() for t in self.transforms])

    @property
    def requires_fit(self) -> bool:
        return any(t.requires_fit for t in self.transforms)

    @property
    def requires_target(self) -> bool:
        return any(t.requires_target for t in self.transforms)

    def fit(self, df: pl.DataFrame, target: pl.Series | None = None) -> "FeaturePipeline":
        """Fit stateful children in order, each on the frame produced so far."""
        cur = ensure_sorted(df)
        for t in self.transforms:
            if t.requires_fit:
                t0 = time.perf_counter()
                t.fit(cur, target if t.requires_target else None)
                t._is_fitted = True
                logger.debug(
                    f"FeaturePipeline.fit: {t.name} fitted on rows={cur.height:,} ({time.perf_counter() - t0:.2f}s)"
                )
            cur = t.compute(cur)
        return self

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Run the children in order on one (pair, ts)-sorted frame.

        The input is sorted at most once. Runs of plain expression features
        (those that keep :meth:`Feature.compute`) are chained as one lazy query
        and collected together, so their intermediate frames are never
        materialized; any other transform (stateful, custom ``compute``) is run
        eagerly on the collected frame, seeing every earlier output.
        """
        t_all = time.perf_counter()
        cur = ensure_sorted(df)
        n_in = len(cur.columns)
        lazy: pl.LazyFrame | None = None
        fused: list[str] = []
        t0 = time.perf_counter()

        def collect(lf: pl.LazyFrame) -> pl.DataFrame:
            out = lf.collect()
            logger.debug(
                f"FeaturePipeline: fused {len(fused)} features -> +{names(fused)} ({time.perf_counter() - t0:.2f}s)"
            )
            fused.clear()
            return out

        for t in self.transforms:
            if _is_plain_feature(t):
                if lazy is None:
                    t0 = time.perf_counter()
                lazy = (cur.lazy() if lazy is None else lazy).with_columns(t.window_exprs())
                fused.extend(t.outputs)
                continue
            if lazy is not None:
                cur, lazy = collect(lazy), None
            t0 = time.perf_counter()
            cur = t.compute(cur)
            logger.debug(f"FeaturePipeline: {t.name} -> +{names(t.outputs)} ({time.perf_counter() - t0:.2f}s)")
        if lazy is not None:
            cur = collect(lazy)
        logger.debug(
            f"FeaturePipeline.compute: {len(self.transforms)} transforms, rows={cur.height:,}, "
            f"cols {n_in} -> {len(cur.columns)} ({time.perf_counter() - t_all:.2f}s)"
        )
        return cur

    def to_config(self) -> dict:
        return {
            "transform": "feature_pipeline",
            "role": "pipeline",
            "params": {"transforms": [t.to_config() for t in self.transforms]},
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "FeaturePipeline":
        children = [build_transform(c) for c in (cfg.get("params") or {}).get("transforms", [])]
        return cls(*children)

    def save(self, path: str) -> str:
        """Serialize the pipeline (config only) to a portable YAML file."""
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_config(), fh, sort_keys=False, allow_unicode=True)
        return path

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        """Rebuild a pipeline from a YAML file written by :meth:`save`; reject non-pipeline roots."""
        with open(path, encoding="utf-8") as fh:
            result = build_transform(yaml.safe_load(fh))
        if not isinstance(result, FeaturePipeline):
            root = getattr(result, "name", type(result).__name__)
            raise PipelineError(f"{path} does not describe a FeaturePipeline; its root is {root!r}")
        return result

    def __repr__(self) -> str:
        return f"FeaturePipeline({', '.join(t.name for t in self.transforms)})"
