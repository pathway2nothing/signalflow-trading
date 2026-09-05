"""FeaturePipe - an ordered, nestable composition of Transforms."""

import time

import polars as pl
import yaml
from loguru import logger

from signalflow._logging import names
from signalflow.decorators import transform
from signalflow.enums import SIGNAL_COL
from signalflow.errors import PipeError
from signalflow.transform.base import Feature, Transform, build_transform, ensure_sorted


def _is_plain_feature(t: Transform) -> bool:
    """A :class:`Feature` that relies on the base ``compute`` (pure ``exprs``) and can be fused lazily."""
    return isinstance(t, Feature) and type(t).compute is Feature.compute


@transform("feature_pipe")
class FeaturePipe(Transform):
    """Run child transforms in order; outputs are the union of their outputs."""

    def __init__(self, *transforms: Transform):
        for t in transforms:
            if SIGNAL_COL in t.outputs:
                raise PipeError(
                    f"{t.name!r} outputs {SIGNAL_COL!r}; detectors go in detectors=, "
                    "not in a FeaturePipe (signal-as-feature is a separate explicit step)"
                )
        self.transforms: tuple[Transform, ...] = transforms

    @property
    def warmup(self) -> int:
        return max((t.warmup for t in self.transforms), default=0)

    @property
    def outputs(self) -> list[str]:
        out: list[str] = []
        for t in self.transforms:
            out.extend(t.outputs)
        return out

    @property
    def requires_fit(self) -> bool:
        return any(t.requires_fit for t in self.transforms)

    @property
    def requires_target(self) -> bool:
        return any(t.requires_target for t in self.transforms)

    def fit(self, df: pl.DataFrame, target: pl.Series | None = None) -> "FeaturePipe":
        """Fit stateful children in order, each on the frame produced so far."""
        cur = ensure_sorted(df)
        for t in self.transforms:
            if t.requires_fit:
                t0 = time.perf_counter()
                t.fit(cur, target if t.requires_target else None)
                logger.debug(f"FeaturePipe.fit: {t.name} fitted on rows={cur.height:,} ({time.perf_counter() - t0:.2f}s)")
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
            logger.debug(f"FeaturePipe: fused {len(fused)} features -> +{names(fused)} ({time.perf_counter() - t0:.2f}s)")
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
            logger.debug(f"FeaturePipe: {t.name} -> +{names(t.outputs)} ({time.perf_counter() - t0:.2f}s)")
        if lazy is not None:
            cur = collect(lazy)
        logger.debug(
            f"FeaturePipe.compute: {len(self.transforms)} transforms, rows={cur.height:,}, "
            f"cols {n_in} -> {len(cur.columns)} ({time.perf_counter() - t_all:.2f}s)"
        )
        return cur

    def to_config(self) -> dict:
        return {
            "transform": "feature_pipe",
            "role": "pipe",
            "params": {"transforms": [t.to_config() for t in self.transforms]},
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "FeaturePipe":
        children = [build_transform(c) for c in (cfg.get("params") or {}).get("transforms", [])]
        return cls(*children)

    def save(self, path: str) -> str:
        """Serialize the pipe (config only) to a portable YAML file."""
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_config(), fh, sort_keys=False, allow_unicode=True)
        return path

    @classmethod
    def load(cls, path: str) -> "FeaturePipe":
        """Rebuild a pipe from a YAML file written by :meth:`save`; reject non-pipe roots."""
        with open(path, encoding="utf-8") as fh:
            result = build_transform(yaml.safe_load(fh))
        if not isinstance(result, FeaturePipe):
            root = getattr(result, "name", type(result).__name__)
            raise PipeError(f"{path} does not describe a FeaturePipe; its root is {root!r}")
        return result
