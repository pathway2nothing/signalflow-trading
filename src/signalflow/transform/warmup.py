"""Warmup canary - measure the bars a transform really needs and compare with its declaration.

A transform declares ``warmup``: bars of history before its output at a bar is
valid. Nothing used to check that number, and an under-declared warmup surfaced
late as ``simulate != backtest`` or as null features in live. The canary makes
it explicit: compute the transform on a long deterministic synthetic series, then
on a trailing window of exactly the declared bars, and require the last row(s)
to agree (nulls must match; floats within a tolerance, so recursive filters such
as EMA/RSI pass once they have converged).

Chains are additive: a transform that reads another transform's output needs its
own warmup plus the producer's. :meth:`FeaturePipeline.effective_warmups` composes
declarations that way, and :func:`check_pipeline` measures each step in the
context of the steps before it, so the comparison is effective vs effective.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from signalflow._time import interval_seconds, to_epoch
from signalflow.errors import WarmupError

_START = "2024-01-01"
_MIN_CAP = 512
_MAX_CAP = 8192
_PAIRS = ("BTCUSDT", "ETHUSDT")
_SPARSE_ROWS = 32
_cache: dict[str, "WarmupCheck"] = {}


@dataclass(frozen=True)
class WarmupCheck:
    """Outcome of one measurement: ``ok`` when the declaration covers the measured need."""

    component: str
    declared: int
    measured: int | None
    ok: bool
    detail: str = ""

    def __str__(self) -> str:
        measured = "?" if self.measured is None else str(self.measured)
        tail = f" - {self.detail}" if self.detail else ""
        return f"{'ok  ' if self.ok else 'FAIL'} {self.component}: declared={self.declared} measured={measured}{tail}"


def synthetic_frame(bars: int, interval: str = "1h", pairs: Sequence[str] = _PAIRS, seed: int = 7) -> pl.DataFrame:
    """``bars`` closed bars per pair of the deterministic synthetic source."""
    from signalflow.data.source.synthetic import SyntheticSource

    step = interval_seconds(interval)
    start = to_epoch(_START)
    return SyntheticSource(seed=seed).fetch(list(pairs), start=_START, end=start + bars * step, interval=interval)


def _tail(frame: pl.DataFrame, bars: int) -> pl.DataFrame:
    ts = frame.get_column("ts").unique().sort()
    if bars >= ts.len():
        return frame
    return frame.filter(pl.col("ts") >= ts[-bars])


def _apply(inputs: Any, transform: Any, frame: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    cur = frame if inputs is None else inputs.compute(frame)
    before = set(cur.columns)
    out = transform.compute(cur)
    return out, [c for c in out.columns if c not in before]


def _last_rows(frame: pl.DataFrame, rows: int) -> pl.DataFrame:
    return frame.sort(["pair", "ts"]).group_by("pair", maintain_order=True).tail(rows)


def _null_count(col: pl.Series) -> int:
    nulls = int(col.is_null().sum())
    if col.dtype in (pl.Float32, pl.Float64):
        nulls += int(col.is_nan().sum())
    return nulls


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _values_equal(probe: Any, ref: Any, rtol: float, atol: float) -> bool:
    if _is_missing(probe) or _is_missing(ref):
        return _is_missing(probe) and _is_missing(ref)
    if isinstance(probe, (int, float)) and isinstance(ref, (int, float)) and not isinstance(probe, bool):
        return abs(probe - ref) <= atol + rtol * abs(ref)
    return probe == ref


def _agree(ref: pl.DataFrame, out: pl.DataFrame, cols: list[str], rows: int, rtol: float, atol: float) -> bool:
    key = ["pair", "ts"]
    r = _last_rows(ref.select([*key, *cols]), rows)
    o = _last_rows(out.select([*key, *cols]), rows)
    joined = r.join(o, on=key, how="inner", suffix="__probe")
    if joined.height != r.height:
        return False
    for row in joined.iter_rows(named=True):
        for c in cols:
            if not _values_equal(row[f"{c}__probe"], row[c], rtol, atol):
                return False
    return True


def _key(transform: Any, inputs: Any, **kw: Any) -> str:
    def cfg(t: Any) -> Any:
        try:
            return t.to_config()
        except Exception:
            return repr(t)

    return json.dumps(
        {"t": cfg(transform), "in": cfg(inputs) if inputs is not None else None, **kw}, sort_keys=True, default=str
    )


def measure_warmup(
    transform: Any,
    *,
    declared: int | None = None,
    inputs: Any = None,
    interval: str = "1h",
    pairs: Sequence[str] = _PAIRS,
    cap: int | None = None,
    rows: int = 1,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    exact: bool = False,
    name: str | None = None,
) -> WarmupCheck:
    """Measure how many bars ``transform`` needs and compare with ``declared`` (default: its ``warmup``).

    ``inputs`` (a pipeline) is computed first so the transform sees its producers;
    ``declared`` should then be the *effective* number. The last ``rows`` rows of
    a window must equal the long-history values (``rows=1`` for features; larger
    for sparse signal columns). ``cap`` bounds the search (default ``4 x declared``,
    at least 512). With ``exact=True`` the minimal need is bisected even when the
    declaration already holds, so the slack is reported.
    """
    name = name or getattr(transform, "name", type(transform).__name__)
    declared = int(getattr(transform, "warmup", 0)) if declared is None else int(declared)
    cap = min(max(4 * declared, _MIN_CAP), _MAX_CAP) if cap is None else int(cap)
    cap = max(cap, declared + rows)
    key = _key(
        transform, inputs, interval=interval, pairs=list(pairs), cap=cap, rows=rows, rtol=rtol, atol=atol, exact=exact
    )
    if key in _cache:
        return _cache[key]

    def done(check: WarmupCheck) -> WarmupCheck:
        _cache[key] = check
        return check

    if getattr(transform, "warmup_invariant", True) is False:
        return done(
            WarmupCheck(
                name,
                declared,
                None,
                False,
                "declares warmup_invariant=False: output depends on where the series starts",
            )
        )

    length = cap + rows
    base = synthetic_frame(length, interval, pairs)
    try:
        ref, cols = _apply(inputs, transform, base)
    except Exception as exc:
        return done(WarmupCheck(name, declared, None, False, f"compute failed on {length} bars: {exc}"))
    if not cols:
        return done(WarmupCheck(name, declared, None, True, "adds no columns; nothing to measure"))
    # A feature may be undefined on most bars by design (e.g. "no shock in the last N bars"):
    # a null on the last row proves nothing, so such columns are compared over a longer tail
    # and only count as dead when the reference column is null everywhere.
    probe_rows = rows
    tail = _last_rows(ref.select(["pair", "ts", *cols]), rows)
    if any(_null_count(tail.get_column(c)) > 0 for c in cols):
        probe_rows = max(rows, _SPARSE_ROWS)
        tail = _last_rows(ref.select(["pair", "ts", *cols]), probe_rows)
    dead = [c for c in cols if _null_count(ref.get_column(c)) == ref.height]
    if dead:
        return done(WarmupCheck(name, declared, None, False, f"still null after {cap} bars: {dead}"))
    rows = probe_rows

    def stable(window: int) -> bool:
        if window < rows:
            return False
        try:
            out, _ = _apply(inputs, transform, _tail(base, window))
        except Exception:
            return False
        return _agree(ref, out, cols, rows, rtol, atol)

    need = max(declared, 1) + rows - 1
    if stable(need):
        if not exact:
            return done(WarmupCheck(name, declared, declared, True))
        lo, hi = rows - 1, need
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if stable(mid):
                hi = mid
            else:
                lo = mid
        measured = hi - rows + 1
        return done(WarmupCheck(name, declared, measured, True, f"slack={declared - measured}"))

    top = length - 1
    lo, hi, n = need, None, need
    while n < top:
        n = min(max(2 * n, n + 1), top)
        if stable(n):
            hi = n
            break
        lo = n
    if hi is None:
        return done(WarmupCheck(name, declared, None, False, f"not stable within {cap} bars"))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if stable(mid):
            hi = mid
        else:
            lo = mid
    measured = hi - rows + 1
    return done(WarmupCheck(name, declared, measured, False, f"under-declared by {measured - declared}"))


def check_pipeline(pipeline: Any, *, prefix_name: str = "", **kw: Any) -> list[WarmupCheck]:
    """Measure every stateless step of a pipeline against its *effective* declared warmup."""
    from signalflow.transform.pipeline import FeaturePipeline

    checks: list[WarmupCheck] = []
    effective = pipeline.effective_warmups()
    for i, (t, eff) in enumerate(zip(pipeline.transforms, effective, strict=True)):
        label = f"{prefix_name}{t.name}[{i}]"
        if t.requires_fit:
            checks.append(
                WarmupCheck(label, eff, None, True, "stateful step: inherits its inputs' warmup, not measured")
            )
            continue
        before = pipeline.transforms[:i]
        if any(p.requires_fit for p in before):
            checks.append(WarmupCheck(label, eff, None, True, "follows a stateful step; not measured"))
            continue
        inputs = FeaturePipeline(*before) if before else None
        checks.append(measure_warmup(t, declared=eff, inputs=inputs, name=label, **kw))
    return checks


def _check_model(model: Any, prefix: str, **kw: Any) -> list[WarmupCheck]:
    features = getattr(model, "features", None)
    if features is not None and hasattr(features, "transforms"):
        return check_pipeline(features, prefix_name=prefix, **kw)
    checks: list[WarmupCheck] = []
    for i, child in enumerate(getattr(model, "children", None) or []):
        checks += _check_model(child, f"{prefix}child{i}/", **kw)
    return checks


def check_flow(flow: Any, *, signal_rows: int = 64, **kw: Any) -> list[WarmupCheck]:
    """Measure every detector (and the features it computes internally) and every model pipeline of a flow.

    Detectors that read forecast slots are not measured on their own: their need
    is the model pipeline's, which is measured. Signal columns are sparse, so a
    detector is compared over its last ``signal_rows`` rows instead of one.
    """
    checks: list[WarmupCheck] = []
    for det in flow.detectors:
        label = f"detector {det.name}"
        slots = tuple(getattr(det, "required_slots", lambda: ())())
        if slots:
            checks.append(
                WarmupCheck(
                    label,
                    int(getattr(det, "warmup", 0)),
                    None,
                    True,
                    f"reads forecast slots {list(slots)}; covered by the model pipeline",
                )
            )
            continue
        inner = getattr(det, "features", None)
        inner_list = list(inner) if isinstance(inner, (list, tuple)) else [inner] if inner is not None else []
        for f in inner_list:
            if hasattr(f, "compute") and hasattr(f, "warmup"):
                checks.append(measure_warmup(f, name=f"{label}/{getattr(f, 'name', type(f).__name__)}", **kw))
        checks.append(measure_warmup(det, name=label, rows=signal_rows, **kw))
    for slot, model in flow.forecasts.items():
        checks += _check_model(model, f"forecast {slot}/", **kw)
    if getattr(flow, "validator", None) is not None:
        checks += _check_model(flow.validator, "validator/", **kw)
    return checks


def raise_if_failed(checks: Sequence[WarmupCheck]) -> None:
    failed = [c for c in checks if not c.ok]
    if failed:
        raise WarmupError("warmup check failed:\n  " + "\n  ".join(str(c) for c in failed))


__all__ = ["WarmupCheck", "check_flow", "check_pipeline", "measure_warmup", "raise_if_failed", "synthetic_frame"]
