"""Warmup reproducibility contract (Stage 3 refactor).

Turns the verbal agreement "recursive indicators must use SMA-init" into a
machine-checkable contract on the Feature base class:

- ``is_recursive`` / ``warmup_invariant`` declarations,
- ``Feature.assert_reproducible()`` raising for recursive non-invariant features,
- ``FeaturePipeline.assert_reproducible()`` aggregating offenders,
- empirical entry-point invariance tests on the real sf-ta indicators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl
import pytest

from signalflow.feature.base import Feature
from signalflow.feature.feature_pipeline import FeaturePipeline

# --------------------------------------------------------------------------- #
# Synthetic features exercising the contract API without sf-ta.
# --------------------------------------------------------------------------- #


@dataclass
class _NonRecursiveFeat(Feature):
    """Pure windowed feature (default flags)."""

    period: int = 10
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["nr_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("close").rolling_mean(self.period).alias(f"nr_{self.period}"))


@dataclass
class _InvariantRecursiveFeat(Feature):
    """Recursive but SMA-seeded → invariant."""

    period: int = 10
    is_recursive: ClassVar[bool] = True
    warmup_invariant: ClassVar[bool] = True
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["ir_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("close").alias(f"ir_{self.period}"))


@dataclass
class _NonInvariantRecursiveFeat(Feature):
    """Recursive, not SMA-seeded → NOT invariant (breaks parity)."""

    period: int = 10
    is_recursive: ClassVar[bool] = True
    warmup_invariant: ClassVar[bool] = False
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["nir_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("close").alias(f"nir_{self.period}"))


# --------------------------------------------------------------------------- #
# Contract declarations & assert_reproducible (no sf-ta required)
# --------------------------------------------------------------------------- #


def test_base_defaults_are_safe():
    """Plain Feature defaults to non-recursive / invariant and never raises."""
    f = Feature()
    assert f.is_recursive is False
    assert f.warmup_invariant is True
    f.assert_reproducible()  # must not raise


def test_classvars_do_not_become_init_fields():
    """is_recursive / warmup_invariant are ClassVar — not dataclass init params."""
    f = _InvariantRecursiveFeat(period=5)
    assert f.period == 5
    assert "is_recursive" not in f.__dict__
    assert "warmup_invariant" not in f.__dict__


def test_non_recursive_does_not_raise():
    _NonRecursiveFeat().assert_reproducible()


def test_invariant_recursive_does_not_raise():
    _InvariantRecursiveFeat().assert_reproducible()


def test_non_invariant_recursive_raises():
    with pytest.raises(RuntimeError, match="recursive"):
        _NonInvariantRecursiveFeat().assert_reproducible()


def test_non_invariant_recursive_error_mentions_parity():
    with pytest.raises(RuntimeError, match="parity"):
        _NonInvariantRecursiveFeat().assert_reproducible()


# --------------------------------------------------------------------------- #
# FeaturePipeline.assert_reproducible aggregation
# --------------------------------------------------------------------------- #


def test_pipeline_clean_does_not_raise():
    pipe = FeaturePipeline(
        features=[_NonRecursiveFeat(), _InvariantRecursiveFeat()],
        raw_data_type="spot",
    )
    pipe.assert_reproducible()


def test_pipeline_with_offender_raises_with_name():
    pipe = FeaturePipeline(
        features=[_NonRecursiveFeat(), _NonInvariantRecursiveFeat()],
        raw_data_type="spot",
    )
    with pytest.raises(RuntimeError) as exc:
        pipe.assert_reproducible()
    assert "_NonInvariantRecursiveFeat" in str(exc.value)


def test_pipeline_aggregates_multiple_offenders():
    @dataclass
    class _SecondOffender(_NonInvariantRecursiveFeat):
        outputs: ClassVar[list[str]] = ["nir2_{period}"]

        def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(pl.col("close").alias(f"nir2_{self.period}"))

    pipe = FeaturePipeline(
        features=[_NonInvariantRecursiveFeat(), _SecondOffender()],
        raw_data_type="spot",
    )
    with pytest.raises(RuntimeError) as exc:
        pipe.assert_reproducible()
    msg = str(exc.value)
    assert "_NonInvariantRecursiveFeat" in msg
    assert "_SecondOffender" in msg


# --------------------------------------------------------------------------- #
# Empirical entry-point invariance on real sf-ta indicators.
# Skipped gracefully if signalflow.ta is not installed.
# --------------------------------------------------------------------------- #

ta = pytest.importorskip("signalflow.ta", reason="signalflow-ta not installed")


def _make_ohlc(n: int = 4000, seed: int = 13) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    return pl.DataFrame({"close": close, "high": high, "low": low})


def _output_col(df_before: pl.DataFrame, df_after: pl.DataFrame) -> str:
    new = [c for c in df_after.columns if c not in df_before.columns]
    assert len(new) == 1, new
    return new[0]


def _convergence_diff(feat: Feature, cut: int = 1000, n: int = 4000) -> float:
    """Max *relative* diff on the convergent tail (after warmup) between the value
    computed on the full series and on the series truncated `cut` bars on the left.

    Relative (scale-free) is the correct metric here: the test data is a random
    walk whose level drifts, so an absolute diff would be dominated by price scale,
    not by entry-point sensitivity.
    """
    df = _make_ohlc(n=n)
    full = feat.compute_pair(df)
    out_col = _output_col(df, full)
    full_vals = full[out_col].to_numpy()

    trimmed = feat.compute_pair(df.slice(cut, n - cut))
    tr_vals = trimmed[out_col].to_numpy()

    aligned_full = full_vals[cut:]  # full index cut+i aligns with trimmed index i
    w = feat.warmup
    a = aligned_full[w:]
    b = tr_vals[w:]
    mask = ~(np.isnan(a) | np.isnan(b))
    assert mask.any(), "no overlapping valid values after warmup"
    rel = np.abs(a[mask] - b[mask]) / (np.abs(a[mask]) + 1e-9)
    return float(np.max(rel))


def _seed_point_diff(feat: Feature, cut: int = 1000, n: int = 4000) -> float:
    """Max abs diff measured BEFORE warmup elapses (near the truncated seed point),
    where a non-invariant recursive indicator must visibly diverge.
    """
    df = _make_ohlc(n=n)
    full = feat.compute_pair(df)
    out_col = _output_col(df, full)
    full_vals = full[out_col].to_numpy()

    trimmed = feat.compute_pair(df.slice(cut, n - cut))
    tr_vals = trimmed[out_col].to_numpy()

    aligned_full = full_vals[cut:]
    first_valid = int(np.argmax(~np.isnan(tr_vals)))
    w = max(feat.warmup, first_valid + 1)
    a = aligned_full[first_valid:w]
    b = tr_vals[first_valid:w]
    mask = ~(np.isnan(a) | np.isnan(b))
    assert mask.any()
    return float(np.max(np.abs(a[mask] - b[mask])))


# Invariant recursive indicators: relative diff must converge below tolerance
# after their declared warmup (tolerances are RELATIVE / scale-free).
_INVARIANT_CASES = [
    ("RsiMom", lambda: ta.momentum.core.RsiMom(period=14), 1e-3),
    ("AtrVol", lambda: ta.volatility.range.AtrVol(period=14, ma_type="rma"), 1e-2),
    ("NatrVol", lambda: ta.volatility.range.NatrVol(period=14, ma_type="rma"), 1e-2),
    ("AtrPercentVol", lambda: ta.volatility.measures.AtrPercentVol(period=14), 1e-2),
    ("KamaSmooth", lambda: ta.overlap.adaptive.KamaSmooth(period=10), 1e-3),
    ("JmaSmooth", lambda: ta.overlap.adaptive.JmaSmooth(period=7), 1e-2),
    ("VidyaSmooth", lambda: ta.overlap.adaptive.VidyaSmooth(period=14), 1e-2),
    ("T3Smooth", lambda: ta.overlap.adaptive.T3Smooth(period=10), 1e-3),
    ("RviVol", lambda: ta.volatility.measures.RviVol(period=14, std_period=10), 1e-2),
]


@pytest.mark.parametrize("name,factory,tol", _INVARIANT_CASES, ids=[c[0] for c in _INVARIANT_CASES])
def test_declared_invariant_indicators_converge_after_warmup(name, factory, tol):
    feat = factory()
    assert feat.is_recursive is True
    assert feat.warmup_invariant is True
    feat.assert_reproducible()  # contract says: must not raise
    diff = _convergence_diff(feat)
    assert diff < tol, f"{name}: relative diff after warmup {diff:.3e} >= tol {tol:.0e}"


# Non-invariant recursive indicators: their declaration must be honest —
# there IS an entry-point-dependent divergence before warmup converges.
_NON_INVARIANT_CASES = [
    ("EmaSmooth", lambda: ta.overlap.smoothers.EmaSmooth(period=20)),
    ("RmaSmooth", lambda: ta.overlap.smoothers.RmaSmooth(period=14)),
    ("DemaSmooth", lambda: ta.overlap.smoothers.DemaSmooth(period=20)),
    ("TemaSmooth", lambda: ta.overlap.smoothers.TemaSmooth(period=20)),
    ("McGinleySmooth", lambda: ta.overlap.adaptive.McGinleySmooth(period=10)),
    ("FramaSmooth", lambda: ta.overlap.adaptive.FramaSmooth(period=16)),
    ("MassIndexVol", lambda: ta.volatility.measures.MassIndexVol(fast=9, slow=25)),
]


@pytest.mark.parametrize("name,factory", _NON_INVARIANT_CASES, ids=[c[0] for c in _NON_INVARIANT_CASES])
def test_declared_non_invariant_indicators_diverge_at_seed(name, factory):
    feat = factory()
    assert feat.is_recursive is True
    assert feat.warmup_invariant is False
    # assert_reproducible must flag it (declaration is not a lie).
    with pytest.raises(RuntimeError):
        feat.assert_reproducible()
    # And the divergence must be empirically real before warmup converges,
    # otherwise warmup_invariant=False would be a false (overly pessimistic) label.
    seed_diff = _seed_point_diff(feat)
    assert seed_diff > 1e-3, (
        f"{name}: declared non-invariant but seed-point diff is only {seed_diff:.3e}; "
        f"the declaration would be misleading"
    )
