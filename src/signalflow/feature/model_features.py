"""ModelFeaturesPipeline — reproducibility layer over the compute engine.

``FeaturePipeline`` is the *computational engine*: it knows how to turn a
DataFrame into feature columns. It does not know how to be rebuilt from a
config, how to detect that the recipe drifted between train and serve, or how
to prove that its recursive features survive a warmup truncation.

``ModelFeaturesPipeline`` adds exactly those reproducibility concerns and
nothing else. It is a **composition** over a ``FeaturePipeline`` (the one and
only compute engine) plus a ``FeatureSpec`` (the serializable recipe). There is
**zero** duplicated feature computation here: :meth:`compute` delegates straight
into the wrapped pipeline.

See VISION.md §5.2 for the design intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import polars as pl

from signalflow.feature.feature_pipeline import FeaturePipeline
from signalflow.feature.spec import FeatureSpec


@dataclass
class ModelFeaturesPipeline:
    """Reproducibility wrapper around a :class:`FeaturePipeline`.

    Holds both the live compute engine (``_pipeline``) and its serializable,
    hashable recipe (``_spec``). All feature computation is delegated to the
    nested pipeline — this class only adds reconstruction, hash verification and
    warmup-reproducibility checks.

    Attributes:
        _pipeline: The wrapped computational engine (single source of compute).
        _spec: The serializable recipe used for config round-trip and hashing.
    """

    _pipeline: FeaturePipeline
    _spec: FeatureSpec

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, data: dict) -> Self:
        """Reconstruct from a plain config dict.

        Builds the recipe (:class:`FeatureSpec`) and uses it to instantiate the
        compute engine. The spec is the single source of truth for both the
        engine and the hash, so they can never disagree.
        """
        spec = FeatureSpec.from_config(data)
        pipeline = spec.build()
        return cls(_pipeline=pipeline, _spec=spec)

    @classmethod
    def from_pipeline(
        cls,
        pipeline: FeaturePipeline,
        *,
        ta_version: str | None = None,
    ) -> Self:
        """Wrap an already-built pipeline, deriving its recipe.

        The recipe is extracted from the live pipeline via
        :meth:`FeatureSpec.from_pipeline`; ``ta_version`` pins the TA
        implementation in the hash when provided.
        """
        spec = FeatureSpec.from_pipeline(pipeline, ta_version=ta_version)
        return cls(_pipeline=pipeline, _spec=spec)

    # ── Serialization ────────────────────────────────────────────────────

    def to_config(self) -> dict:
        """Serialize the recipe to a plain dict (delegates to the spec)."""
        return self._spec.to_config()

    def to_artifact_dict(self) -> dict:
        """Bundle the recipe with its hash for storing alongside a model.

        The persisted ``feature_hash`` is what :meth:`verify_hash` checks at
        load time, turning silent recipe drift into a loud failure.
        """
        return {
            "features_config": self._spec.to_config(),
            "feature_hash": self.feature_hash,
        }

    @property
    def feature_hash(self) -> str:
        """Stable SHA-256 of the recipe (delegates to the spec)."""
        return self._spec.feature_hash()

    # ── Reproducibility guarantees ───────────────────────────────────────

    def validate_reproducible(self) -> None:
        """Assert every nested feature honours the warmup-invariance contract.

        Delegates to :meth:`FeaturePipeline.assert_reproducible`, which raises if
        any nested feature is recursive and not warmup-invariant (entry-point
        dependent → would break live/backtest parity).

        Raises:
            RuntimeError: if a recursive non-invariant feature is present.
        """
        self._pipeline.assert_reproducible()

    def verify_hash(self, expected: str) -> None:
        """Verify the recipe hash matches an expected value (drift detector).

        Recompute :attr:`feature_hash` and compare it to the hash recorded when
        the artifact was produced. A mismatch means the feature recipe changed
        between train and serve — never continue silently.

        Raises:
            RuntimeError: if the recomputed hash differs from ``expected``.
        """
        actual = self.feature_hash
        if actual != expected:
            raise RuntimeError(
                "Feature hash mismatch — the feature recipe drifted between "
                "train and serve. Refusing to continue.\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )

    # ── Computation (pure delegation, zero duplication) ──────────────────

    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Compute all features by delegating to the wrapped engine.

        No feature math lives here — this is a thin pass-through to
        :meth:`FeaturePipeline.compute`, which is the single computational
        engine for the whole system.
        """
        return self._pipeline.compute(df, context)
