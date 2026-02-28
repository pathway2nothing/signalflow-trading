"""Signal reconciliation for multi-detector strategies.

This module implements the RECONCILER from SFFLOW.md specification:
- any: Any signal triggers (union)
- all: All sources must agree (intersection)
- weighted: Weighted combination by source
- voting: Majority voting
- model: ML model decides

Example:
    >>> from signalflow.strategy.reconciler import Reconciler, ReconcileMode
    >>> reconciler = Reconciler(mode=ReconcileMode.WEIGHTED, weights={"trend": 0.6, "momentum": 0.4})
    >>> merged_signals = reconciler.reconcile(signals_dict)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger

if TYPE_CHECKING:
    from signalflow.core import Signals


class ReconcileMode(StrEnum):
    """Signal reconciliation modes."""

    ANY = "any"  # Any signal triggers (union)
    ALL = "all"  # All sources must agree (intersection)
    WEIGHTED = "weighted"  # Weighted combination by source
    VOTING = "voting"  # Majority voting
    MODEL = "model"  # ML model decides


@dataclass
class ReconcileConfig:
    """Reconciler configuration.

    Attributes:
        mode: Reconciliation mode
        weights: Source weights for weighted mode (source_id -> weight)
        threshold: Minimum threshold for weighted/voting (default 0.5)
        model_type: Model type for model mode
        model_path: Path to trained model for model mode
    """

    mode: ReconcileMode = ReconcileMode.ANY
    weights: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5
    model_type: str | None = None
    model_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReconcileConfig:
        """Create from dict."""
        mode_str = data.get("mode", "any")
        return cls(
            mode=ReconcileMode(mode_str),
            weights=data.get("weights", {}),
            threshold=data.get("threshold", 0.5),
            model_type=data.get("model_type"),
            model_path=data.get("model_path"),
        )


class BaseReconciler(ABC):
    """Base class for signal reconcilers."""

    @abstractmethod
    def reconcile(
        self,
        signals: dict[str, pl.DataFrame],
        features: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """Reconcile signals from multiple sources.

        Args:
            signals: Dict of source_id -> signals DataFrame
            features: Optional dict of source_id -> features DataFrame

        Returns:
            Merged signals DataFrame
        """
        ...


class AnyReconciler(BaseReconciler):
    """Any signal triggers (union).

    All signals from all sources are passed through.
    Signals are tagged with their source_id.
    """

    def reconcile(
        self,
        signals: dict[str, pl.DataFrame],
        features: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """Union all signals."""
        if not signals:
            return pl.DataFrame()

        dfs = []
        for source_id, df in signals.items():
            if df is None or df.is_empty():
                continue

            # Ensure source_id column
            if "source_id" not in df.columns:
                df = df.with_columns(pl.lit(source_id).alias("source_id"))

            dfs.append(df)

        if not dfs:
            return pl.DataFrame()

        return pl.concat(dfs, how="diagonal").sort("timestamp")


class AllReconciler(BaseReconciler):
    """All sources must agree (intersection).

    Only signals where all sources agree on direction at the same time.
    Grouped by timestamp and pair.
    """

    def __init__(self, sources: list[str] | None = None):
        """Initialize with expected sources.

        Args:
            sources: List of expected source IDs
        """
        self.sources = sources

    def reconcile(
        self,
        signals: dict[str, pl.DataFrame],
        features: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """Intersect signals where all sources agree."""
        if not signals:
            return pl.DataFrame()

        sources = self.sources or list(signals.keys())
        if len(sources) < 2:
            # Single source, just return it
            for df in signals.values():
                return df if df is not None else pl.DataFrame()
            return pl.DataFrame()

        # Prepare DataFrames with source prefix
        dfs = []
        for source_id in sources:
            source_df = signals.get(source_id)
            if source_df is None or source_df.is_empty():
                logger.warning(f"Source {source_id} has no signals, ALL reconciliation will yield empty")
                return pl.DataFrame()

            # Select key columns with prefix
            prefixed_df = source_df.select([
                pl.col("timestamp"),
                pl.col("pair"),
                pl.col("direction").alias(f"direction_{source_id}"),
            ])
            dfs.append(prefixed_df)

        # Join all sources on timestamp and pair
        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, on=["timestamp", "pair"], how="inner")

        # Filter where all directions match
        direction_cols = [f"direction_{s}" for s in sources]
        first_dir = direction_cols[0]

        for col in direction_cols[1:]:
            result = result.filter(pl.col(first_dir) == pl.col(col))

        # Clean up and return
        return result.select([
            pl.col("timestamp"),
            pl.col("pair"),
            pl.col(first_dir).alias("direction"),
            pl.lit("all").alias("source_id"),
        ]).sort("timestamp")


class WeightedReconciler(BaseReconciler):
    """Weighted combination of signals.

    Each source has a weight. Signals are combined by computing
    a weighted score and thresholding.
    """

    def __init__(
        self,
        weights: dict[str, float],
        threshold: float = 0.5,
    ):
        """Initialize with source weights.

        Args:
            weights: Dict of source_id -> weight
            threshold: Minimum score to trigger signal
        """
        self.weights = weights
        self.threshold = threshold

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in weights.items()}

    def reconcile(
        self,
        signals: dict[str, pl.DataFrame],
        features: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """Compute weighted signal scores."""
        if not signals:
            return pl.DataFrame()

        # Collect all unique (timestamp, pair) combinations
        all_points: set[tuple[Any, str]] = set()
        for df in signals.values():
            if df is not None and not df.is_empty():
                for row in df.select(["timestamp", "pair"]).iter_rows():
                    all_points.add((row[0], row[1]))

        if not all_points:
            return pl.DataFrame()

        # Build result
        results = []
        for timestamp, pair in all_points:
            long_score = 0.0
            short_score = 0.0

            for source_id, df in signals.items():
                weight = self.weights.get(source_id, 0.0)
                if weight == 0 or df is None or df.is_empty():
                    continue

                # Find signal at this point
                match = df.filter(
                    (pl.col("timestamp") == timestamp) & (pl.col("pair") == pair)
                )

                if match.is_empty():
                    continue

                direction = match["direction"][0]
                if direction == "long":
                    long_score += weight
                elif direction == "short":
                    short_score += weight

            # Determine final direction based on scores
            if long_score >= self.threshold and long_score > short_score:
                results.append({
                    "timestamp": timestamp,
                    "pair": pair,
                    "direction": "long",
                    "score": long_score,
                    "source_id": "weighted",
                })
            elif short_score >= self.threshold and short_score > long_score:
                results.append({
                    "timestamp": timestamp,
                    "pair": pair,
                    "direction": "short",
                    "score": short_score,
                    "source_id": "weighted",
                })

        if not results:
            return pl.DataFrame()

        return pl.DataFrame(results).sort("timestamp")


class VotingReconciler(BaseReconciler):
    """Majority voting reconciler.

    Each source gets one vote. Majority direction wins.
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize with voting threshold.

        Args:
            threshold: Minimum fraction of votes to trigger signal
        """
        self.threshold = threshold

    def reconcile(
        self,
        signals: dict[str, pl.DataFrame],
        features: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """Compute majority vote."""
        if not signals:
            return pl.DataFrame()

        n_sources = len(signals)
        if n_sources == 0:
            return pl.DataFrame()

        # Collect all unique (timestamp, pair) combinations
        all_points: set[tuple[Any, str]] = set()
        for df in signals.values():
            if df is not None and not df.is_empty():
                for row in df.select(["timestamp", "pair"]).iter_rows():
                    all_points.add((row[0], row[1]))

        if not all_points:
            return pl.DataFrame()

        # Build result
        results = []
        for timestamp, pair in all_points:
            votes: dict[str, int] = {"long": 0, "short": 0, "close": 0}

            for df in signals.values():
                if df is None or df.is_empty():
                    continue

                match = df.filter(
                    (pl.col("timestamp") == timestamp) & (pl.col("pair") == pair)
                )

                if match.is_empty():
                    continue

                direction = match["direction"][0]
                if direction in votes:
                    votes[direction] += 1

            # Find winner
            max_votes = max(votes.values())
            if max_votes == 0:
                continue

            vote_ratio = max_votes / n_sources
            if vote_ratio < self.threshold:
                continue

            # Get winning direction
            winner = max(votes.keys(), key=lambda k: votes[k])
            if winner == "close":
                continue  # Skip close signals in reconciliation

            results.append({
                "timestamp": timestamp,
                "pair": pair,
                "direction": winner,
                "votes": max_votes,
                "vote_ratio": vote_ratio,
                "source_id": "voting",
            })

        if not results:
            return pl.DataFrame()

        return pl.DataFrame(results).sort("timestamp")


class ModelReconciler(BaseReconciler):
    """ML model-based reconciler.

    Uses a trained model to decide signal direction based on
    features from all sources.
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        model_path: str | None = None,
    ):
        """Initialize with model configuration.

        Args:
            model_type: Type of model (lightgbm, xgboost, neural)
            model_path: Path to trained model
        """
        self.model_type = model_type
        self.model_path = model_path
        self._model: Any = None

    def load_model(self) -> None:
        """Load the trained model."""
        if self.model_path is None:
            raise ValueError("model_path required for ModelReconciler")

        if self.model_type == "lightgbm":
            import lightgbm as lgb

            self._model = lgb.Booster(model_file=self.model_path)
        elif self.model_type == "xgboost":
            import xgboost as xgb

            self._model = xgb.Booster()
            self._model.load_model(self.model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def reconcile(
        self,
        signals: dict[str, pl.DataFrame],
        features: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """Use model to predict reconciled signals."""
        if self._model is None:
            self.load_model()

        if not signals or features is None:
            return pl.DataFrame()

        # Build feature matrix from all sources
        # This is a simplified implementation - in practice you'd need
        # to align timestamps and build proper feature vectors

        # For now, fall back to voting if model fails
        logger.warning("ModelReconciler falling back to voting (not fully implemented)")
        return VotingReconciler().reconcile(signals, features)


class Reconciler:
    """Signal reconciler factory and unified interface.

    Example:
        >>> reconciler = Reconciler.from_config({"mode": "weighted", "weights": {"a": 0.6, "b": 0.4}})
        >>> merged = reconciler.reconcile({"a": signals_a, "b": signals_b})
    """

    def __init__(
        self,
        mode: ReconcileMode = ReconcileMode.ANY,
        weights: dict[str, float] | None = None,
        threshold: float = 0.5,
        model_type: str | None = None,
        model_path: str | None = None,
    ):
        """Initialize reconciler.

        Args:
            mode: Reconciliation mode
            weights: Source weights for weighted mode
            threshold: Score threshold for weighted/voting
            model_type: Model type for model mode
            model_path: Path to model for model mode
        """
        self.mode = mode
        self.weights = weights or {}
        self.threshold = threshold
        self.model_type = model_type
        self.model_path = model_path

        # Create underlying reconciler
        self._reconciler = self._create_reconciler()

    def _create_reconciler(self) -> BaseReconciler:
        """Create the appropriate reconciler."""
        if self.mode == ReconcileMode.ANY:
            return AnyReconciler()
        elif self.mode == ReconcileMode.ALL:
            return AllReconciler(list(self.weights.keys()) if self.weights else None)
        elif self.mode == ReconcileMode.WEIGHTED:
            return WeightedReconciler(self.weights, self.threshold)
        elif self.mode == ReconcileMode.VOTING:
            return VotingReconciler(self.threshold)
        elif self.mode == ReconcileMode.MODEL:
            return ModelReconciler(self.model_type or "lightgbm", self.model_path)
        else:
            raise ValueError(f"Unknown reconcile mode: {self.mode}")

    @classmethod
    def from_config(cls, config: dict[str, Any] | ReconcileConfig) -> Reconciler:
        """Create from configuration.

        Args:
            config: ReconcileConfig or dict with mode, weights, etc.

        Returns:
            Configured Reconciler
        """
        if isinstance(config, ReconcileConfig):
            return cls(
                mode=config.mode,
                weights=config.weights,
                threshold=config.threshold,
                model_type=config.model_type,
                model_path=config.model_path,
            )

        cfg = ReconcileConfig.from_dict(config)
        return cls.from_config(cfg)

    def reconcile(
        self,
        signals: dict[str, pl.DataFrame | Signals],
        features: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """Reconcile signals from multiple sources.

        Args:
            signals: Dict of source_id -> signals (DataFrame or Signals object)
            features: Optional dict of source_id -> features DataFrame

        Returns:
            Merged signals DataFrame
        """
        # Convert Signals objects to DataFrames
        signal_dfs: dict[str, pl.DataFrame] = {}
        for source_id, sig in signals.items():
            if sig is None:
                continue
            if hasattr(sig, "value"):
                signal_dfs[source_id] = sig.value
            elif isinstance(sig, pl.DataFrame):
                signal_dfs[source_id] = sig
            else:
                logger.warning(f"Unknown signal type from {source_id}: {type(sig)}")

        return self._reconciler.reconcile(signal_dfs, features)
