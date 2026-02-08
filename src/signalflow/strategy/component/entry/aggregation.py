"""Signal aggregation for combining multiple detectors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import polars as pl

from signalflow.core import SfComponentType, Signals, SignalType, sf_component


class VotingMode(str, Enum):  # noqa: UP042
    """Signal aggregation voting modes."""

    MAJORITY = "majority"  # Most common signal type wins
    WEIGHTED = "weighted"  # Weight by probability or custom weights
    UNANIMOUS = "unanimous"  # All must agree
    ANY = "any"  # Any non-NONE signal passes
    META_LABELING = "meta_labeling"  # Detector signal * validator probability


@dataclass
@sf_component(name="signal_aggregator")
class SignalAggregator:
    """Combine signals from multiple detectors.

    Aggregates multiple Signals DataFrames into one based on voting/weighting logic.

    Args:
        voting_mode: How to combine signals (see VotingMode).
        min_agreement: Minimum fraction of detectors agreeing (for MAJORITY).
        weights: Optional weights per detector (for WEIGHTED mode).
        probability_threshold: Minimum combined probability to emit signal.
        pair_col: Column name for pair.
        ts_col: Column name for timestamp.

    Example:
        >>> # Majority voting
        >>> aggregator = SignalAggregator(voting_mode=VotingMode.MAJORITY)
        >>> combined = aggregator.aggregate([signals1, signals2, signals3])

        >>> # Meta-labeling: detector direction * validator confidence
        >>> aggregator = SignalAggregator(voting_mode=VotingMode.META_LABELING)
        >>> combined = aggregator.aggregate([detector_signals, validator_signals])
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_ENTRY_RULE

    voting_mode: VotingMode = VotingMode.MAJORITY
    min_agreement: float = 0.5
    weights: list[float] | None = None
    probability_threshold: float = 0.5
    pair_col: str = "pair"
    ts_col: str = "timestamp"

    def aggregate(
        self,
        signals_list: list[Signals],
        detector_names: list[str] | None = None,
    ) -> Signals:
        """Aggregate multiple signal sources into one.

        Args:
            signals_list: List of Signals from different detectors.
            detector_names: Optional names for tracing (len must match signals_list).

        Returns:
            Aggregated Signals DataFrame.
        """
        if not signals_list:
            return Signals(pl.DataFrame())

        if len(signals_list) == 1:
            return signals_list[0]

        if self.voting_mode == VotingMode.MAJORITY:
            return self._aggregate_majority(signals_list)
        elif self.voting_mode == VotingMode.WEIGHTED:
            return self._aggregate_weighted(signals_list)
        elif self.voting_mode == VotingMode.UNANIMOUS:
            return self._aggregate_unanimous(signals_list)
        elif self.voting_mode == VotingMode.ANY:
            return self._aggregate_any(signals_list)
        elif self.voting_mode == VotingMode.META_LABELING:
            return self._aggregate_meta_labeling(signals_list)
        else:
            raise ValueError(f"Unknown voting mode: {self.voting_mode}")

    def _aggregate_majority(self, signals_list: list[Signals]) -> Signals:
        """Majority voting: most common signal type wins."""
        # Collect all non-NONE signals
        dfs = []
        for i, sig in enumerate(signals_list):
            df = sig.value.filter(
                pl.col("signal_type").is_not_null() & (pl.col("signal_type") != SignalType.NONE.value)
            )
            if df.height > 0:
                df = df.with_columns(pl.lit(i).alias("_detector_idx"))
                dfs.append(df)

        if not dfs:
            return Signals(pl.DataFrame())

        combined = pl.concat(dfs, how="vertical_relaxed")

        # Group by (pair, timestamp, signal_type) and count votes
        aggregated = combined.group_by([self.pair_col, self.ts_col, "signal_type"]).agg(
            [
                pl.len().alias("vote_count"),
                pl.col("probability").mean().alias("avg_probability"),
            ]
        )

        n_detectors = len(signals_list)
        min_votes = max(1, int(n_detectors * self.min_agreement))

        # Filter by minimum agreement and pick highest vote count per (pair, ts)
        result = (
            aggregated.filter(pl.col("vote_count") >= min_votes)
            .sort([self.pair_col, self.ts_col, "vote_count"], descending=[False, False, True])
            .group_by([self.pair_col, self.ts_col], maintain_order=True)
            .first()
            .with_columns(
                [
                    pl.lit(1).alias("signal"),
                    pl.col("avg_probability").alias("probability"),
                ]
            )
            .select([self.pair_col, self.ts_col, "signal_type", "signal", "probability"])
        )

        return Signals(result)

    def _aggregate_weighted(self, signals_list: list[Signals]) -> Signals:
        """Weighted average of probabilities, majority signal type."""
        weights = self.weights or [1.0] * len(signals_list)

        dfs = []
        for i, sig in enumerate(signals_list):
            df = sig.value.filter(
                pl.col("signal_type").is_not_null() & (pl.col("signal_type") != SignalType.NONE.value)
            )
            if df.height > 0:
                df = df.with_columns(pl.lit(weights[i]).alias("_weight"))
                dfs.append(df)

        if not dfs:
            return Signals(pl.DataFrame())

        combined = pl.concat(dfs, how="vertical_relaxed")

        # Weighted probability aggregation
        result = (
            combined.group_by([self.pair_col, self.ts_col])
            .agg(
                [
                    # Majority signal type (mode)
                    pl.col("signal_type").mode().first().alias("signal_type"),
                    # Weighted average probability
                    ((pl.col("probability") * pl.col("_weight")).sum() / pl.col("_weight").sum()).alias("probability"),
                ]
            )
            .with_columns(pl.lit(1).alias("signal"))
            .filter(pl.col("probability") >= self.probability_threshold)
            .select([self.pair_col, self.ts_col, "signal_type", "signal", "probability"])
        )

        return Signals(result)

    def _aggregate_unanimous(self, signals_list: list[Signals]) -> Signals:
        """All detectors must agree on signal type."""
        # Get non-NONE signals from each detector
        filtered = [
            sig.value.filter(pl.col("signal_type").is_not_null() & (pl.col("signal_type") != SignalType.NONE.value))
            for sig in signals_list
        ]

        # Check if any detector has no signals
        if any(df.height == 0 for df in filtered):
            return Signals(pl.DataFrame())

        # Start with first detector's signals
        base = filtered[0].select([self.pair_col, self.ts_col, "signal_type"])

        # Inner join with all other detectors on (pair, ts, signal_type)
        for df in filtered[1:]:
            base = base.join(
                df.select([self.pair_col, self.ts_col, "signal_type"]),
                on=[self.pair_col, self.ts_col, "signal_type"],
                how="inner",
            )

        if base.height == 0:
            return Signals(pl.DataFrame())

        # Average probabilities across all detectors
        prob_dfs = [
            sig.value.select([self.pair_col, self.ts_col, pl.col("probability").alias(f"_prob_{i}")])
            for i, sig in enumerate(signals_list)
        ]

        result = base
        for prob_df in prob_dfs:
            result = result.join(prob_df, on=[self.pair_col, self.ts_col], how="left")

        prob_cols = [f"_prob_{i}" for i in range(len(signals_list))]
        result = (
            result.with_columns(
                [
                    pl.mean_horizontal(prob_cols).alias("probability"),
                    pl.lit(1).alias("signal"),
                ]
            )
            .select([self.pair_col, self.ts_col, "signal_type", "signal", "probability"])
            .filter(pl.col("probability") >= self.probability_threshold)
        )

        return Signals(result)

    def _aggregate_any(self, signals_list: list[Signals]) -> Signals:
        """Any non-NONE signal passes (union with highest probability)."""
        dfs = [
            sig.value.filter(pl.col("signal_type").is_not_null() & (pl.col("signal_type") != SignalType.NONE.value))
            for sig in signals_list
        ]

        dfs = [df for df in dfs if df.height > 0]
        if not dfs:
            return Signals(pl.DataFrame())

        combined = pl.concat(dfs, how="vertical_relaxed")

        # Keep highest probability signal per (pair, ts)
        result = (
            combined.sort([self.pair_col, self.ts_col, "probability"], descending=[False, False, True])
            .group_by([self.pair_col, self.ts_col], maintain_order=True)
            .first()
            .select([self.pair_col, self.ts_col, "signal_type", "signal", "probability"])
        )

        return Signals(result)

    def _aggregate_meta_labeling(self, signals_list: list[Signals]) -> Signals:
        """Meta-labeling: detector[0] signal * detector[1] probability.

        First signal source provides direction, second provides confidence.
        Combined probability = detector_probability * validator_probability
        """
        if len(signals_list) < 2:
            return signals_list[0] if signals_list else Signals(pl.DataFrame())

        detector = signals_list[0].value.filter(
            pl.col("signal_type").is_not_null() & (pl.col("signal_type") != SignalType.NONE.value)
        )
        validator = signals_list[1].value

        if detector.height == 0:
            return Signals(pl.DataFrame())

        # Join detector with validator on (pair, timestamp)
        result = detector.join(
            validator.select(
                [
                    self.pair_col,
                    self.ts_col,
                    pl.col("probability").alias("_validator_prob"),
                ]
            ),
            on=[self.pair_col, self.ts_col],
            how="left",
        )

        # Combine probabilities
        result = (
            result.with_columns((pl.col("probability") * pl.col("_validator_prob").fill_null(1.0)).alias("probability"))
            .filter(pl.col("probability") >= self.probability_threshold)
            .select([self.pair_col, self.ts_col, "signal_type", "signal", "probability"])
        )

        return Signals(result)

    def __call__(
        self,
        signals_list: list[Signals],
        detector_names: list[str] | None = None,
    ) -> Signals:
        """Alias for aggregate()."""
        return self.aggregate(signals_list, detector_names)
