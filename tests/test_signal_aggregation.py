"""Tests for signal aggregation."""

from datetime import datetime

import polars as pl
import pytest

from signalflow.core import SignalType, Signals
from signalflow.strategy.component.entry.aggregation import SignalAggregator, VotingMode


@pytest.fixture
def signals_rise():
    """RISE signal for BTCUSDT."""
    return Signals(
        pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "signal_type": [SignalType.RISE.value],
                "signal": [1],
                "probability": [0.8],
            }
        )
    )


@pytest.fixture
def signals_fall():
    """FALL signal for BTCUSDT."""
    return Signals(
        pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "signal_type": [SignalType.FALL.value],
                "signal": [-1],
                "probability": [0.7],
            }
        )
    )


@pytest.fixture
def signals_none():
    """NONE signal for BTCUSDT."""
    return Signals(
        pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "signal_type": [SignalType.NONE.value],
                "signal": [0],
                "probability": [0.0],
            }
        )
    )


@pytest.fixture
def signals_rise_low_prob():
    """RISE signal with low probability."""
    return Signals(
        pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "signal_type": [SignalType.RISE.value],
                "signal": [1],
                "probability": [0.6],
            }
        )
    )


# ── Basic Aggregation ────────────────────────────────────────────────────────


class TestBasicAggregation:
    def test_empty_list_returns_empty(self):
        agg = SignalAggregator()
        result = agg.aggregate([])
        assert result.value.height == 0

    def test_single_source_returns_same(self, signals_rise):
        agg = SignalAggregator()
        result = agg.aggregate([signals_rise])
        assert result.value.height == 1
        assert result.value["signal_type"][0] == SignalType.RISE.value

    def test_callable_interface(self, signals_rise):
        agg = SignalAggregator()
        result = agg([signals_rise])
        assert result.value.height == 1


# ── Majority Voting ──────────────────────────────────────────────────────────


class TestMajorityVoting:
    def test_majority_wins_2_vs_1(self, signals_rise, signals_fall):
        agg = SignalAggregator(voting_mode=VotingMode.MAJORITY)
        signals_rise2 = Signals(signals_rise.value.clone())
        # 2 RISE vs 1 FALL
        result = agg.aggregate([signals_rise, signals_rise2, signals_fall])

        assert result.value.height == 1
        assert result.value["signal_type"][0] == SignalType.RISE.value

    def test_filters_below_min_agreement(self, signals_rise, signals_fall, signals_none):
        agg = SignalAggregator(voting_mode=VotingMode.MAJORITY, min_agreement=0.8)
        # Only 33% agreement (1/3 RISE, 1/3 FALL, 1/3 NONE)
        result = agg.aggregate([signals_rise, signals_fall, signals_none])

        # Neither RISE nor FALL has 80% agreement
        assert result.value.height == 0

    def test_averages_probabilities(self, signals_rise, signals_rise_low_prob):
        agg = SignalAggregator(voting_mode=VotingMode.MAJORITY)
        result = agg.aggregate([signals_rise, signals_rise_low_prob])

        assert result.value.height == 1
        # Average of 0.8 and 0.6 = 0.7
        assert result.value["probability"][0] == pytest.approx(0.7)


# ── Weighted Voting ──────────────────────────────────────────────────────────


class TestWeightedVoting:
    def test_weighted_average_probability(self, signals_rise, signals_rise_low_prob):
        agg = SignalAggregator(
            voting_mode=VotingMode.WEIGHTED,
            weights=[2.0, 1.0],
            probability_threshold=0.0,
        )
        result = agg.aggregate([signals_rise, signals_rise_low_prob])

        assert result.value.height == 1
        # Weighted avg: (0.8*2 + 0.6*1) / (2+1) = 2.2/3 = 0.733
        assert result.value["probability"][0] == pytest.approx(0.733, rel=0.01)

    def test_filters_below_probability_threshold(self, signals_rise_low_prob):
        # Need two sources to actually aggregate (single source returns as-is)
        signals_low2 = Signals(signals_rise_low_prob.value.with_columns(pl.lit(0.5).alias("probability")))
        agg = SignalAggregator(
            voting_mode=VotingMode.WEIGHTED,
            probability_threshold=0.7,
        )
        result = agg.aggregate([signals_rise_low_prob, signals_low2])
        # Weighted avg: (0.6*1 + 0.5*1) / 2 = 0.55 < 0.7 threshold
        assert result.value.height == 0


# ── Unanimous Voting ─────────────────────────────────────────────────────────


class TestUnanimousVoting:
    def test_unanimous_requires_all_agree(self, signals_rise):
        signals_rise2 = Signals(signals_rise.value.clone())
        agg = SignalAggregator(voting_mode=VotingMode.UNANIMOUS, probability_threshold=0.0)
        result = agg.aggregate([signals_rise, signals_rise2])

        assert result.value.height == 1
        assert result.value["signal_type"][0] == SignalType.RISE.value

    def test_unanimous_fails_on_disagreement(self, signals_rise, signals_fall):
        agg = SignalAggregator(voting_mode=VotingMode.UNANIMOUS)
        result = agg.aggregate([signals_rise, signals_fall])

        assert result.value.height == 0

    def test_unanimous_fails_if_one_has_none(self, signals_rise, signals_none):
        agg = SignalAggregator(voting_mode=VotingMode.UNANIMOUS)
        result = agg.aggregate([signals_rise, signals_none])

        assert result.value.height == 0


# ── Any Voting ───────────────────────────────────────────────────────────────


class TestAnyVoting:
    def test_any_passes_single_signal(self, signals_rise):
        agg = SignalAggregator(voting_mode=VotingMode.ANY)
        result = agg.aggregate([signals_rise])

        assert result.value.height == 1

    def test_any_takes_highest_probability(self, signals_rise, signals_fall):
        agg = SignalAggregator(voting_mode=VotingMode.ANY)
        result = agg.aggregate([signals_rise, signals_fall])

        # RISE has probability 0.8, FALL has 0.7
        assert result.value.height == 1
        assert result.value["probability"][0] == 0.8
        assert result.value["signal_type"][0] == SignalType.RISE.value

    def test_any_filters_none_signals(self, signals_none, signals_rise):
        agg = SignalAggregator(voting_mode=VotingMode.ANY)
        result = agg.aggregate([signals_none, signals_rise])

        assert result.value.height == 1
        assert result.value["signal_type"][0] == SignalType.RISE.value


# ── Meta-Labeling ────────────────────────────────────────────────────────────


class TestMetaLabeling:
    def test_meta_labeling_combines_probabilities(self, signals_rise):
        validator = Signals(
            pl.DataFrame(
                {
                    "pair": ["BTCUSDT"],
                    "timestamp": [datetime(2024, 1, 1)],
                    "signal_type": [SignalType.RISE.value],
                    "signal": [1],
                    "probability": [0.9],
                }
            )
        )

        agg = SignalAggregator(
            voting_mode=VotingMode.META_LABELING,
            probability_threshold=0.5,
        )
        result = agg.aggregate([signals_rise, validator])

        assert result.value.height == 1
        # Combined prob = 0.8 * 0.9 = 0.72
        assert result.value["probability"][0] == pytest.approx(0.72)

    def test_meta_labeling_uses_detector_direction(self, signals_rise):
        # Validator has different signal type - doesn't matter for direction
        validator = Signals(
            pl.DataFrame(
                {
                    "pair": ["BTCUSDT"],
                    "timestamp": [datetime(2024, 1, 1)],
                    "signal_type": [SignalType.NONE.value],
                    "signal": [0],
                    "probability": [0.9],
                }
            )
        )

        agg = SignalAggregator(
            voting_mode=VotingMode.META_LABELING,
            probability_threshold=0.5,
        )
        result = agg.aggregate([signals_rise, validator])

        assert result.value.height == 1
        # Uses detector's signal type (RISE)
        assert result.value["signal_type"][0] == SignalType.RISE.value

    def test_meta_labeling_filters_low_probability(self, signals_rise):
        validator = Signals(
            pl.DataFrame(
                {
                    "pair": ["BTCUSDT"],
                    "timestamp": [datetime(2024, 1, 1)],
                    "signal_type": [SignalType.RISE.value],
                    "signal": [1],
                    "probability": [0.3],  # Low confidence
                }
            )
        )

        agg = SignalAggregator(
            voting_mode=VotingMode.META_LABELING,
            probability_threshold=0.5,
        )
        result = agg.aggregate([signals_rise, validator])

        # Combined prob = 0.8 * 0.3 = 0.24 < 0.5 threshold
        assert result.value.height == 0

    def test_meta_labeling_fills_missing_validator(self, signals_rise):
        # Validator has no matching (pair, timestamp)
        validator = Signals(
            pl.DataFrame(
                {
                    "pair": ["ETHUSDT"],  # Different pair
                    "timestamp": [datetime(2024, 1, 1)],
                    "signal_type": [SignalType.RISE.value],
                    "signal": [1],
                    "probability": [0.9],
                }
            )
        )

        agg = SignalAggregator(
            voting_mode=VotingMode.META_LABELING,
            probability_threshold=0.5,
        )
        result = agg.aggregate([signals_rise, validator])

        # Missing validator prob filled with 1.0
        assert result.value.height == 1
        assert result.value["probability"][0] == pytest.approx(0.8)


# ── Multiple Pairs ───────────────────────────────────────────────────────────


class TestMultiplePairs:
    def test_aggregates_multiple_pairs(self):
        signals1 = Signals(
            pl.DataFrame(
                {
                    "pair": ["BTCUSDT", "ETHUSDT"],
                    "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
                    "signal_type": [SignalType.RISE.value, SignalType.FALL.value],
                    "signal": [1, -1],
                    "probability": [0.8, 0.7],
                }
            )
        )

        agg = SignalAggregator(voting_mode=VotingMode.ANY)
        result = agg.aggregate([signals1])

        assert result.value.height == 2

    def test_majority_per_pair_timestamp(self):
        ts = datetime(2024, 1, 1)

        # Detector 1: RISE for BTC, FALL for ETH
        signals1 = Signals(
            pl.DataFrame(
                {
                    "pair": ["BTCUSDT", "ETHUSDT"],
                    "timestamp": [ts, ts],
                    "signal_type": [SignalType.RISE.value, SignalType.FALL.value],
                    "signal": [1, -1],
                    "probability": [0.8, 0.7],
                }
            )
        )

        # Detector 2: RISE for both
        signals2 = Signals(
            pl.DataFrame(
                {
                    "pair": ["BTCUSDT", "ETHUSDT"],
                    "timestamp": [ts, ts],
                    "signal_type": [SignalType.RISE.value, SignalType.RISE.value],
                    "signal": [1, 1],
                    "probability": [0.7, 0.6],
                }
            )
        )

        agg = SignalAggregator(voting_mode=VotingMode.MAJORITY, min_agreement=0.5)
        result = agg.aggregate([signals1, signals2])

        # BTC: 2 RISE (100% agreement)
        # ETH: 1 FALL, 1 RISE (50% each)
        assert result.value.height >= 1

        btc_row = result.value.filter(pl.col("pair") == "BTCUSDT")
        assert btc_row["signal_type"][0] == SignalType.RISE.value
