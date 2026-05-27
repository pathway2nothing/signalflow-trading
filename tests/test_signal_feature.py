"""Tests for SignalFeature base class and infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, ClassVar

import polars as pl
import pytest

from signalflow.core.enums import SfComponentType
from signalflow.signal_feature.base import SignalFeature

# =============================================================================
# Test fixtures
# =============================================================================


def _make_signals(n: int = 20, pairs: int = 1) -> pl.DataFrame:
    """Build a synthetic signals DataFrame."""
    base = datetime(2024, 1, 1)
    rows: list[dict[str, Any]] = []
    for p in range(pairs):
        pair = f"PAIR{p}"
        for i in range(n):
            rows.append(
                {
                    "pair": pair,
                    "timestamp": base + timedelta(hours=i),
                    "signal_type": "rise" if i % 3 != 0 else "fall",
                    "signal": 1 if i % 3 != 0 else -1,
                    "probability": 0.5 + (i % 5) * 0.1,
                }
            )
    return pl.DataFrame(rows)


def _make_labels(
    signals: pl.DataFrame,
    *,
    accuracy: float = 0.7,
    with_t_hit: bool = True,
    hit_delay_hours: int = 4,
) -> pl.DataFrame:
    """Build synthetic labels aligned to signals.

    Labels ~accuracy fraction of signals as correct (matching signal_type).
    """
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(signals.to_dicts()):
        correct = (i % 10) < int(accuracy * 10)
        label = row["signal_type"] if correct else ("fall" if row["signal_type"] == "rise" else "rise")

        entry: dict[str, Any] = {
            "pair": row["pair"],
            "timestamp": row["timestamp"],
            "label": label,
        }
        if with_t_hit:
            entry["t_hit"] = row["timestamp"] + timedelta(hours=hit_delay_hours)
        rows.append(entry)
    return pl.DataFrame(rows)


# =============================================================================
# Concrete test implementations
# =============================================================================


@dataclass
class DummyUnsupervised(SignalFeature):
    """Counts signals per pair up to each row (cumulative)."""

    requires_labels: ClassVar[bool] = False
    outputs: ClassVar[list[str]] = ["signal_count"]

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        return (
            signals.sort([self.group_col, self.ts_col])
            .with_columns(
                pl.col(self.ts_col).cum_count().over(self.group_col).alias("signal_count"),
            )
            .select([self.group_col, self.ts_col, "signal_count"])
        )


@dataclass
class DummySupervised(SignalFeature):
    """Rolling accuracy over last ``window`` resolved signals."""

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["rolling_acc_{window}"]

    window: int = 5

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        assert labels is not None
        merged = self.prepare_labels(signals, labels)
        merged = self.mask_unresolved(merged)

        # "hit" = signal_type matches label (both are non-null)
        merged = merged.sort([self.group_col, self.ts_col]).with_columns(
            pl.when(pl.col("label").is_not_null())
            .then(
                (pl.col("signal_type") == pl.col("label")).cast(pl.Int32),
            )
            .otherwise(pl.lit(None, dtype=pl.Int32))
            .alias("_hit"),
        )

        col_name = f"rolling_acc_{self.window}"
        merged = merged.with_columns(
            pl.col("_hit").rolling_mean(window_size=self.window, min_samples=1).over(self.group_col).alias(col_name),
        )

        return merged.select([self.group_col, self.ts_col, col_name])

    @property
    def warmup(self) -> int:
        return self.window


# =============================================================================
# Tests — base class behaviour
# =============================================================================


class TestSignalFeatureBase:
    def test_component_type(self) -> None:
        feat = DummyUnsupervised()
        assert feat.component_type == SfComponentType.SIGNAL_FEATURE

    def test_output_cols_plain(self) -> None:
        feat = DummyUnsupervised()
        assert feat.output_cols() == ["signal_count"]

    def test_output_cols_template(self) -> None:
        feat = DummySupervised(window=10)
        assert feat.output_cols() == ["rolling_acc_10"]

    def test_warmup_default(self) -> None:
        assert DummyUnsupervised().warmup == 0

    def test_warmup_override(self) -> None:
        feat = DummySupervised(window=20)
        assert feat.warmup == 20


class TestUnsupervisedSignalFeature:
    def test_compute_basic(self) -> None:
        signals = _make_signals(10, pairs=1)
        feat = DummyUnsupervised()
        result = feat(signals)

        assert isinstance(result, pl.DataFrame)
        assert "signal_count" in result.columns
        assert "pair" in result.columns
        assert "timestamp" in result.columns
        # Should have same number of rows
        assert result.height == signals.height

    def test_compute_multi_pair(self) -> None:
        signals = _make_signals(10, pairs=3)
        feat = DummyUnsupervised()
        result = feat(signals)

        assert result.height == signals.height
        # Each pair should have its own count sequence
        for pair in ["PAIR0", "PAIR1", "PAIR2"]:
            subset = result.filter(pl.col("pair") == pair)
            counts = subset.sort("timestamp")["signal_count"].to_list()
            assert counts == list(range(1, 11))

    def test_labels_not_required(self) -> None:
        """Unsupervised feature should not complain about None labels."""
        signals = _make_signals(5)
        feat = DummyUnsupervised()
        result = feat(signals, labels=None)
        assert result.height == 5

    def test_validation_missing_columns(self) -> None:
        bad_df = pl.DataFrame({"pair": ["A"], "timestamp": [datetime.now()]})
        feat = DummyUnsupervised()
        with pytest.raises(ValueError, match="signal_type"):
            feat(bad_df)


class TestSupervisedSignalFeature:
    def test_requires_labels_enforced(self) -> None:
        signals = _make_signals(5)
        feat = DummySupervised()
        with pytest.raises(ValueError, match="requires labels"):
            feat(signals, labels=None)

    def test_compute_with_labels(self) -> None:
        signals = _make_signals(20)
        labels = _make_labels(signals)
        feat = DummySupervised(window=5)
        result = feat(signals, labels=labels)

        assert isinstance(result, pl.DataFrame)
        assert "rolling_acc_5" in result.columns
        assert result.height == signals.height

    def test_rolling_accuracy_values(self) -> None:
        """Verify accuracy values are between 0 and 1."""
        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7)
        feat = DummySupervised(window=10)
        result = feat(signals, labels=labels)

        acc_col = result["rolling_acc_10"]
        non_null = acc_col.drop_nulls()
        assert (non_null >= 0).all()
        assert (non_null <= 1).all()


class TestLookAheadPrevention:
    """Core tests: supervised features must not peek at future labels."""

    def test_prepare_labels_with_t_hit(self) -> None:
        """Labels with t_hit: _resolved_at should equal t_hit."""
        signals = _make_signals(5)
        labels = _make_labels(signals, with_t_hit=True, hit_delay_hours=4)

        feat = DummySupervised()
        merged = feat.prepare_labels(signals, labels)

        assert "_resolved_at" in merged.columns
        # _resolved_at should be the t_hit values
        resolved = merged["_resolved_at"]
        t_hit = merged["t_hit"]
        assert resolved.to_list() == t_hit.to_list()

    def test_prepare_labels_with_delay(self) -> None:
        """Fallback to label_delay when no t_hit column."""
        signals = _make_signals(10)
        labels = _make_labels(signals, with_t_hit=False)

        feat = DummySupervised(label_resolve_col=None, label_delay=3)
        merged = feat.prepare_labels(signals, labels)

        assert "_resolved_at" in merged.columns
        # _resolved_at should be shifted by 3 rows within each pair
        pair0 = merged.filter(pl.col("pair") == "PAIR0").sort("timestamp")
        ts = pair0["timestamp"].to_list()
        resolved = pair0["_resolved_at"].to_list()
        # First rows: _resolved_at = ts shifted forward by 3
        for i in range(len(ts) - 3):
            assert resolved[i] == ts[i + 3]

    def test_mask_unresolved_nulls_future_labels(self) -> None:
        """mask_unresolved should null labels where _resolved_at > timestamp."""
        signals = _make_signals(10)
        labels = _make_labels(signals, with_t_hit=True, hit_delay_hours=4)

        feat = DummySupervised()
        merged = feat.prepare_labels(signals, labels)
        masked = feat.mask_unresolved(merged)

        # For each row: label should be null if t_hit > timestamp
        for row in masked.to_dicts():
            resolved = row["_resolved_at"]
            ts = row["timestamp"]
            if resolved is not None and ts is not None and resolved > ts:
                assert row["label"] is None, f"Label should be null at {ts} (resolves at {resolved})"

    def test_mask_unresolved_keeps_resolved_labels(self) -> None:
        """mask_unresolved should keep labels where _resolved_at <= timestamp."""
        # Use hit_delay=0 so all labels resolve immediately
        signals = _make_signals(10)
        labels = _make_labels(signals, with_t_hit=True, hit_delay_hours=0)

        feat = DummySupervised()
        merged = feat.prepare_labels(signals, labels)
        masked = feat.mask_unresolved(merged)

        # All labels should be visible (t_hit == timestamp)
        null_count = masked["label"].null_count()
        assert null_count == 0, f"Expected 0 nulls, got {null_count}"

    def test_mask_requires_prepare_labels(self) -> None:
        """mask_unresolved should fail without _resolved_at column."""
        df = pl.DataFrame(
            {
                "pair": ["A"],
                "timestamp": [datetime.now()],
                "label": ["rise"],
            }
        )
        feat = DummySupervised()
        with pytest.raises(ValueError, match="_resolved_at"):
            feat.mask_unresolved(df)

    def test_no_future_information_in_rolling(self) -> None:
        """End-to-end: supervised feature should use only past-resolved labels.

        Create signals where early labels resolve quickly (0h delay) and
        later labels resolve with large delay (100h). The rolling accuracy
        computed at early timestamps should NOT include late-resolving labels.
        """
        base = datetime(2024, 1, 1)
        n = 20

        signals = pl.DataFrame(
            {
                "pair": ["PAIR0"] * n,
                "timestamp": [base + timedelta(hours=i) for i in range(n)],
                "signal_type": ["rise"] * n,
                "signal": [1] * n,
                "probability": [0.8] * n,
            }
        )

        # First 10 labels resolve instantly (all correct = "rise")
        # Last 10 labels resolve way in the future (all wrong = "fall")
        labels_rows = []
        for i in range(n):
            ts = base + timedelta(hours=i)
            if i < 10:
                labels_rows.append(
                    {
                        "pair": "PAIR0",
                        "timestamp": ts,
                        "label": "rise",  # correct
                        "t_hit": ts,  # resolves immediately
                    }
                )
            else:
                labels_rows.append(
                    {
                        "pair": "PAIR0",
                        "timestamp": ts,
                        "label": "fall",  # wrong
                        "t_hit": ts + timedelta(hours=100),  # resolves far in future
                    }
                )

        labels = pl.DataFrame(labels_rows)

        feat = DummySupervised(window=5)
        result = feat(signals, labels=labels)

        # At timestamp index 12 (hour 12): t_hit for labels 10-12 is > 100h away
        # Only labels 0-9 should be visible, all correct → accuracy ~1.0
        row_12 = result.sort("timestamp").row(12, named=True)
        acc = row_12["rolling_acc_5"]
        assert acc is not None
        assert acc > 0.9, (
            f"At hour 12, rolling accuracy should be ~1.0 (only resolved correct labels visible), got {acc}"
        )


class TestValidation:
    def test_missing_signal_columns(self) -> None:
        df = pl.DataFrame({"pair": ["A"]})
        feat = DummyUnsupervised()
        with pytest.raises(ValueError, match="signal_type"):
            feat(df)

    def test_missing_label_columns(self) -> None:
        signals = _make_signals(5)
        bad_labels = pl.DataFrame({"pair": ["A"], "timestamp": [datetime.now()]})
        feat = DummySupervised()
        with pytest.raises(ValueError, match="label"):
            feat(signals, labels=bad_labels)


class TestDecoratorRegistration:
    def test_decorator_registers_component(self) -> None:
        from signalflow.core.registry import default_registry

        @dataclass
        class _TestFeature(SignalFeature):
            outputs: ClassVar[list[str]] = ["test_col"]

            def compute(
                self,
                signals: pl.DataFrame,
                labels: pl.DataFrame | None = None,
                context: dict[str, Any] | None = None,
            ) -> pl.DataFrame:
                return signals.select(["pair", "timestamp"]).with_columns(pl.lit(1).alias("test_col"))

        # Register via decorator
        import signalflow as sf

        sf.signal_feature("test/signal_feature_test")(_TestFeature)

        # Verify registration
        info = default_registry.list(SfComponentType.SIGNAL_FEATURE)
        assert "test/signal_feature_test" in info

        # Verify creation
        instance = default_registry.create(SfComponentType.SIGNAL_FEATURE, "test/signal_feature_test")
        assert isinstance(instance, SignalFeature)

    def test_infer_component_type(self) -> None:
        """@sf.register should auto-detect SignalFeature type."""
        import signalflow as sf

        @dataclass
        @sf.register("test/signal_feature_infer")
        class _InferFeature(SignalFeature):
            outputs: ClassVar[list[str]] = ["inferred"]

            def compute(
                self,
                signals: pl.DataFrame,
                labels: pl.DataFrame | None = None,
                context: dict[str, Any] | None = None,
            ) -> pl.DataFrame:
                return signals.select(["pair", "timestamp"]).with_columns(pl.lit(1).alias("inferred"))

        from signalflow.core.registry import default_registry

        info = default_registry.list(SfComponentType.SIGNAL_FEATURE)
        assert "test/signal_feature_infer" in info
