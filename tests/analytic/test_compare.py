"""Tests for signalflow.analytic.compare module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
import pytest

from signalflow.analytic.compare import ComparisonResult, compare_results


# ── Helpers ────────────────────────────────────────────────────────────


@dataclass
class _MockResult:
    """Minimal result object with .metrics and .strategy_id."""

    strategy_id: str
    metrics: dict[str, float]


@dataclass
class _NamedResult:
    """Result that uses .name instead of .strategy_id."""

    name: str
    metrics: dict[str, float]


@dataclass
class _BareResult:
    """Result with no name attributes."""

    metrics: dict[str, float]


# ── compare_results ────────────────────────────────────────────────────


class TestCompareResults:
    """Tests for compare_results() function."""

    def test_basic_comparison(self) -> None:
        a = _MockResult("A", {"sharpe_ratio": 1.5, "total_return": 0.20})
        b = _MockResult("B", {"sharpe_ratio": 2.0, "total_return": 0.10})
        cmp = compare_results(a, b)

        assert isinstance(cmp, ComparisonResult)
        assert cmp.names == ["A", "B"]
        assert cmp.metrics_table.shape == (2, 3)  # metric + 2 runs

    def test_three_results(self) -> None:
        a = _MockResult("A", {"sharpe_ratio": 1.0})
        b = _MockResult("B", {"sharpe_ratio": 2.0})
        c = _MockResult("C", {"sharpe_ratio": 1.5})
        cmp = compare_results(a, b, c)

        assert len(cmp.names) == 3
        assert cmp.metrics_table.shape[1] == 4  # metric + 3 runs

    def test_requires_at_least_two(self) -> None:
        a = _MockResult("A", {"sharpe_ratio": 1.0})
        with pytest.raises(ValueError, match="at least 2"):
            compare_results(a)

    def test_no_metrics_raises_type_error(self) -> None:
        class _NoMetrics:
            strategy_id = "bad"

        with pytest.raises(TypeError, match="no .metrics"):
            compare_results(_NoMetrics(), _NoMetrics())

    def test_non_dict_metrics_raises_type_error(self) -> None:
        class _BadMetrics:
            strategy_id = "bad"
            metrics = [1, 2, 3]

        with pytest.raises(TypeError, match="expected dict"):
            compare_results(_BadMetrics(), _BadMetrics())

    def test_deduplicates_names(self) -> None:
        a = _MockResult("run", {"x": 1.0})
        b = _MockResult("run", {"x": 2.0})
        cmp = compare_results(a, b)
        assert cmp.names[0] != cmp.names[1]
        assert "run" in cmp.names[0]

    def test_name_from_strategy_id(self) -> None:
        r = _MockResult("my_strat", {"x": 1.0})
        cmp = compare_results(r, _MockResult("other", {"x": 2.0}))
        assert "my_strat" in cmp.names

    def test_name_from_name_attr(self) -> None:
        r = _NamedResult("named_strat", {"x": 1.0})
        cmp = compare_results(r, _MockResult("other", {"x": 2.0}))
        assert "named_strat" in cmp.names

    def test_fallback_name(self) -> None:
        a = _BareResult({"x": 1.0})
        b = _BareResult({"x": 2.0})
        cmp = compare_results(a, b)
        assert cmp.names[0] == "run_0"
        assert cmp.names[1] == "run_1"

    def test_mismatched_metric_keys(self) -> None:
        a = _MockResult("A", {"sharpe_ratio": 1.0, "total_return": 0.20})
        b = _MockResult("B", {"sharpe_ratio": 2.0, "max_drawdown": -0.05})
        cmp = compare_results(a, b)

        metrics = cmp.metrics_table["metric"].to_list()
        assert "sharpe_ratio" in metrics
        assert "total_return" in metrics
        assert "max_drawdown" in metrics
        # total_return missing for B → None
        row_tr = cmp.metrics_table.filter(pl.col("metric") == "total_return")
        assert row_tr["B"].item() is None

    def test_backtest_metrics_fallback(self) -> None:
        """FlowResult-style object with .backtest_metrics instead of .metrics."""

        class _FlowLike:
            strategy_id = "flow1"
            backtest_metrics: dict[str, float] = {"sharpe_ratio": 1.8}

        cmp = compare_results(_FlowLike(), _MockResult("B", {"sharpe_ratio": 1.0}))
        assert cmp.best("sharpe_ratio") == "flow1"


# ── ComparisonResult methods ──────────────────────────────────────────


class TestComparisonResult:
    """Tests for ComparisonResult methods."""

    @pytest.fixture()
    def cmp(self) -> ComparisonResult:
        a = _MockResult("A", {"sharpe_ratio": 1.5, "total_return": 0.25, "max_drawdown": -0.10})
        b = _MockResult("B", {"sharpe_ratio": 2.1, "total_return": 0.18, "max_drawdown": -0.08})
        c = _MockResult("C", {"sharpe_ratio": 0.9, "total_return": 0.35, "max_drawdown": -0.15})
        return compare_results(a, b, c)

    def test_best(self, cmp: ComparisonResult) -> None:
        assert cmp.best("sharpe_ratio") == "B"
        assert cmp.best("total_return") == "C"

    def test_worst(self, cmp: ComparisonResult) -> None:
        assert cmp.worst("sharpe_ratio") == "C"
        assert cmp.worst("max_drawdown") == "C"  # -0.15 is lowest

    def test_best_missing_metric_raises(self, cmp: ComparisonResult) -> None:
        with pytest.raises(KeyError, match="not found"):
            cmp.best("nonexistent")

    def test_worst_missing_metric_raises(self, cmp: ComparisonResult) -> None:
        with pytest.raises(KeyError, match="not found"):
            cmp.worst("nonexistent")

    def test_summary_contains_metrics(self, cmp: ComparisonResult) -> None:
        s = cmp.summary()
        assert "sharpe_ratio" in s
        assert "total_return" in s
        assert "A" in s
        assert "B" in s
        assert "C" in s
        assert "Best per metric:" in s

    def test_to_dict(self, cmp: ComparisonResult) -> None:
        d = cmp.to_dict()
        assert set(d["names"]) == {"A", "B", "C"}
        assert "A" in d["metrics"]
        assert "sharpe_ratio" in d["metrics"]["A"]

    def test_metrics_table_is_polars(self, cmp: ComparisonResult) -> None:
        assert isinstance(cmp.metrics_table, pl.DataFrame)
        assert "metric" in cmp.metrics_table.columns

    def test_raw_metrics(self, cmp: ComparisonResult) -> None:
        assert len(cmp.raw_metrics) == 3
        assert cmp.raw_metrics[0]["sharpe_ratio"] == 1.5

    def test_best_with_none_values(self) -> None:
        """best() skips None values when comparing."""
        a = _MockResult("A", {"x": 1.0})
        b = _MockResult("B", {})  # x is missing → None
        cmp = compare_results(a, b)
        assert cmp.best("x") == "A"
