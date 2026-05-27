"""Tests for signalflow.api.batch module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from signalflow.api.batch import BatchResult, batch_run

# ── Helpers ────────────────────────────────────────────────────────────


@dataclass
class _MockFlowResult:
    """Minimal mock of FlowResult."""

    strategy_id: str = ""
    flow_config: Any = None
    backtest_metrics: dict[str, float] | None = None

    @property
    def metrics(self) -> dict[str, float]:
        return self.backtest_metrics or {}


@dataclass
class _MockFlowConfig:
    strategy_id: str = ""


class _MockFlowBuilder:
    """Minimal mock of FlowBuilder that returns a preset result."""

    def __init__(self, label: str = "", metrics: dict[str, float] | None = None, fail: bool = False) -> None:
        self._strategy_id = label
        self._metrics = metrics or {"sharpe_ratio": 1.0}
        self._fail = fail

    def run(self, **kwargs: Any) -> _MockFlowResult:
        if self._fail:
            msg = "Intentional test failure"
            raise RuntimeError(msg)
        config = _MockFlowConfig(strategy_id=self._strategy_id)
        return _MockFlowResult(
            strategy_id=self._strategy_id,
            flow_config=config,
            backtest_metrics=self._metrics,
        )


# ── batch_run ──────────────────────────────────────────────────────────


class TestBatchRun:
    """Tests for batch_run()."""

    def test_basic_sequential(self) -> None:
        configs = [
            _MockFlowBuilder("a", {"sharpe_ratio": 1.0}),
            _MockFlowBuilder("b", {"sharpe_ratio": 2.0}),
        ]
        result = batch_run(configs)  # type: ignore[arg-type]
        assert isinstance(result, BatchResult)
        assert len(result.results) == 2
        assert len(result.successful) == 2
        assert len(result.errors) == 0

    def test_parallel_mode(self) -> None:
        configs = [
            _MockFlowBuilder("a", {"sharpe_ratio": 1.0}),
            _MockFlowBuilder("b", {"sharpe_ratio": 2.0}),
            _MockFlowBuilder("c", {"sharpe_ratio": 1.5}),
        ]
        result = batch_run(configs, parallel=True, max_workers=2)  # type: ignore[arg-type]
        assert len(result.successful) == 3

    def test_handles_failures(self) -> None:
        configs = [
            _MockFlowBuilder("a", {"sharpe_ratio": 1.0}),
            _MockFlowBuilder("b", fail=True),
            _MockFlowBuilder("c", {"sharpe_ratio": 1.5}),
        ]
        result = batch_run(configs)  # type: ignore[arg-type]
        assert len(result.successful) == 2
        assert 1 in result.errors
        assert result.results[1] is None

    def test_custom_labels(self) -> None:
        configs = [_MockFlowBuilder(), _MockFlowBuilder()]
        result = batch_run(configs, labels=["alpha", "beta"])  # type: ignore[arg-type]
        assert result.labels == ["alpha", "beta"]

    def test_auto_labels_from_strategy_id(self) -> None:
        configs = [_MockFlowBuilder("my_strat"), _MockFlowBuilder("")]
        result = batch_run(configs)  # type: ignore[arg-type]
        assert result.labels[0] == "my_strat"
        assert result.labels[1] == "run_1"

    def test_labels_length_mismatch_raises(self) -> None:
        configs = [_MockFlowBuilder(), _MockFlowBuilder()]
        with pytest.raises(ValueError, match="labels length"):
            batch_run(configs, labels=["only_one"])  # type: ignore[arg-type]

    def test_empty_configs_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            batch_run([])

    def test_progress_callback(self) -> None:
        configs = [_MockFlowBuilder("a"), _MockFlowBuilder("b")]
        calls: list[tuple[int, int, str]] = []

        def cb(completed: int, total: int, label: str) -> None:
            calls.append((completed, total, label))

        batch_run(configs, progress=cb)  # type: ignore[arg-type]
        assert len(calls) == 2
        assert calls[-1][0] == 2  # Last call has completed=2
        assert calls[-1][1] == 2  # Total=2

    def test_run_kwargs_passed(self) -> None:
        builder = MagicMock()
        builder._strategy_id = ""
        mock_result = _MockFlowResult(backtest_metrics={"x": 1.0})
        builder.run.return_value = mock_result

        batch_run([builder, builder], run_kwargs={"mode": "walk_forward"})
        builder.run.assert_called_with(mode="walk_forward")

    def test_elapsed_time(self) -> None:
        configs = [_MockFlowBuilder("a")]
        result = batch_run(configs)  # type: ignore[arg-type]
        assert result.elapsed >= 0


# ── BatchResult ────────────────────────────────────────────────────────


class TestBatchResult:
    """Tests for BatchResult methods."""

    def test_comparison(self) -> None:
        configs = [
            _MockFlowBuilder("a", {"sharpe_ratio": 1.0, "total_return": 0.10}),
            _MockFlowBuilder("b", {"sharpe_ratio": 2.0, "total_return": 0.20}),
        ]
        result = batch_run(configs)  # type: ignore[arg-type]
        cmp = result.comparison
        assert cmp.best("sharpe_ratio") in ("a", "b")

    def test_comparison_needs_two(self) -> None:
        configs = [_MockFlowBuilder("a")]
        result = batch_run(configs)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="at least 2"):
            _ = result.comparison

    def test_summary(self) -> None:
        configs = [
            _MockFlowBuilder("a", {"sharpe_ratio": 1.0}),
            _MockFlowBuilder("b", {"sharpe_ratio": 2.0}),
        ]
        result = batch_run(configs)  # type: ignore[arg-type]
        s = result.summary()
        assert "2 configs" in s
        assert "2 succeeded" in s

    def test_summary_with_failures(self) -> None:
        configs = [
            _MockFlowBuilder("a"),
            _MockFlowBuilder("b", fail=True),
        ]
        result = batch_run(configs)  # type: ignore[arg-type]
        s = result.summary()
        assert "1 failed" in s
