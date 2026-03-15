"""Tests for FlowBuilder.clone() and FlowBuilder.sweep() methods."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from signalflow.api.flow import FlowBuilder


class TestClone:
    """Tests for FlowBuilder.clone()."""

    def test_clone_returns_new_instance(self) -> None:
        base = FlowBuilder(strategy_id="base")
        cloned = base.clone()
        assert cloned is not base
        assert cloned.strategy_id == "base"

    def test_clone_override_strategy_id(self) -> None:
        base = FlowBuilder(strategy_id="base")
        cloned = base.clone(strategy_id="variant")
        assert cloned.strategy_id == "variant"
        assert base.strategy_id == "base"

    def test_clone_override_capital(self) -> None:
        base = FlowBuilder(strategy_id="base")
        base._capital = 10_000.0
        cloned = base.clone(capital=50_000.0)
        assert cloned._capital == 50_000.0
        assert base._capital == 10_000.0

    def test_clone_override_fee(self) -> None:
        base = FlowBuilder(strategy_id="base")
        base._fee = 0.001
        cloned = base.clone(fee=0.0005)
        assert cloned._fee == 0.0005
        assert base._fee == 0.001

    def test_clone_is_deep_copy(self) -> None:
        base = FlowBuilder(strategy_id="base")
        base._entry_config = {"size_pct": 0.1}
        cloned = base.clone()
        cloned._entry_config["size_pct"] = 0.5
        assert base._entry_config["size_pct"] == 0.1  # Unchanged

    def test_clone_with_detector_override(self) -> None:
        base = FlowBuilder(strategy_id="base")
        mock_det = MagicMock()
        mock_det.fast_period = 20
        base._named_detectors = {"default": mock_det}

        cloned = base.clone(detector={"fast_period": 30})
        # The detector on the clone should have been updated
        assert cloned._named_detectors["default"].fast_period == 30

    def test_clone_with_entry_override(self) -> None:
        base = FlowBuilder(strategy_id="base")
        base._entry_config = {"size_pct": 0.1}
        cloned = base.clone(entry={"size_pct": 0.2})
        assert cloned._entry_config["size_pct"] == 0.2
        assert base._entry_config["size_pct"] == 0.1

    def test_clone_with_exit_override(self) -> None:
        base = FlowBuilder(strategy_id="base")
        base._exit_config = {"tp": 0.02, "sl": 0.01}
        cloned = base.clone(exit={"tp": 0.05})
        assert cloned._exit_config["tp"] == 0.05
        assert cloned._exit_config["sl"] == 0.01
        assert base._exit_config["tp"] == 0.02


class TestSweep:
    """Tests for FlowBuilder.sweep()."""

    @patch("signalflow.api.batch.batch_run")
    def test_sweep_single_param(self, mock_batch: MagicMock) -> None:
        base = FlowBuilder(strategy_id="base")
        base._exit_config = {"tp": 0.02}

        mock_batch.return_value = MagicMock()
        base.sweep({"exit.tp": [0.02, 0.03, 0.05]})

        mock_batch.assert_called_once()
        configs = mock_batch.call_args[0][0]
        assert len(configs) == 3  # 3 values

    @patch("signalflow.api.batch.batch_run")
    def test_sweep_grid_product(self, mock_batch: MagicMock) -> None:
        base = FlowBuilder(strategy_id="base")
        base._exit_config = {"tp": 0.02, "sl": 0.01}

        mock_batch.return_value = MagicMock()
        base.sweep({
            "exit.tp": [0.02, 0.03],
            "exit.sl": [0.01, 0.02],
        })

        configs = mock_batch.call_args[0][0]
        assert len(configs) == 4  # 2 * 2

    @patch("signalflow.api.batch.batch_run")
    def test_sweep_labels_generated(self, mock_batch: MagicMock) -> None:
        base = FlowBuilder(strategy_id="base")

        mock_batch.return_value = MagicMock()
        base.sweep({"capital": [10_000, 20_000]})

        labels = mock_batch.call_args[1]["labels"]
        assert len(labels) == 2
        assert "capital=10000" in labels[0]
        assert "capital=20000" in labels[1]

    @patch("signalflow.api.batch.batch_run")
    def test_sweep_passes_parallel(self, mock_batch: MagicMock) -> None:
        base = FlowBuilder(strategy_id="base")

        mock_batch.return_value = MagicMock()
        base.sweep({"fee": [0.001, 0.002]}, parallel=True, max_workers=2)

        assert mock_batch.call_args[1]["parallel"] is True
        assert mock_batch.call_args[1]["max_workers"] == 2

    @patch("signalflow.api.batch.batch_run")
    def test_sweep_passes_run_kwargs(self, mock_batch: MagicMock) -> None:
        base = FlowBuilder(strategy_id="base")

        mock_batch.return_value = MagicMock()
        base.sweep({"fee": [0.001]}, run_kwargs={"mode": "walk_forward"})

        assert mock_batch.call_args[1]["run_kwargs"] == {"mode": "walk_forward"}

    @patch("signalflow.api.batch.batch_run")
    def test_sweep_detector_params(self, mock_batch: MagicMock) -> None:
        base = FlowBuilder(strategy_id="base")
        mock_det = MagicMock()
        mock_det.fast_period = 20
        base._named_detectors = {"default": mock_det}

        mock_batch.return_value = MagicMock()
        base.sweep({"detector.fast_period": [10, 20, 30]})

        configs = mock_batch.call_args[0][0]
        assert len(configs) == 3
