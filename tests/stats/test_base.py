"""Tests for signalflow.analytic.stats.base module."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from signalflow.analytic.stats.base import (
    SimulationConfig,
    extract_pnls,
    extract_returns,
)


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_default_values(self):
        config = SimulationConfig()
        assert config.n_simulations == 10_000
        assert config.random_seed is None
        assert config.confidence_levels == (0.05, 0.50, 0.95)
        assert config.n_jobs == 1

    def test_custom_values(self):
        config = SimulationConfig(
            n_simulations=5000,
            random_seed=42,
            confidence_levels=(0.01, 0.99),
            n_jobs=4,
        )
        assert config.n_simulations == 5000
        assert config.random_seed == 42
        assert config.confidence_levels == (0.01, 0.99)
        assert config.n_jobs == 4


class TestExtractPnls:
    """Tests for extract_pnls function."""

    def test_empty_trades(self):
        result = extract_pnls([])
        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_pnl_attribute(self):
        """Test extraction from objects with .pnl attribute."""

        @dataclass
        class Trade:
            pnl: float

        trades = [Trade(pnl=100.0), Trade(pnl=-50.0), Trade(pnl=75.0)]
        result = extract_pnls(trades)

        np.testing.assert_array_equal(result, [100.0, -50.0, 75.0])

    def test_realized_pnl_attribute(self):
        """Test extraction from objects with .realized_pnl attribute."""

        @dataclass
        class Trade:
            realized_pnl: float

        trades = [Trade(realized_pnl=200.0), Trade(realized_pnl=-30.0)]
        result = extract_pnls(trades)

        np.testing.assert_array_equal(result, [200.0, -30.0])

    def test_total_pnl_attribute(self):
        """Test extraction from objects with .total_pnl attribute."""

        @dataclass
        class Trade:
            total_pnl: float

        trades = [Trade(total_pnl=150.0)]
        result = extract_pnls(trades)

        np.testing.assert_array_equal(result, [150.0])

    def test_dict_with_pnl(self):
        """Test extraction from dictionaries with pnl key."""
        trades = [{"pnl": 100.0}, {"pnl": -25.0}]
        result = extract_pnls(trades)

        np.testing.assert_array_equal(result, [100.0, -25.0])

    def test_dict_with_realized_pnl(self):
        """Test extraction from dictionaries with realized_pnl key."""
        trades = [{"realized_pnl": 50.0}]
        result = extract_pnls(trades)

        np.testing.assert_array_equal(result, [50.0])

    def test_dict_with_total_pnl(self):
        """Test extraction from dictionaries with total_pnl key."""
        trades = [{"total_pnl": 75.0}]
        result = extract_pnls(trades)

        np.testing.assert_array_equal(result, [75.0])

    def test_none_pnl_becomes_zero(self):
        """Test that None pnl values become 0.0."""

        @dataclass
        class Trade:
            pnl: float | None

        trades = [Trade(pnl=100.0), Trade(pnl=None)]
        result = extract_pnls(trades)

        np.testing.assert_array_equal(result, [100.0, 0.0])


class TestExtractReturns:
    """Tests for extract_returns function."""

    def test_from_metrics_df(self):
        """Test extraction from metrics_df total_return column."""
        # Create mock result with metrics_df
        result = MagicMock()
        result.metrics_df = pl.DataFrame(
            {
                "total_return": [0.0, 0.01, 0.03, 0.06, 0.10],
            }
        )
        result.trades = []
        result.initial_capital = 10000.0

        returns = extract_returns(result)

        assert len(returns) == 4  # diff reduces by 1
        assert isinstance(returns, np.ndarray)

    def test_from_trades(self):
        """Test extraction from trades when metrics_df not available."""

        @dataclass
        class Trade:
            pnl: float

        result = MagicMock()
        result.metrics_df = None
        result.trades = [Trade(pnl=100.0), Trade(pnl=-50.0), Trade(pnl=200.0)]
        result.initial_capital = 10000.0

        returns = extract_returns(result)

        np.testing.assert_array_almost_equal(returns, [0.01, -0.005, 0.02])

    def test_empty_trades_returns_empty(self):
        """Test that empty trades returns empty array."""
        result = MagicMock()
        result.metrics_df = None
        result.trades = []
        result.initial_capital = 10000.0

        returns = extract_returns(result)

        assert len(returns) == 0

    def test_zero_initial_capital_fallback(self):
        """Test handling of zero or negative initial capital."""

        @dataclass
        class Trade:
            pnl: float

        result = MagicMock()
        result.metrics_df = None
        result.trades = [Trade(pnl=100.0)]
        result.initial_capital = 0.0  # Would cause division by zero

        returns = extract_returns(result)

        # Should use 1.0 as fallback
        np.testing.assert_array_equal(returns, [100.0])

    def test_metrics_df_without_total_return(self):
        """Test fallback when metrics_df exists but lacks total_return."""

        @dataclass
        class Trade:
            pnl: float

        result = MagicMock()
        result.metrics_df = pl.DataFrame({"other_column": [1, 2, 3]})
        result.trades = [Trade(pnl=50.0)]
        result.initial_capital = 1000.0

        returns = extract_returns(result)

        # Should fall back to trade-based returns
        np.testing.assert_array_almost_equal(returns, [0.05])
