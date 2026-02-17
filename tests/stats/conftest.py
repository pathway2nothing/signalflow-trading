"""Fixtures for stats module tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from signalflow.core import Portfolio, RawData, Signals, StrategyState


@dataclass
class MockTrade:
    """Mock trade for testing."""

    pnl: float
    pair: str = "BTCUSDT"
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: datetime = field(default_factory=datetime.now)


@dataclass
class MockBacktestResult:
    """Mock BacktestResult for testing stats module."""

    state: StrategyState
    trades: list[MockTrade]
    signals: Signals | None
    raw: RawData
    config: dict[str, Any]
    metrics_df: pl.DataFrame | None = None

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def final_capital(self) -> float:
        return float(self.state.portfolio.cash)

    @property
    def initial_capital(self) -> float:
        return float(self.config.get("capital", 10_000.0))

    @property
    def total_return(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.final_capital - self.initial_capital) / self.initial_capital

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def profit_factor(self) -> float:
        if not self.trades:
            return 0.0
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def metrics(self) -> dict[str, float]:
        return {
            "n_trades": float(self.n_trades),
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "profit_factor": self.profit_factor,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "max_drawdown": 0.1,  # Mock value
        }


@pytest.fixture
def sample_trades() -> list[MockTrade]:
    """Generate sample trades for testing."""
    np.random.seed(42)

    trades = []
    for i in range(50):
        # Mix of winning and losing trades
        pnl = np.random.uniform(50, 200) if np.random.random() > 0.4 else -np.random.uniform(30, 100)

        trades.append(
            MockTrade(
                pnl=pnl,
                pair="BTCUSDT",
                entry_time=datetime(2024, 1, 1) + timedelta(hours=i * 4),
                exit_time=datetime(2024, 1, 1) + timedelta(hours=i * 4 + 2),
            )
        )

    return trades


@pytest.fixture
def sample_pnls() -> np.ndarray:
    """Generate sample PnLs for direct testing."""
    np.random.seed(42)

    pnls = []
    for _ in range(50):
        if np.random.random() > 0.4:
            pnls.append(np.random.uniform(50, 200))
        else:
            pnls.append(-np.random.uniform(30, 100))

    return np.array(pnls, dtype=np.float64)


@pytest.fixture
def sample_returns() -> np.ndarray:
    """Generate sample returns for direct testing."""
    np.random.seed(42)
    # Generate returns with slight positive drift
    returns = np.random.normal(0.001, 0.02, 100)
    return returns.astype(np.float64)


@pytest.fixture
def mock_raw_data() -> RawData:
    """Create minimal mock RawData."""
    # Create minimal spot data
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(100)]

    spot_df = pl.DataFrame(
        {
            "pair": ["BTCUSDT"] * 100,
            "timestamp": timestamps,
            "open": [40000.0 + i * 10 for i in range(100)],
            "high": [40050.0 + i * 10 for i in range(100)],
            "low": [39950.0 + i * 10 for i in range(100)],
            "close": [40020.0 + i * 10 for i in range(100)],
            "volume": [100.0] * 100,
        }
    )

    return RawData(
        datetime_start=datetime(2024, 1, 1),
        datetime_end=datetime(2024, 1, 5),
        pairs=["BTCUSDT"],
        data={"spot": spot_df},
    )


@pytest.fixture
def mock_backtest_result(
    sample_trades: list[MockTrade],
    mock_raw_data: RawData,
) -> MockBacktestResult:
    """Create mock BacktestResult for testing."""
    initial_capital = 10_000.0
    total_pnl = sum(t.pnl for t in sample_trades)
    final_capital = initial_capital + total_pnl

    state = StrategyState(
        strategy_id="test_strategy",
        portfolio=Portfolio(cash=final_capital),
    )

    # Create metrics_df with time series data
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(len(sample_trades))]
    cumulative_pnl = np.cumsum([t.pnl for t in sample_trades])
    total_returns = cumulative_pnl / initial_capital

    metrics_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "total_return": total_returns.tolist(),
            "equity": (initial_capital + cumulative_pnl).tolist(),
        }
    )

    return MockBacktestResult(
        state=state,
        trades=sample_trades,
        signals=None,
        raw=mock_raw_data,
        config={"capital": initial_capital},
        metrics_df=metrics_df,
    )
