"""Fixtures for API tests."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import polars as pl
import pytest

from signalflow.core import RawData, Signals, StrategyState, Portfolio


@pytest.fixture
def sample_ohlcv_df() -> pl.DataFrame:
    """Create sample OHLCV DataFrame."""
    n_bars = 100
    base_price = 50000.0
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_bars)]

    return pl.DataFrame({
        "pair": ["BTCUSDT"] * n_bars,
        "timestamp": timestamps,
        "open": [base_price + i * 10 for i in range(n_bars)],
        "high": [base_price + i * 10 + 50 for i in range(n_bars)],
        "low": [base_price + i * 10 - 50 for i in range(n_bars)],
        "close": [base_price + i * 10 + 25 for i in range(n_bars)],
        "volume": [1000.0] * n_bars,
    })


@pytest.fixture
def sample_raw_data(sample_ohlcv_df: pl.DataFrame) -> RawData:
    """Create sample RawData container."""
    return RawData(
        datetime_start=datetime(2024, 1, 1),
        datetime_end=datetime(2024, 1, 5),
        pairs=["BTCUSDT"],
        data={"spot": sample_ohlcv_df},
    )


@pytest.fixture
def sample_signals() -> Signals:
    """Create sample Signals."""
    df = pl.DataFrame({
        "pair": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        "timestamp": [
            datetime(2024, 1, 1, 10),
            datetime(2024, 1, 1, 20),
            datetime(2024, 1, 2, 10),
        ],
        "signal_type": ["rise", "fall", "rise"],
        "signal": [1, -1, 1],
    })
    return Signals(df)


@pytest.fixture
def sample_state() -> StrategyState:
    """Create sample StrategyState."""
    state = MagicMock(spec=StrategyState)
    state.capital = 12000.0
    state.portfolio = MagicMock(spec=Portfolio)
    state.portfolio.positions = {}
    return state


@pytest.fixture
def sample_trades() -> list:
    """Create sample trades list."""
    class MockTrade:
        def __init__(self, pnl: float):
            self.pnl = pnl
            self.pair = "BTCUSDT"

    return [
        MockTrade(100.0),   # win
        MockTrade(-50.0),   # loss
        MockTrade(200.0),   # win
        MockTrade(-30.0),   # loss
        MockTrade(150.0),   # win
    ]


@pytest.fixture
def sample_result(sample_state, sample_raw_data, sample_signals, sample_trades):
    """Create sample BacktestResult."""
    from signalflow.api.result import BacktestResult

    return BacktestResult(
        state=sample_state,
        trades=sample_trades,
        signals=sample_signals,
        raw=sample_raw_data,
        config={"capital": 10_000.0, "tp": 0.02, "sl": 0.01},
    )
