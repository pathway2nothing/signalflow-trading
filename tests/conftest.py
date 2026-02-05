"""Shared fixtures for signalflow tests."""

import pytest
import polars as pl
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_df() -> pl.DataFrame:
    """Minimal OHLCV DataFrame for two pairs."""
    base = datetime(2024, 1, 1)
    rows = []
    for pair in ["BTCUSDT", "ETHUSDT"]:
        for i in range(20):
            rows.append(
                {
                    "pair": pair,
                    "timestamp": base + timedelta(hours=i),
                    "open": 100.0 + i,
                    "high": 105.0 + i,
                    "low": 95.0 + i,
                    "close": 102.0 + i,
                    "volume": 1000.0 + i * 10,
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def sample_signals_df() -> pl.DataFrame:
    """Minimal signals DataFrame."""
    from signalflow.core.enums import SignalType

    base = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "pair": ["BTCUSDT", "BTCUSDT", "ETHUSDT"],
            "timestamp": [base, base + timedelta(hours=1), base],
            "signal_type": [SignalType.RISE.value, SignalType.FALL.value, SignalType.NONE.value],
            "signal": [1, -1, 0],
            "probability": [0.9, 0.8, 0.0],
        }
    )


@pytest.fixture
def raw_data(sample_ohlcv_df) -> "RawData":
    """RawData instance with spot data."""
    from signalflow.core.containers.raw_data import RawData

    return RawData(
        datetime_start=datetime(2024, 1, 1),
        datetime_end=datetime(2024, 1, 2),
        pairs=["BTCUSDT", "ETHUSDT"],
        data={"spot": sample_ohlcv_df},
    )
