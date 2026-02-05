"""Shared fixtures for signalflow tests."""

import pytest
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path


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


@pytest.fixture
def strategy_state():
    """Fresh StrategyState with 10k cash."""
    from signalflow.core.containers.strategy_state import StrategyState

    state = StrategyState(strategy_id="test")
    state.portfolio.cash = 10000.0
    return state


@pytest.fixture
def long_position():
    """Open LONG position in BTCUSDT."""
    from signalflow.core.containers.position import Position
    from signalflow.core.enums import PositionType

    return Position(
        id="pos_long",
        pair="BTCUSDT",
        position_type=PositionType.LONG,
        entry_price=100.0,
        last_price=100.0,
        qty=1.0,
        entry_time=datetime(2024, 1, 1),
        last_time=datetime(2024, 1, 1),
    )


@pytest.fixture
def short_position():
    """Open SHORT position in ETHUSDT."""
    from signalflow.core.containers.position import Position
    from signalflow.core.enums import PositionType

    return Position(
        id="pos_short",
        pair="ETHUSDT",
        position_type=PositionType.SHORT,
        entry_price=100.0,
        last_price=100.0,
        qty=1.0,
        entry_time=datetime(2024, 1, 1),
        last_time=datetime(2024, 1, 1),
    )


@pytest.fixture
def rise_signals_df():
    """Signals DataFrame with a RISE signal for BTCUSDT."""
    from signalflow.core.enums import SignalType

    base = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "pair": ["BTCUSDT"],
            "timestamp": [base],
            "signal_type": [SignalType.RISE.value],
            "signal": [1],
            "probability": [0.9],
        }
    )


@pytest.fixture
def ohlcv_100_bars() -> pl.DataFrame:
    """100 one-minute bars per pair for feature warmup tests."""
    import math

    base = datetime(2024, 1, 1)
    rows = []
    for pair in ["BTCUSDT", "ETHUSDT"]:
        for i in range(100):
            price = 100.0 + 10 * math.sin(i / 5.0) + i * 0.1
            rows.append(
                {
                    "pair": pair,
                    "timestamp": base + timedelta(minutes=i),
                    "open": price - 0.5,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "volume": 1000.0 + i * 10,
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Store fixtures (parametrized over backends)
# ---------------------------------------------------------------------------


@pytest.fixture(params=["duckdb", "sqlite"])
def raw_store(request, tmp_path):
    """Parametrized RawDataStore fixture — runs tests against DuckDB and SQLite."""
    backend = request.param
    db_path = tmp_path / f"test_spot.{backend}"

    if backend == "duckdb":
        from signalflow.data.raw_store import DuckDbSpotStore

        store = DuckDbSpotStore(db_path=db_path, timeframe="1m")
    else:
        from signalflow.data.raw_store import SqliteSpotStore

        store = SqliteSpotStore(db_path=db_path, timeframe="1m")

    yield store
    store.close()


@pytest.fixture(params=["duckdb", "sqlite"])
def strategy_store(request, tmp_path):
    """Parametrized StrategyStore fixture — runs tests against DuckDB and SQLite."""
    backend = request.param
    db_path = tmp_path / f"test_strategy.{backend}"

    if backend == "duckdb":
        from signalflow.data.strategy_store import DuckDbStrategyStore

        store = DuckDbStrategyStore(str(db_path))
    else:
        from signalflow.data.strategy_store import SqliteStrategyStore

        store = SqliteStrategyStore(str(db_path))

    store.init()
    yield store
    store.close()


@pytest.fixture
def sample_klines():
    """Sample klines for insert_klines tests (100 one-minute bars)."""
    base = datetime(2024, 1, 1)
    return [
        {
            "timestamp": base + timedelta(minutes=i),
            "open": 100.0 + i,
            "high": 105.0 + i,
            "low": 95.0 + i,
            "close": 102.0 + i,
            "volume": 1000.0 + i * 10,
            "trades": 50 + i,
        }
        for i in range(100)
    ]
