"""Fixtures for viz tests."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.core.containers.raw_data import RawData
from signalflow.feature import FeaturePipeline
from signalflow.feature.examples import ExampleRsiFeature, ExampleSmaFeature


@pytest.fixture
def sample_feature_pipeline():
    """Simple feature pipeline for testing."""
    return FeaturePipeline(
        features=[
            ExampleRsiFeature(period=14),
            ExampleSmaFeature(period=20),
        ]
    )


@pytest.fixture
def multi_source_raw_data():
    """Multi-source RawData with nested structure."""
    base = datetime(2024, 1, 1)
    rows = []
    for pair in ["BTCUSDT", "ETHUSDT"]:
        for i in range(10):
            rows.append(
                {
                    "pair": pair,
                    "timestamp": base + timedelta(hours=i),
                    "open": 100.0 + i,
                    "high": 105.0 + i,
                    "low": 95.0 + i,
                    "close": 102.0 + i,
                    "volume": 1000.0,
                    "open_interest": 50000.0 + i * 100,
                    "funding_rate": 0.0001,
                }
            )
    df = pl.DataFrame(rows)

    return RawData(
        datetime_start=datetime(2024, 1, 1),
        datetime_end=datetime(2024, 1, 2),
        pairs=["BTCUSDT", "ETHUSDT"],
        data={
            "perpetual": {
                "binance": df,
                "okx": df,
            }
        },
        default_source="binance",
    )


@pytest.fixture
def flat_raw_data():
    """Single-source RawData with flat structure."""
    base = datetime(2024, 1, 1)
    rows = []
    for pair in ["BTCUSDT"]:
        for i in range(10):
            rows.append(
                {
                    "pair": pair,
                    "timestamp": base + timedelta(hours=i),
                    "open": 100.0 + i,
                    "high": 105.0 + i,
                    "low": 95.0 + i,
                    "close": 102.0 + i,
                    "volume": 1000.0,
                }
            )

    return RawData(
        datetime_start=datetime(2024, 1, 1),
        datetime_end=datetime(2024, 1, 2),
        pairs=["BTCUSDT"],
        data={"spot": pl.DataFrame(rows)},
    )
