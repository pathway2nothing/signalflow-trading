"""Shared pytest fixtures for the SignalFlow test suite."""


import warnings

import pytest

import signalflow as sf

warnings.filterwarnings("ignore", message="X does not have valid feature names")


@pytest.fixture(scope="session")
def ds():
    """Small deterministic two-pair hourly dataset."""
    return sf.dataset("synthetic", pairs=["BTCUSDT", "ETHUSDT"], start="2023-01-01", end="2023-03-01", interval="1h")


@pytest.fixture(scope="session")
def fitted_forecast(ds):
    """A fitted tier-1 forecast model (reused across tests)."""
    m = sf.ForecastModel(
        backend="lightgbm",
        target=sf.FixedHorizon(bars=12),
        features=sf.FeaturePipeline(sf.SMA(20), sf.SMA(10), sf.SMA(50), sf.WoE(), sf.IVSelector()),
        output="p_rise",
        cv=sf.Rolling(step="1d", window="365d"),  # daily blocks: near-full OOS coverage for the flow tests
    )
    return m.fit(ds)
