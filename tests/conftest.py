"""Shared pytest fixtures for the SignalFlow V5 test suite."""


import warnings

import pytest

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import signalflow as sf


@pytest.fixture(scope="session")
def ds():
    """Small deterministic two-pair hourly dataset."""
    return sf.data("memory", pairs=["BTCUSDT", "ETHUSDT"], start="2023-01-01", end="2023-03-01", interval="1h")


@pytest.fixture(scope="session")
def fitted_forecast(ds):
    """A fitted tier-1 forecast model (reused across tests)."""
    m = sf.ForecastModel(
        backend="lightgbm",
        target=sf.FixedHorizon(bars=12),
        features=sf.FeaturePipe(sf.SMA(20), sf.SMA(10), sf.SMA(50)),
        output="p_rise",
        n_folds=3,
    )
    return m.fit(ds)
