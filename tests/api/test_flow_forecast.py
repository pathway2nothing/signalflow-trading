"""Етап 2 — .forecast() + контракт forecasts= (вікно) на FlowBuilder.

Перевіряє лише декларативний рівень (lazy ModelRef, warmup-silence-контракт вікна,
валідація посилань). Завантаження ваг і нарізка вікна — Етап 6 / runtime.
"""

from __future__ import annotations

import pytest

from signalflow.api.exceptions import ConfigurationError
from signalflow.api.flow import flow
from signalflow.models import ModelRef


class TestForecastRegistration:
    def test_forecast_from_mlflow_uri(self):
        b = flow().forecast("revert", mlflow="models:/revert/3")
        ref = b._named_forecasts["revert"]
        assert isinstance(ref, ModelRef)
        assert ref.name == "revert" and str(ref.version) == "3" and ref.source == "mlflow"

    def test_forecast_lazy_no_weights_loaded(self):
        # Building must not touch the network / load weights — only a ModelRef is held.
        b = flow().forecast("revert", mlflow="models:/revert/3")
        assert isinstance(b._named_forecasts["revert"], ModelRef)

    def test_forecast_hf_requires_version(self):
        with pytest.raises(ConfigurationError, match="explicit version"):
            flow().forecast("m", hf_path="org/model")

    def test_forecast_hf_with_version(self):
        b = flow().forecast("m", hf_path="org/model", version="2")
        ref = b._named_forecasts["m"]
        assert ref.source == "hf" and ref.name == "org/model" and str(ref.version) == "2"

    def test_forecast_needs_source(self):
        with pytest.raises(ConfigurationError, match="mlflow= or hf_path="):
            flow().forecast("m")

    def test_duplicate_forecast_rejected(self):
        b = flow().forecast("revert", mlflow="models:/revert/3")
        with pytest.raises(ConfigurationError, match="already registered"):
            b.forecast("revert", mlflow="models:/revert/4")

    def test_conflicting_version_rejected(self):
        with pytest.raises(ConfigurationError, match="Conflicting version"):
            flow().forecast("revert", mlflow="models:/revert/3", version="4")

    def test_latest_rejected_without_env(self, monkeypatch):
        monkeypatch.delenv("SF_ALLOW_LATEST", raising=False)
        with pytest.raises(ValueError):
            flow().forecast("revert", mlflow="models:/revert/latest")

    def test_latest_allowed_with_env(self, monkeypatch):
        monkeypatch.setenv("SF_ALLOW_LATEST", "1")
        b = flow().forecast("revert", mlflow="models:/revert/latest")
        assert str(b._named_forecasts["revert"].version) == "latest"


class TestForecastWindowContract:
    def test_detector_forecasts_requires_window(self):
        b = flow().forecast("revert", mlflow="models:/revert/3")
        with pytest.raises(ConfigurationError, match="positive forecast_window"):
            b.detector("example/sma_cross", forecasts=["revert"])

    def test_window_without_forecasts_rejected(self):
        with pytest.raises(ConfigurationError, match="without forecasts"):
            flow().detector("example/sma_cross", forecast_window=10)

    def test_detector_records_consumer_and_window(self):
        b = (
            flow()
            .forecast("revert", mlflow="models:/revert/3")
            .detector("example/sma_cross", name="trend", forecasts=["revert"], forecast_window=5)
        )
        assert b._forecast_consumers["trend"] == ["revert"]
        assert b._forecast_windows["trend"] == 5

    def test_validator_and_exit_consumers(self):
        b = (
            flow()
            .forecast("revert", mlflow="models:/revert/3")
            .validator("validator/lightgbm", name="meta", forecasts=["revert"], forecast_window=3)
            .exit(tp=0.02, forecasts=["revert"], forecast_window=7)
        )
        assert b._forecast_consumers["validator:meta"] == ["revert"]
        assert b._forecast_consumers["exit"] == ["revert"]
        assert b._forecast_windows["exit"] == 7


class TestForecastRefValidation:
    def test_unknown_forecast_ref_caught_at_run(self):
        # Detector references a forecast that was never declared → caught at run().
        b = flow().detector("example/sma_cross", name="trend")
        # inject a dangling reference (as if .forecast() was forgotten)
        b._forecast_consumers["trend"] = ["ghost"]
        b._forecast_windows["trend"] = 5
        with pytest.raises(ConfigurationError, match="unknown forecast"):
            b._validate_forecast_refs()

    def test_valid_refs_pass(self):
        b = (
            flow()
            .forecast("revert", mlflow="models:/revert/3")
            .detector("example/sma_cross", name="trend", forecasts=["revert"], forecast_window=5)
        )
        b._validate_forecast_refs()  # no raise
