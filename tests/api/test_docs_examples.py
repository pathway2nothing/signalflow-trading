"""Doc-example smoke tests — keep documentation snippets in sync with the real API.

These mirror the v2 examples added to docs/ (model-integration, glossary, models API).
If a public signature changes and a doc example would break, one of these fails.
"""

from __future__ import annotations

import pytest

from signalflow.api.exceptions import ConfigurationError
from signalflow.api.flow import flow
from signalflow.models import CachingModelRegistry, ModelRef


class TestModelsDocExamples:
    def test_modelref_parse_examples(self):
        """docs/api/models.md: ModelRef.parse for both URI and shorthand forms."""
        a = ModelRef.parse("models:/revert/3")
        b = ModelRef.parse("revert@3")
        assert a.name == "revert" and str(a.version) == "3" and a.uri == "models:/revert/3"
        assert b.name == "revert" and str(b.version) == "3"

    def test_modelref_latest_rejected(self):
        """docs: latest is rejected (parity) without SF_ALLOW_LATEST."""
        with pytest.raises(ValueError):
            ModelRef("revert", "latest")

    def test_caching_registry_lazy(self):
        """docs/guide/model-integration.md: registry resolves lazily and caches."""
        calls = {"n": 0}

        class _FakeResolver:
            def resolve(self, ref: ModelRef):
                calls["n"] += 1
                return f"weights::{ref.uri}"

        reg = CachingModelRegistry(_FakeResolver())
        ref = ModelRef.parse("models:/revert/3")
        assert reg.has(ref) is False or reg.has(ref) is True  # has() must not resolve
        before = calls["n"]
        _ = reg.get(ref)
        _ = reg.get(ref)
        assert calls["n"] == before + 1  # resolved once, then cached


class TestFlowForecastDocExample:
    def test_model_integration_flow_builds(self):
        """docs/guide/model-integration.md: the headline forecast-workflow flow builds."""
        builder = (
            flow()
            .forecast("revert", mlflow="models:/revert/3")
            .detector("example/sma_cross", forecasts=["revert"], forecast_window=30)
            .validator("validator/lightgbm", forecasts=["revert"], forecast_window=60)
            .exit(tp=0.02, sl=0.01)
        )
        assert "revert" in builder._named_forecasts
        builder._validate_forecast_refs()  # references resolve

    def test_glossary_contract_errors(self):
        """docs/glossary.md warmup/forecast contract: misuse raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            flow().detector("example/sma_cross", forecast_window=10)  # window w/o forecasts
        with pytest.raises(ConfigurationError):
            flow().forecast("r", mlflow="models:/r/3").detector(
                "example/sma_cross",
                forecasts=["r"],
            )  # forecasts w/o window
