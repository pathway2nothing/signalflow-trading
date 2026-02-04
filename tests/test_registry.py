"""Tests for signalflow.core.registry.SignalFlowRegistry."""

import pytest

from signalflow.core.registry import SignalFlowRegistry
from signalflow.core.enums import SfComponentType


class DummyDetector:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyDetectorV2:
    pass


class TestRegistryRegister:
    def test_register_and_get(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.DETECTOR, "my_det", DummyDetector)
        cls = reg.get(SfComponentType.DETECTOR, "my_det")
        assert cls is DummyDetector

    def test_case_insensitive(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.DETECTOR, "MyDet", DummyDetector)
        cls = reg.get(SfComponentType.DETECTOR, "mydet")
        assert cls is DummyDetector

    def test_duplicate_raises(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.DETECTOR, "dup", DummyDetector)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(SfComponentType.DETECTOR, "dup", DummyDetector)

    def test_override(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.DETECTOR, "det", DummyDetector)
        reg.register(SfComponentType.DETECTOR, "det", DummyDetectorV2, override=True)
        cls = reg.get(SfComponentType.DETECTOR, "det")
        assert cls is DummyDetectorV2

    def test_empty_name_raises(self):
        reg = SignalFlowRegistry()
        with pytest.raises(ValueError, match="non-empty string"):
            reg.register(SfComponentType.DETECTOR, "", DummyDetector)

    def test_whitespace_name_raises(self):
        reg = SignalFlowRegistry()
        with pytest.raises(ValueError, match="non-empty string"):
            reg.register(SfComponentType.DETECTOR, "   ", DummyDetector)


class TestRegistryGet:
    def test_missing_raises_keyerror(self):
        reg = SignalFlowRegistry()
        with pytest.raises(KeyError, match="Component not found"):
            reg.get(SfComponentType.DETECTOR, "nonexistent")

    def test_error_shows_available(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.DETECTOR, "alpha", DummyDetector)
        with pytest.raises(KeyError, match="alpha"):
            reg.get(SfComponentType.DETECTOR, "beta")


class TestRegistryCreate:
    def test_create_with_kwargs(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.DETECTOR, "det", DummyDetector)
        obj = reg.create(SfComponentType.DETECTOR, "det", foo=42, bar="x")
        assert isinstance(obj, DummyDetector)
        assert obj.kwargs == {"foo": 42, "bar": "x"}

    def test_create_missing_raises(self):
        reg = SignalFlowRegistry()
        with pytest.raises(KeyError):
            reg.create(SfComponentType.DETECTOR, "nope")


class TestRegistryList:
    def test_list_empty(self):
        reg = SignalFlowRegistry()
        assert reg.list(SfComponentType.DETECTOR) == []

    def test_list_sorted(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.DETECTOR, "beta", DummyDetector)
        reg.register(SfComponentType.DETECTOR, "alpha", DummyDetectorV2)
        assert reg.list(SfComponentType.DETECTOR) == ["alpha", "beta"]


class TestRegistrySnapshot:
    def test_snapshot(self):
        reg = SignalFlowRegistry()
        reg.register(SfComponentType.DETECTOR, "det1", DummyDetector)
        reg.register(SfComponentType.FEATURE, "feat1", DummyDetector)
        snap = reg.snapshot()
        assert SfComponentType.DETECTOR.value in snap
        assert "det1" in snap[SfComponentType.DETECTOR.value]
        assert SfComponentType.FEATURE.value in snap
        assert "feat1" in snap[SfComponentType.FEATURE.value]
