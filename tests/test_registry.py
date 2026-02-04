"""Tests for signalflow.core.registry.SignalFlowRegistry."""

import pytest

from signalflow.core.registry import SignalFlowRegistry
from signalflow.core.enums import SfComponentType, RawDataType


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


class TestRawDataTypeRegistry:
    """Tests for extensible raw data type registration."""

    def test_builtin_spot_columns(self):
        reg = SignalFlowRegistry()
        cols = reg.get_raw_data_columns("spot")
        assert cols == {"pair", "timestamp", "open", "high", "low", "close", "volume"}

    def test_builtin_futures_columns(self):
        reg = SignalFlowRegistry()
        cols = reg.get_raw_data_columns("futures")
        assert "open_interest" in cols
        assert "close" in cols

    def test_builtin_perpetual_columns(self):
        reg = SignalFlowRegistry()
        cols = reg.get_raw_data_columns("perpetual")
        assert "funding_rate" in cols
        assert "open_interest" in cols

    def test_register_custom_type(self):
        reg = SignalFlowRegistry()
        reg.register_raw_data_type(
            "lob",
            columns=["pair", "timestamp", "bid", "ask", "depth"],
        )
        cols = reg.get_raw_data_columns("lob")
        assert cols == {"pair", "timestamp", "bid", "ask", "depth"}

    def test_case_insensitive(self):
        reg = SignalFlowRegistry()
        reg.register_raw_data_type("MyType", columns=["a", "b"])
        cols = reg.get_raw_data_columns("mytype")
        assert cols == {"a", "b"}

    def test_duplicate_raises(self):
        reg = SignalFlowRegistry()
        with pytest.raises(ValueError, match="already registered"):
            reg.register_raw_data_type("spot", columns=["x"])

    def test_override_builtin(self):
        reg = SignalFlowRegistry()
        reg.register_raw_data_type(
            "spot",
            columns=["pair", "timestamp", "price"],
            override=True,
        )
        cols = reg.get_raw_data_columns("spot")
        assert cols == {"pair", "timestamp", "price"}

    def test_empty_name_raises(self):
        reg = SignalFlowRegistry()
        with pytest.raises(ValueError, match="non-empty string"):
            reg.register_raw_data_type("", columns=["a"])

    def test_empty_columns_raises(self):
        reg = SignalFlowRegistry()
        with pytest.raises(ValueError, match="non-empty collection"):
            reg.register_raw_data_type("empty", columns=[])

    def test_missing_type_raises(self):
        reg = SignalFlowRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get_raw_data_columns("nonexistent")

    def test_error_shows_available(self):
        reg = SignalFlowRegistry()
        with pytest.raises(KeyError, match="spot"):
            reg.get_raw_data_columns("nonexistent")

    def test_list_raw_data_types(self):
        reg = SignalFlowRegistry()
        types = reg.list_raw_data_types()
        assert "spot" in types
        assert "futures" in types
        assert "perpetual" in types
        assert types == sorted(types)

    def test_list_includes_custom(self):
        reg = SignalFlowRegistry()
        reg.register_raw_data_type("tick", columns=["pair", "timestamp", "price", "qty"])
        types = reg.list_raw_data_types()
        assert "tick" in types

    def test_get_returns_copy(self):
        """Mutating returned set should not affect registry."""
        reg = SignalFlowRegistry()
        cols = reg.get_raw_data_columns("spot")
        cols.add("extra_column")
        assert "extra_column" not in reg.get_raw_data_columns("spot")

    def test_accepts_raw_data_type_enum(self):
        """get_raw_data_columns should accept RawDataType enum members."""
        reg = SignalFlowRegistry()
        cols = reg.get_raw_data_columns(RawDataType.SPOT)
        assert "close" in cols

    def test_enum_columns_delegates_to_registry(self):
        """RawDataType.SPOT.columns should return same as registry."""
        cols_enum = RawDataType.SPOT.columns
        assert cols_enum == {"pair", "timestamp", "open", "high", "low", "close", "volume"}
