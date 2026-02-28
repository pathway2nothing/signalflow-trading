"""Tests for signalflow.core.containers.raw_data.RawData."""

import warnings
from datetime import datetime

import polars as pl
import pytest

from signalflow.core.containers.raw_data import DataTypeAccessor, RawData


class TestRawDataCreation:
    def test_create_empty(self):
        rd = RawData(
            datetime_start=datetime(2024, 1, 1),
            datetime_end=datetime(2024, 1, 2),
        )
        assert rd.pairs == []
        assert rd.data == {}

    def test_create_with_data(self, raw_data):
        assert raw_data.pairs == ["BTCUSDT", "ETHUSDT"]
        assert "spot" in raw_data.data

    def test_frozen(self, raw_data):
        with pytest.raises(AttributeError):
            raw_data.pairs = ["OTHER"]


class TestRawDataAccess:
    def test_get_existing(self, raw_data):
        df = raw_data.get("spot")
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()

    def test_get_missing_returns_empty(self, raw_data):
        df = raw_data.get("nonexistent")
        assert isinstance(df, pl.DataFrame)
        assert df.is_empty()

    def test_getitem(self, raw_data):
        df = raw_data["spot"]
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()

    def test_contains(self, raw_data):
        assert "spot" in raw_data
        assert "nonexistent" not in raw_data

    def test_keys(self, raw_data):
        keys = list(raw_data.keys())
        assert "spot" in keys

    def test_items(self, raw_data):
        items = list(raw_data.items())
        assert len(items) == 1
        key, df = items[0]
        assert key == "spot"
        assert isinstance(df, pl.DataFrame)

    def test_values(self, raw_data):
        vals = list(raw_data.values())
        assert len(vals) == 1
        assert isinstance(vals[0], pl.DataFrame)

    def test_get_wrong_type_raises(self):
        rd = RawData(
            datetime_start=datetime(2024, 1, 1),
            datetime_end=datetime(2024, 1, 2),
            data={"bad": "not_a_dataframe"},  # type: ignore
        )
        with pytest.raises(TypeError, match="invalid type"):
            rd.get("bad")


# ---------------------------------------------------------------------------
# Multi-source (nested structure) tests
# ---------------------------------------------------------------------------


@pytest.fixture
def nested_raw_data():
    """RawData with nested multi-source structure."""
    base = datetime(2024, 1, 1)

    def make_df(source: str, multiplier: float):
        return pl.DataFrame(
            {
                "pair": ["BTCUSDT"] * 5,
                "timestamp": [base.replace(hour=i) for i in range(5)],
                "open_interest": [1000.0 * multiplier + i for i in range(5)],
            }
        )

    return RawData(
        datetime_start=base,
        datetime_end=base.replace(hour=4),
        pairs=["BTCUSDT"],
        data={
            "perpetual": {
                "binance": make_df("binance", 1.0),
                "okx": make_df("okx", 0.8),
                "bybit": make_df("bybit", 0.5),
            }
        },
        default_source="binance",
    )


class TestDataTypeAccessor:
    def test_accessor_sources(self, nested_raw_data):
        accessor = nested_raw_data.perpetual
        assert isinstance(accessor, DataTypeAccessor)
        assert accessor.sources == ["binance", "okx", "bybit"]

    def test_accessor_getattr(self, nested_raw_data):
        df = nested_raw_data.perpetual.binance
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()
        assert df["open_interest"][0] == 1000.0

    def test_accessor_to_polars_warns(self, nested_raw_data):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = nested_raw_data.perpetual.to_polars()
            assert len(w) == 1
            assert "default source 'binance'" in str(w[0].message)
            assert isinstance(df, pl.DataFrame)

    def test_accessor_contains(self, nested_raw_data):
        assert "binance" in nested_raw_data.perpetual
        assert "unknown" not in nested_raw_data.perpetual

    def test_accessor_len(self, nested_raw_data):
        assert len(nested_raw_data.perpetual) == 3

    def test_accessor_iter(self, nested_raw_data):
        items = list(nested_raw_data.perpetual)
        assert len(items) == 3
        source, df = items[0]
        assert source == "binance"
        assert isinstance(df, pl.DataFrame)

    def test_accessor_missing_source_raises(self, nested_raw_data):
        with pytest.raises(AttributeError, match="No source 'unknown'"):
            _ = nested_raw_data.perpetual.unknown


class TestNestedRawDataAccess:
    def test_get_with_source(self, nested_raw_data):
        df = nested_raw_data.get("perpetual", source="okx")
        assert isinstance(df, pl.DataFrame)
        assert df["open_interest"][0] == 800.0  # 1000 * 0.8

    def test_get_without_source_warns(self, nested_raw_data):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = nested_raw_data.get("perpetual")
            assert len(w) == 1
            assert "default source" in str(w[0].message)
            assert isinstance(df, pl.DataFrame)

    def test_get_invalid_source_raises(self, nested_raw_data):
        with pytest.raises(KeyError, match="Source 'unknown' not found"):
            nested_raw_data.get("perpetual", source="unknown")

    def test_getitem_tuple(self, nested_raw_data):
        df = nested_raw_data["perpetual", "bybit"]
        assert isinstance(df, pl.DataFrame)
        assert df["open_interest"][0] == 500.0  # 1000 * 0.5

    def test_sources_method(self, nested_raw_data):
        sources = nested_raw_data.sources("perpetual")
        assert sources == ["binance", "okx", "bybit"]

    def test_sources_missing_key_raises(self, nested_raw_data):
        with pytest.raises(KeyError, match="No data type 'spot'"):
            nested_raw_data.sources("spot")

    def test_attribute_access_missing_raises(self, nested_raw_data):
        with pytest.raises(AttributeError, match="No data type 'spot'"):
            _ = nested_raw_data.spot


class TestBackwardsCompatibility:
    """Ensure flat structure still works with new code."""

    def test_flat_getitem(self, raw_data):
        # raw_data fixture has flat structure
        df = raw_data["spot"]
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()

    def test_flat_get(self, raw_data):
        df = raw_data.get("spot")
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()

    def test_flat_attribute_access(self, raw_data):
        # Flat data wraps as single-source accessor
        accessor = raw_data.spot
        assert isinstance(accessor, DataTypeAccessor)
        assert len(accessor.sources) == 1
