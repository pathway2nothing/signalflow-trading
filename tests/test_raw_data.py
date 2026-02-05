"""Tests for signalflow.core.containers.raw_data.RawData."""

import pytest
import polars as pl
from datetime import datetime

from signalflow.core.containers.raw_data import RawData


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
        with pytest.raises(TypeError, match="not a polars.DataFrame"):
            rd.get("bad")
