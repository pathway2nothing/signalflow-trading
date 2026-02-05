"""Tests for signalflow.core.containers.raw_data_view.RawDataView."""

import pandas as pd
import polars as pl
import pytest
from datetime import datetime

from signalflow.core.containers.raw_data_view import RawDataView
from signalflow.core.enums import DataFrameType


class TestRawDataViewPolars:
    def test_to_polars(self, raw_data):
        view = RawDataView(raw=raw_data)
        df = view.to_polars("spot")
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()

    def test_to_polars_missing_key(self, raw_data):
        view = RawDataView(raw=raw_data)
        df = view.to_polars("missing")
        assert df.is_empty()


class TestRawDataViewPandas:
    def test_to_pandas(self, raw_data):
        view = RawDataView(raw=raw_data)
        df = view.to_pandas("spot")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_to_pandas_sorted(self, raw_data):
        view = RawDataView(raw=raw_data)
        df = view.to_pandas("spot")
        pairs = df["pair"].tolist()
        timestamps = df["timestamp"].tolist()
        # Should be sorted by pair, then timestamp
        for i in range(1, len(df)):
            if pairs[i] == pairs[i - 1]:
                assert timestamps[i] >= timestamps[i - 1]

    def test_to_pandas_empty(self, raw_data):
        view = RawDataView(raw=raw_data)
        df = view.to_pandas("missing")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_cache_pandas(self, raw_data):
        view = RawDataView(raw=raw_data, cache_pandas=True)
        df1 = view.to_pandas("spot")
        assert "spot" in view._pandas_cache
        df2 = view.to_pandas("spot")
        assert not df2.empty


class TestRawDataViewGetData:
    def test_get_data_polars(self, raw_data):
        view = RawDataView(raw=raw_data)
        df = view.get_data("spot", DataFrameType.POLARS)
        assert isinstance(df, pl.DataFrame)

    def test_get_data_pandas(self, raw_data):
        view = RawDataView(raw=raw_data)
        df = view.get_data("spot", DataFrameType.PANDAS)
        assert isinstance(df, pd.DataFrame)

    def test_get_data_invalid_type(self, raw_data):
        view = RawDataView(raw=raw_data)
        with pytest.raises(ValueError, match="Unsupported df_type"):
            view.get_data("spot", "invalid")  # type: ignore
