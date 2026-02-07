"""Tests for OffsetFeature."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.feature.offset_feature import OffsetFeature
from signalflow.core.registry import default_registry
from signalflow.core import SfComponentType


TS = datetime(2024, 1, 1)


def _make_df(n=100, pair="BTCUSDT"):
    rows = []
    for i in range(n):
        rows.append(
            {
                "pair": pair,
                "timestamp": TS + timedelta(minutes=i),
                "open": 100.0 + i * 0.1,
                "high": 101.0 + i * 0.1,
                "low": 99.0 + i * 0.1,
                "close": 100.0 + i * 0.1,
                "volume": 1000.0,
            }
        )
    return pl.DataFrame(rows)


class TestOffsetFeatureValidation:
    def test_no_feature_name_raises(self):
        with pytest.raises(ValueError, match="feature_name"):
            OffsetFeature(feature_name=None)

    def test_unknown_feature_raises(self):
        with pytest.raises(Exception):
            OffsetFeature(feature_name="nonexistent_feature_12345")


class TestOffsetFeatureBasic:
    def test_output_cols(self):
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=15)
        cols = f.output_cols()
        assert "offset" in cols

    def test_required_cols(self):
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=15)
        req = f.required_cols()
        assert "open" in req
        assert "close" in req
        assert "timestamp" in req

    def test_prefix_default(self):
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=15)
        assert f.prefix == "15m_"

    def test_prefix_custom(self):
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=30, prefix="custom_")
        assert f.prefix == "custom_"

    def test_to_dict(self):
        f = OffsetFeature(feature_name="example/rsi", feature_params={"period": 14}, window=15, prefix="pfx_")
        d = f.to_dict()
        assert d["feature_name"] == "example/rsi"
        assert d["window"] == 15
        assert d["prefix"] == "pfx_"

    def test_from_dict(self):
        data = {
            "feature_name": "example/rsi",
            "feature_params": {"period": 14},
            "window": 15,
            "prefix": "pfx_",
        }
        f = OffsetFeature.from_dict(data)
        assert f.window == 15
        assert f.prefix == "pfx_"
