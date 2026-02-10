"""Tests for signalflow.core.enums."""

from signalflow.core.enums import (
    SignalType,
    SignalCategory,
    PositionType,
    SfComponentType,
    DataFrameType,
    RawDataType,
)


class TestSignalType:
    def test_values(self):
        assert SignalType.NONE.value == "none"
        assert SignalType.RISE.value == "rise"
        assert SignalType.FALL.value == "fall"
        assert SignalType.FLAT.value == "flat"

    def test_is_str_enum(self):
        assert isinstance(SignalType.RISE, str)
        assert SignalType.RISE == "rise"

    def test_members_count(self):
        assert len(SignalType) == 4


class TestSignalCategory:
    def test_values(self):
        assert SignalCategory.PRICE_DIRECTION.value == "price_direction"
        assert SignalCategory.PRICE_STRUCTURE.value == "price_structure"
        assert SignalCategory.TREND_MOMENTUM.value == "trend_momentum"
        assert SignalCategory.VOLATILITY.value == "volatility"
        assert SignalCategory.VOLUME_LIQUIDITY.value == "volume_liquidity"
        assert SignalCategory.MARKET_WIDE.value == "market_wide"
        assert SignalCategory.ANOMALY.value == "anomaly"

    def test_is_str_enum(self):
        assert isinstance(SignalCategory.PRICE_DIRECTION, str)
        assert SignalCategory.PRICE_DIRECTION == "price_direction"

    def test_members_count(self):
        assert len(SignalCategory) == 7


class TestPositionType:
    def test_values(self):
        assert PositionType.LONG.value == "long"
        assert PositionType.SHORT.value == "short"

    def test_is_str_enum(self):
        assert isinstance(PositionType.LONG, str)


class TestSfComponentType:
    def test_detector_value(self):
        assert SfComponentType.DETECTOR.value == "signals/detector"

    def test_feature_value(self):
        assert SfComponentType.FEATURE.value == "feature"

    def test_is_str_enum(self):
        assert isinstance(SfComponentType.DETECTOR, str)


class TestDataFrameType:
    def test_values(self):
        assert DataFrameType.POLARS.value == "polars"
        assert DataFrameType.PANDAS.value == "pandas"


class TestRawDataType:
    def test_values(self):
        assert RawDataType.SPOT.value == "spot"
        assert RawDataType.FUTURES.value == "futures"
        assert RawDataType.PERPETUAL.value == "perpetual"

    def test_columns_spot(self):
        cols = RawDataType.SPOT.columns
        assert "pair" in cols
        assert "timestamp" in cols
        assert "open" in cols
        assert "high" in cols
        assert "low" in cols
        assert "close" in cols
        assert "volume" in cols

    def test_columns_futures_has_open_interest(self):
        cols = RawDataType.FUTURES.columns
        assert "open_interest" in cols

    def test_columns_perpetual_has_funding_rate(self):
        cols = RawDataType.PERPETUAL.columns
        assert "funding_rate" in cols
        assert "open_interest" in cols
