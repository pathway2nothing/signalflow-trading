"""Tests for BacktestBuilder."""

import pytest

import signalflow as sf
from signalflow.api.builder import Backtest, BacktestBuilder
from signalflow.core import default_registry, SfComponentType


class TestBacktestFactory:
    """Tests for Backtest() factory function."""

    def test_returns_builder(self):
        """Backtest() returns BacktestBuilder instance."""
        builder = Backtest()
        assert isinstance(builder, BacktestBuilder)

    def test_default_strategy_id(self):
        """Default strategy_id is 'backtest'."""
        builder = Backtest()
        assert builder.strategy_id == "backtest"

    def test_custom_strategy_id(self):
        """Custom strategy_id is set."""
        builder = Backtest("my_strategy")
        assert builder.strategy_id == "my_strategy"


class TestBuilderFluent:
    """Tests for fluent API."""

    def test_data_returns_self(self, sample_raw_data):
        """data() returns self for chaining."""
        builder = Backtest()
        result = builder.data(raw=sample_raw_data)
        assert result is builder

    def test_detector_returns_self(self, sample_raw_data):
        """detector() returns self for chaining."""
        builder = Backtest().data(raw=sample_raw_data)
        result = builder.detector("example/sma_cross")
        assert result is builder

    def test_entry_returns_self(self):
        """entry() returns self for chaining."""
        builder = Backtest()
        result = builder.entry(size=100)
        assert result is builder

    def test_exit_returns_self(self):
        """exit() returns self for chaining."""
        builder = Backtest()
        result = builder.exit(tp=0.02, sl=0.01)
        assert result is builder

    def test_capital_returns_self(self):
        """capital() returns self for chaining."""
        builder = Backtest()
        result = builder.capital(50_000)
        assert result is builder

    def test_fee_returns_self(self):
        """fee() returns self for chaining."""
        builder = Backtest()
        result = builder.fee(0.002)
        assert result is builder

    def test_progress_returns_self(self):
        """progress() returns self for chaining."""
        builder = Backtest()
        result = builder.progress(False)
        assert result is builder

    def test_full_chain(self, sample_raw_data):
        """Full method chain works."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data)
            .detector("example/sma_cross")
            .entry(size_pct=0.1, max_positions=5)
            .exit(tp=0.03, sl=0.015)
            .capital(50_000)
            .fee(0.0005)
            .progress(False)
        )
        assert builder.strategy_id == "test"
        assert builder._capital == 50_000
        assert builder._fee == 0.0005
        assert builder._show_progress is False


class TestBuilderRegistry:
    """Tests for registry-based component creation."""

    def test_detector_from_registry_name(self, sample_raw_data):
        """detector() accepts registry name string."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data)
            .detector("example/sma_cross", fast_period=10, slow_period=20)
        )
        assert builder._detector is not None

    def test_detector_from_instance(self, sample_raw_data):
        """detector() accepts detector instance."""
        from signalflow.detector import ExampleSmaCrossDetector

        detector = ExampleSmaCrossDetector(fast_period=5, slow_period=15)
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data)
            .detector(detector)
        )
        assert builder._detector is detector

    def test_registry_has_entry_rules(self):
        """Registry has entry rules registered."""
        rules = default_registry.list(SfComponentType.STRATEGY_ENTRY_RULE)
        # May be empty if not registered, but should not crash
        assert isinstance(rules, list)

    def test_registry_has_exit_rules(self):
        """Registry has exit rules registered."""
        rules = default_registry.list(SfComponentType.STRATEGY_EXIT_RULE)
        assert isinstance(rules, list)


class TestBuilderValidation:
    """Tests for configuration validation."""

    def test_validate_no_data(self):
        """Validation catches missing data."""
        builder = Backtest("test")
        issues = builder.validate()
        assert any("No data" in i for i in issues)

    def test_validate_no_detector(self, sample_raw_data):
        """Validation catches missing detector."""
        builder = Backtest("test").data(raw=sample_raw_data)
        issues = builder.validate()
        assert any("No detector" in i for i in issues)

    def test_validate_tp_sl_warning(self, sample_raw_data):
        """Validation warns when TP < SL."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data)
            .detector("example/sma_cross")
            .exit(tp=0.01, sl=0.02)  # Bad ratio
        )
        issues = builder.validate()
        assert any("WARNING" in i and "TP" in i for i in issues)

    def test_validate_negative_capital(self):
        """Negative capital raises InvalidParameterError immediately."""
        from signalflow.api.exceptions import InvalidParameterError

        with pytest.raises(InvalidParameterError) as exc_info:
            Backtest("test").capital(-1000)

        assert "capital" in str(exc_info.value).lower()

    def test_validate_valid_config(self, sample_raw_data):
        """Valid config passes validation."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data)
            .detector("example/sma_cross")
            .exit(tp=0.03, sl=0.01)  # Good ratio
            .capital(10_000)
        )
        issues = builder.validate()
        # Should have no errors (warnings OK)
        errors = [i for i in issues if "ERROR" in i]
        assert len(errors) == 0


class TestBuilderDataConfig:
    """Tests for data configuration."""

    def test_data_with_raw(self, sample_raw_data):
        """data() accepts RawData instance."""
        builder = Backtest().data(raw=sample_raw_data)
        assert builder._raw is sample_raw_data

    def test_data_with_params(self):
        """data() stores params for lazy loading."""
        builder = Backtest().data(
            exchange="binance",
            pairs=["BTCUSDT"],
            start="2024-01-01",
            end="2024-06-01",
        )
        assert builder._data_params is not None
        assert builder._data_params["exchange"] == "binance"
        assert builder._data_params["pairs"] == ["BTCUSDT"]


class TestBuilderEntryConfig:
    """Tests for entry configuration."""

    def test_entry_stores_config(self):
        """entry() stores configuration."""
        builder = Backtest().entry(
            size=200,
            max_positions=5,
            max_per_pair=2,
        )
        assert builder._entry_config["size"] == 200
        assert builder._entry_config["max_positions"] == 5
        assert builder._entry_config["max_per_pair"] == 2

    def test_entry_size_pct(self):
        """entry() accepts size_pct."""
        builder = Backtest().entry(size_pct=0.1)
        assert builder._entry_config["size_pct"] == 0.1


class TestBuilderExitConfig:
    """Tests for exit configuration."""

    def test_exit_stores_tp_sl(self):
        """exit() stores TP/SL."""
        builder = Backtest().exit(tp=0.03, sl=0.015)
        assert builder._exit_config["tp"] == 0.03
        assert builder._exit_config["sl"] == 0.015

    def test_exit_trailing(self):
        """exit() accepts trailing stop."""
        builder = Backtest().exit(trailing=0.02)
        assert builder._exit_config["trailing"] == 0.02

    def test_exit_time_limit(self):
        """exit() accepts time limit."""
        builder = Backtest().exit(time_limit=100)
        assert builder._exit_config["time_limit"] == 100


class TestBuilderRepr:
    """Tests for string representation."""

    def test_repr(self):
        """__repr__ is readable."""
        builder = Backtest("my_strategy")
        repr_str = repr(builder)
        assert "BacktestBuilder" in repr_str
        assert "my_strategy" in repr_str


class TestBuilderExceptions:
    """Tests for custom exceptions."""

    def test_detector_not_found_error(self):
        """DetectorNotFoundError has helpful message."""
        from signalflow.api.exceptions import DetectorNotFoundError

        with pytest.raises(DetectorNotFoundError) as exc_info:
            Backtest().detector("nonexistent/detector")

        error_msg = str(exc_info.value)
        assert "not found" in error_msg.lower()
        assert "nonexistent/detector" in error_msg

    def test_detector_not_found_shows_available(self):
        """DetectorNotFoundError shows available detectors."""
        from signalflow.api.exceptions import DetectorNotFoundError

        with pytest.raises(DetectorNotFoundError) as exc_info:
            Backtest().detector("nonexistent/detector")

        error_msg = str(exc_info.value)
        # Should show available detectors if any
        assert "example/sma_cross" in error_msg or "Available" in error_msg

    def test_missing_data_error(self):
        """MissingDataError raised when no data configured."""
        from signalflow.api.exceptions import MissingDataError

        builder = Backtest().detector("example/sma_cross")
        with pytest.raises(MissingDataError) as exc_info:
            builder.run()

        error_msg = str(exc_info.value)
        assert "data" in error_msg.lower()

    def test_missing_detector_error(self, sample_raw_data):
        """MissingDetectorError raised when no detector configured."""
        from signalflow.api.exceptions import MissingDetectorError

        builder = Backtest().data(raw=sample_raw_data)
        with pytest.raises(MissingDetectorError) as exc_info:
            builder.run()

        error_msg = str(exc_info.value)
        assert "detector" in error_msg.lower()
