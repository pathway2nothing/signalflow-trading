"""Tests for BacktestBuilder."""

import pytest
from datetime import datetime, timedelta

import polars as pl

import signalflow as sf
from signalflow.api.builder import Backtest, BacktestBuilder
from signalflow.core import default_registry, SfComponentType, RawData, Signals


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
            Backtest("test").data(raw=sample_raw_data).detector("example/sma_cross", fast_period=10, slow_period=20)
        )
        assert builder._detector is not None

    def test_detector_from_instance(self, sample_raw_data):
        """detector() accepts detector instance."""
        from signalflow.detector import ExampleSmaCrossDetector

        detector = ExampleSmaCrossDetector(fast_period=5, slow_period=15)
        builder = Backtest("test").data(raw=sample_raw_data).detector(detector)
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
            Backtest("test").data(raw=sample_raw_data).detector("example/sma_cross").exit(tp=0.01, sl=0.02)  # Bad ratio
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

    def test_duplicate_component_name_error(self):
        """DuplicateComponentNameError has helpful message."""
        from signalflow.api.exceptions import DuplicateComponentNameError

        err = DuplicateComponentNameError("detector", "trend")
        assert "Duplicate" in str(err)
        assert "trend" in str(err)

    def test_validator_not_found_error(self):
        """ValidatorNotFoundError raised for unknown validator name."""
        from signalflow.api.exceptions import ValidatorNotFoundError

        with pytest.raises(ValidatorNotFoundError) as exc_info:
            Backtest().validator("nonexistent/validator")

        assert "not found" in str(exc_info.value).lower()


# ===========================================================================
# Multi-Component Tests
# ===========================================================================


class TestBuilderMultiData:
    """Tests for named data sources."""

    def test_data_with_name(self, sample_raw_data):
        """data(name=...) stores in _named_data."""
        builder = Backtest().data(raw=sample_raw_data, name="1m")
        assert "1m" in builder._named_data
        assert builder._named_data["1m"] is sample_raw_data

    def test_multiple_named_data(self, sample_raw_data):
        """Multiple named data sources stored correctly."""
        raw2 = RawData(
            datetime_start=datetime(2024, 1, 1),
            datetime_end=datetime(2024, 1, 5),
            pairs=["BTCUSDT"],
            data={"spot": sample_raw_data.get("spot")},
        )
        builder = Backtest().data(raw=sample_raw_data, name="1m").data(raw=raw2, name="1h")
        assert len(builder._named_data) == 2
        assert "1m" in builder._named_data
        assert "1h" in builder._named_data

    def test_duplicate_data_name_raises(self, sample_raw_data):
        """Duplicate data name raises DuplicateComponentNameError."""
        from signalflow.api.exceptions import DuplicateComponentNameError

        with pytest.raises(DuplicateComponentNameError):
            Backtest().data(raw=sample_raw_data, name="x").data(raw=sample_raw_data, name="x")

    def test_datas_dict(self, sample_raw_data):
        """datas() accepts dict."""
        builder = Backtest().datas({"1m": sample_raw_data})
        assert "1m" in builder._named_data

    def test_datas_list(self, sample_raw_data):
        """datas() accepts list of tuples."""
        builder = Backtest().datas([("1m", sample_raw_data)])
        assert "1m" in builder._named_data

    def test_named_data_params(self):
        """data(name=...) stores params dict for lazy loading."""
        builder = Backtest().data(
            name="binance",
            exchange="binance",
            pairs=["BTCUSDT"],
            start="2024-01-01",
        )
        assert "binance" in builder._named_data
        assert isinstance(builder._named_data["binance"], dict)
        assert builder._named_data["binance"]["exchange"] == "binance"


class TestBuilderMultiDetector:
    """Tests for named detectors."""

    def test_detector_auto_names(self, sample_raw_data):
        """First detector auto-named 'default', second 'detector_1'."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data)
            .detector("example/sma_cross", fast_period=10, slow_period=20)
            .detector("example/sma_cross", fast_period=5, slow_period=15)
        )
        assert "default" in builder._named_detectors
        assert "detector_1" in builder._named_detectors

    def test_detector_explicit_name(self, sample_raw_data):
        """detector(name=...) uses explicit name."""
        builder = Backtest("test").data(raw=sample_raw_data).detector("example/sma_cross", name="trend")
        assert "trend" in builder._named_detectors

    def test_detector_data_source(self, sample_raw_data):
        """detector(data_source=...) records data source mapping."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data, name="1h")
            .detector("example/sma_cross", name="trend", data_source="1h")
        )
        assert builder._detector_data_sources["trend"] == "1h"

    def test_duplicate_detector_name_raises(self, sample_raw_data):
        """Duplicate detector name raises DuplicateComponentNameError."""
        from signalflow.api.exceptions import DuplicateComponentNameError

        with pytest.raises(DuplicateComponentNameError):
            (
                Backtest("test")
                .data(raw=sample_raw_data)
                .detector("example/sma_cross", name="trend")
                .detector("example/sma_cross", name="trend")
            )

    def test_detectors_list_tuples(self, sample_raw_data):
        """detectors() accepts list of (name, detector) tuples."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data)
            .detectors(
                [
                    ("trend", "example/sma_cross"),
                    ("volume", "example/sma_cross"),
                ]
            )
        )
        assert "trend" in builder._named_detectors
        assert "volume" in builder._named_detectors

    def test_detectors_list_instances(self, sample_raw_data):
        """detectors() accepts list of unnamed instances."""
        from signalflow.detector import ExampleSmaCrossDetector

        det1 = ExampleSmaCrossDetector(fast_period=5, slow_period=15)
        det2 = ExampleSmaCrossDetector(fast_period=10, slow_period=30)
        builder = Backtest("test").data(raw=sample_raw_data).detectors([det1, det2])
        assert len(builder._named_detectors) == 2

    def test_backward_compat_single_detector(self, sample_raw_data):
        """Single detector() call still works in backward-compat mode."""
        builder = Backtest("test").data(raw=sample_raw_data).detector("example/sma_cross")
        # _detector should be set for backward compat
        assert builder._detector is not None
        # Also appears in _named_detectors
        assert len(builder._named_detectors) == 1


class TestBuilderMultiValidator:
    """Tests for named validators."""

    def test_validator_returns_self(self):
        """validator() returns self for chaining."""
        from unittest.mock import MagicMock

        mock_validator = MagicMock()
        builder = Backtest()
        result = builder.validator(mock_validator)
        assert result is builder

    def test_validator_auto_names(self):
        """First validator auto-named 'default', second 'validator_1'."""
        from unittest.mock import MagicMock

        builder = Backtest()
        builder.validator(MagicMock(), name="first")
        builder.validator(MagicMock())
        # First was explicitly named, second gets auto-name
        assert "first" in builder._named_validators
        assert "validator_1" in builder._named_validators

    def test_duplicate_validator_name_raises(self):
        """Duplicate validator name raises DuplicateComponentNameError."""
        from unittest.mock import MagicMock
        from signalflow.api.exceptions import DuplicateComponentNameError

        with pytest.raises(DuplicateComponentNameError):
            Backtest().validator(MagicMock(), name="ml").validator(MagicMock(), name="ml")

    def test_validators_list(self):
        """validators() accepts list of (name, instance) tuples."""
        from unittest.mock import MagicMock

        v1, v2 = MagicMock(), MagicMock()
        builder = Backtest().validators([("v1", v1), ("v2", v2)])
        assert "v1" in builder._named_validators
        assert "v2" in builder._named_validators


class TestBuilderMultiEntry:
    """Tests for named entry rules."""

    def test_entry_with_name(self):
        """entry(name=...) stores in _named_entries."""
        builder = Backtest().entry(name="trend", size_pct=0.1, source_detector="trend")
        assert "trend" in builder._named_entries
        assert builder._named_entries["trend"]["size_pct"] == 0.1
        assert builder._named_entries["trend"]["source_detector"] == "trend"

    def test_multiple_named_entries(self):
        """Multiple named entries stored correctly."""
        builder = (
            Backtest()
            .entry(name="trend", size_pct=0.15, source_detector="trend")
            .entry(name="volume", size_pct=0.05, source_detector="volume")
        )
        assert len(builder._named_entries) == 2

    def test_duplicate_entry_name_raises(self):
        """Duplicate entry name raises DuplicateComponentNameError."""
        from signalflow.api.exceptions import DuplicateComponentNameError

        with pytest.raises(DuplicateComponentNameError):
            Backtest().entry(name="x", size=100).entry(name="x", size=200)

    def test_entries_dict(self):
        """entries() accepts dict of name -> config."""
        builder = Backtest().entries(
            {
                "trend": {"size_pct": 0.15, "source_detector": "trend"},
                "volume": {"size_pct": 0.05},
            }
        )
        assert "trend" in builder._named_entries
        assert "volume" in builder._named_entries

    def test_entries_list(self):
        """entries() accepts list of (name, config) tuples."""
        builder = Backtest().entries(
            [
                ("trend", {"size_pct": 0.15}),
                ("volume", {"size_pct": 0.05}),
            ]
        )
        assert len(builder._named_entries) == 2

    def test_entry_backward_compat(self):
        """Unnamed entry() still populates _entry_config."""
        builder = Backtest().entry(size=200, max_positions=5)
        assert builder._entry_config["size"] == 200
        assert builder._entry_config["max_positions"] == 5
        assert len(builder._named_entries) == 0


class TestBuilderMultiExit:
    """Tests for named exit rules."""

    def test_exit_with_name(self):
        """exit(name=...) stores in _named_exits."""
        builder = Backtest().exit(name="standard", tp=0.03, sl=0.015)
        assert "standard" in builder._named_exits
        assert builder._named_exits["standard"]["tp"] == 0.03

    def test_multiple_named_exits(self):
        """Multiple named exits stored correctly."""
        builder = Backtest().exit(name="standard", tp=0.03, sl=0.015).exit(name="trailing", trailing=0.02)
        assert len(builder._named_exits) == 2
        assert builder._named_exits["trailing"]["trailing"] == 0.02

    def test_duplicate_exit_name_raises(self):
        """Duplicate exit name raises DuplicateComponentNameError."""
        from signalflow.api.exceptions import DuplicateComponentNameError

        with pytest.raises(DuplicateComponentNameError):
            Backtest().exit(name="x", tp=0.03).exit(name="x", sl=0.01)

    def test_exits_dict(self):
        """exits() accepts dict of name -> config."""
        builder = Backtest().exits(
            {
                "standard": {"tp": 0.03, "sl": 0.015},
                "trailing": {"trailing": 0.02},
            }
        )
        assert len(builder._named_exits) == 2

    def test_exits_list(self):
        """exits() accepts list of (name, config) tuples."""
        builder = Backtest().exits(
            [
                ("standard", {"tp": 0.03, "sl": 0.015}),
                ("trailing", {"trailing": 0.02}),
            ]
        )
        assert len(builder._named_exits) == 2


class TestBuilderAggregation:
    """Tests for signal aggregation configuration."""

    def test_aggregation_returns_self(self):
        """aggregation() returns self for chaining."""
        builder = Backtest()
        result = builder.aggregation(mode="weighted", weights=[0.7, 0.3])
        assert result is builder

    def test_aggregation_stores_config(self):
        """aggregation() stores config dict."""
        builder = Backtest().aggregation(
            mode="majority",
            min_agreement=0.6,
            probability_threshold=0.7,
        )
        assert builder._aggregation_config["mode"] == "majority"
        assert builder._aggregation_config["min_agreement"] == 0.6
        assert builder._aggregation_config["probability_threshold"] == 0.7

    def test_aggregation_weighted_with_weights(self):
        """aggregation() accepts weights for weighted mode."""
        builder = Backtest().aggregation(mode="weighted", weights=[0.7, 0.3])
        assert builder._aggregation_config["weights"] == [0.7, 0.3]


class TestBuilderMultiValidation:
    """Tests for validation of multi-component configuration."""

    def test_validate_cross_ref_entry_detector(self, sample_raw_data):
        """Validation catches entry referencing non-existent detector."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data)
            .detector("example/sma_cross", name="trend")
            .entry(name="bad", source_detector="nonexistent")
        )
        issues = builder.validate()
        assert any("nonexistent" in i for i in issues)

    def test_validate_cross_ref_detector_data(self, sample_raw_data):
        """Validation catches detector referencing non-existent data source."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data, name="1m")
            .detector("example/sma_cross", name="trend", data_source="nonexistent")
        )
        issues = builder.validate()
        assert any("nonexistent" in i for i in issues)

    def test_validate_aggregation_weights_mismatch(self, sample_raw_data):
        """Validation catches aggregation weights count mismatch."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data)
            .detector("example/sma_cross", name="trend")
            .detector("example/sma_cross", name="volume")
            .aggregation(mode="weighted", weights=[0.5])  # 1 weight, 2 detectors
        )
        issues = builder.validate()
        assert any("weights" in i.lower() for i in issues)

    def test_validate_multi_exit_tp_sl_warning(self, sample_raw_data):
        """Validation warns about bad TP/SL in named exits."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data)
            .detector("example/sma_cross")
            .exit(name="bad", tp=0.01, sl=0.02)  # TP < SL
        )
        issues = builder.validate()
        assert any("WARNING" in i and "TP" in i for i in issues)

    def test_validate_valid_multi_config(self, sample_raw_data):
        """Valid multi-component config passes validation."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data, name="1m")
            .detector("example/sma_cross", name="trend", data_source="1m")
            .entry(name="trend_entry", source_detector="trend", size_pct=0.1)
            .exit(name="standard", tp=0.03, sl=0.015)
            .capital(10_000)
        )
        issues = builder.validate()
        errors = [i for i in issues if "ERROR" in i]
        assert len(errors) == 0

    def test_validate_named_data_counts_as_data(self, sample_raw_data):
        """Named data sources satisfy the 'has data' check."""
        builder = Backtest("test").data(raw=sample_raw_data, name="1m").detector("example/sma_cross")
        issues = builder.validate()
        assert not any("No data" in i for i in issues)


class TestBuilderMultiChain:
    """Tests for full multi-component chains."""

    def test_full_multi_chain(self, sample_raw_data):
        """Full multi-component chain builds correctly."""
        builder = (
            Backtest("ensemble")
            .data(raw=sample_raw_data, name="1m")
            .detector("example/sma_cross", name="trend", data_source="1m")
            .detector("example/sma_cross", name="volume", data_source="1m")
            .aggregation(mode="weighted", weights=[0.7, 0.3])
            .entry(name="trend_entry", source_detector="trend", size_pct=0.15)
            .entry(name="volume_entry", source_detector="volume", size_pct=0.05)
            .exit(name="standard", tp=0.03, sl=0.015)
            .exit(name="trailing", trailing=0.02)
            .capital(50_000)
            .fee(0.0005)
            .progress(False)
        )
        assert builder.strategy_id == "ensemble"
        assert len(builder._named_data) == 1
        assert len(builder._named_detectors) == 2
        assert len(builder._named_entries) == 2
        assert len(builder._named_exits) == 2
        assert builder._aggregation_config is not None
        assert builder._capital == 50_000

    def test_repr_shows_named_components(self, sample_raw_data):
        """__repr__ shows named detectors, entries, exits."""
        builder = (
            Backtest("ensemble")
            .data(raw=sample_raw_data)
            .detector("example/sma_cross", name="trend")
            .entry(name="trend_entry", size_pct=0.1)
            .exit(name="standard", tp=0.03)
        )
        repr_str = repr(builder)
        assert "trend" in repr_str
        assert "trend_entry" in repr_str
        assert "standard" in repr_str


class TestBuilderBuildEntryRules:
    """Tests for _build_entry_rules with multi-component support."""

    def test_build_single_entry(self, sample_raw_data):
        """Single unnamed entry builds one rule."""
        builder = (
            Backtest("test").data(raw=sample_raw_data).detector("example/sma_cross").entry(size=200, max_positions=5)
        )
        rules = builder._build_entry_rules()
        assert len(rules) == 1
        assert rules[0].base_position_size == 200
        assert rules[0].max_total_positions == 5

    def test_build_named_entries(self, sample_raw_data):
        """Named entries build multiple rules."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data)
            .detector("example/sma_cross", name="trend")
            .entry(name="trend_entry", source_detector="trend", size_pct=0.1)
            .entry(name="volume_entry", size=50)
            .capital(10_000)
        )
        rules = builder._build_entry_rules()
        assert len(rules) == 2

    def test_build_entry_source_detector(self, sample_raw_data):
        """Entry rule gets source_detector set."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data)
            .detector("example/sma_cross", name="trend")
            .entry(name="trend_entry", source_detector="trend", size_pct=0.1)
            .capital(10_000)
        )
        rules = builder._build_entry_rules()
        assert rules[0].source_detector == "trend"

    def test_build_entry_size_pct_uses_capital(self, sample_raw_data):
        """size_pct entry calculates size from capital."""
        builder = (
            Backtest("test").data(raw=sample_raw_data).detector("example/sma_cross").entry(size_pct=0.1).capital(50_000)
        )
        rules = builder._build_entry_rules()
        # size_pct=0.1 * capital=50_000 = 5_000
        assert rules[0].base_position_size == 5_000.0


class TestBuilderBuildExitRules:
    """Tests for _build_exit_rules with multi-component support."""

    def test_build_single_exit(self, sample_raw_data):
        """Single unnamed exit builds exit rules."""
        builder = Backtest("test").data(raw=sample_raw_data).detector("example/sma_cross").exit(tp=0.03, sl=0.015)
        rules = builder._build_exit_rules()
        assert len(rules) >= 1

    def test_build_named_exits(self, sample_raw_data):
        """Named exits build rules from all configs."""
        builder = (
            Backtest("test")
            .data(raw=sample_raw_data)
            .detector("example/sma_cross")
            .exit(name="standard", tp=0.03, sl=0.015)
            .exit(name="aggressive", tp=0.05, sl=0.02)
        )
        rules = builder._build_exit_rules()
        # Each tp/sl config creates one TakeProfitStopLossExit
        assert len(rules) >= 2

    def test_build_default_exit(self, sample_raw_data):
        """No exit config gives default TP/SL rule."""
        builder = Backtest("test").data(raw=sample_raw_data).detector("example/sma_cross")
        rules = builder._build_exit_rules()
        assert len(rules) >= 1


class TestBuilderSignalResolution:
    """Tests for signal resolution with pre-computed signals."""

    def test_precomputed_signals(self, sample_raw_data, sample_signals):
        """Pre-computed signals skip detection."""
        builder = Backtest("test").data(raw=sample_raw_data).signals(sample_signals)
        merged, named = builder._resolve_signals(sample_raw_data)
        assert merged is sample_signals
        assert named == {}

    def test_single_detector_resolution(self, sample_raw_data):
        """Single named detector resolves correctly."""
        builder = Backtest("test").data(raw=sample_raw_data).detector("example/sma_cross", name="trend")
        merged, named = builder._resolve_signals(sample_raw_data)
        assert isinstance(merged, Signals)
        assert "trend" in named
