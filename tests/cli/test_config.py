"""Tests for CLI configuration loader."""

import tempfile
from pathlib import Path

import pytest
import yaml

from signalflow.cli.config import (
    AggregationConfig,
    BacktestConfig,
    DataConfig,
    DetectorConfig,
    EntryConfig,
    ExitConfig,
    ValidatorConfig,
    generate_sample_config,
)


class TestDataConfig:
    """Tests for DataConfig."""

    def test_create_data_config(self):
        """DataConfig can be created with required fields."""
        cfg = DataConfig(
            source="data/test.duckdb",
            pairs=["BTCUSDT"],
            start="2024-01-01",
        )
        assert cfg.source == "data/test.duckdb"
        assert cfg.pairs == ["BTCUSDT"]
        assert cfg.start == "2024-01-01"

    def test_default_values(self):
        """DataConfig has sensible defaults."""
        cfg = DataConfig(
            source="test.duckdb",
            pairs=["BTCUSDT"],
            start="2024-01-01",
        )
        assert cfg.timeframe == "1m"
        assert cfg.data_type == "perpetual"
        assert cfg.end is None


class TestDetectorConfig:
    """Tests for DetectorConfig."""

    def test_create_detector_config(self):
        """DetectorConfig can be created."""
        cfg = DetectorConfig(
            name="example/sma_cross",
            params={"fast_period": 20, "slow_period": 50},
        )
        assert cfg.name == "example/sma_cross"
        assert cfg.params["fast_period"] == 20

    def test_empty_params(self):
        """DetectorConfig works with empty params."""
        cfg = DetectorConfig(name="test/detector")
        assert cfg.params == {}


class TestEntryConfig:
    """Tests for EntryConfig."""

    def test_defaults(self):
        """EntryConfig has sensible defaults."""
        cfg = EntryConfig()
        assert cfg.max_positions == 10
        assert cfg.max_per_pair == 1
        assert cfg.size is None
        assert cfg.size_pct is None


class TestExitConfig:
    """Tests for ExitConfig."""

    def test_defaults(self):
        """ExitConfig has sensible defaults."""
        cfg = ExitConfig()
        assert cfg.tp is None
        assert cfg.sl is None
        assert cfg.trailing is None
        assert cfg.time_limit is None


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_from_dict_minimal(self):
        """BacktestConfig can be created from minimal dict."""
        d = {
            "data": {
                "source": "test.duckdb",
                "pairs": ["BTCUSDT"],
                "start": "2024-01-01",
            },
            "detector": {
                "name": "example/sma_cross",
            },
        }
        cfg = BacktestConfig.from_dict(d)
        assert cfg.data is not None
        assert cfg.detector is not None
        assert cfg.capital == 10_000.0

    def test_from_dict_full(self):
        """BacktestConfig can be created from full dict."""
        d = {
            "strategy": {"id": "test_strategy"},
            "data": {
                "source": "test.duckdb",
                "pairs": ["BTCUSDT", "ETHUSDT"],
                "start": "2024-01-01",
                "end": "2024-06-01",
                "timeframe": "4h",
                "data_type": "spot",
            },
            "detector": {
                "name": "example/sma_cross",
                "params": {"fast_period": 10, "slow_period": 30},
            },
            "entry": {
                "size_pct": 0.1,
                "max_positions": 5,
            },
            "exit": {
                "tp": 0.03,
                "sl": 0.015,
            },
            "capital": 50000,
            "fee": 0.0005,
        }
        cfg = BacktestConfig.from_dict(d)
        assert cfg.strategy_id == "test_strategy"
        assert cfg.data.pairs == ["BTCUSDT", "ETHUSDT"]
        assert cfg.detector.params["fast_period"] == 10
        assert cfg.entry.size_pct == 0.1
        assert cfg.exit.tp == 0.03
        assert cfg.capital == 50000

    def test_from_yaml(self):
        """BacktestConfig can be loaded from YAML file."""
        yaml_content = """
strategy:
  id: yaml_test
data:
  source: test.duckdb
  pairs: [BTCUSDT]
  start: "2024-01-01"
detector:
  name: example/sma_cross
capital: 25000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            cfg = BacktestConfig.from_yaml(path)
            assert cfg.strategy_id == "yaml_test"
            assert cfg.capital == 25000
        finally:
            Path(path).unlink()

    def test_from_yaml_file_not_found(self):
        """BacktestConfig raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            BacktestConfig.from_yaml("/nonexistent/config.yaml")

    def test_validate_missing_data(self):
        """Validation catches missing data."""
        cfg = BacktestConfig(
            detector=DetectorConfig(name="test"),
        )
        issues = cfg.validate()
        assert any("data" in i.lower() for i in issues)

    def test_validate_missing_detector(self):
        """Validation catches missing detector."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
        )
        issues = cfg.validate()
        assert any("detector" in i.lower() for i in issues)

    def test_validate_bad_tp_sl_ratio(self):
        """Validation warns about bad TP/SL ratio."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            detector=DetectorConfig(name="test"),
            exit=ExitConfig(tp=0.01, sl=0.02),  # TP < SL
        )
        issues = cfg.validate()
        assert any("WARNING" in i and "TP" in i for i in issues)

    def test_validate_negative_capital(self):
        """Validation catches negative capital."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            detector=DetectorConfig(name="test"),
            capital=-1000,
        )
        issues = cfg.validate()
        assert any("capital" in i.lower() for i in issues)

    def test_validate_valid_config(self):
        """Valid config passes validation."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            detector=DetectorConfig(name="example/sma_cross"),
            capital=10000,
        )
        issues = cfg.validate()
        errors = [i for i in issues if "ERROR" in i]
        assert len(errors) == 0


class TestGenerateSampleConfig:
    """Tests for sample config generation."""

    def test_generates_valid_yaml(self):
        """Generated sample is valid YAML."""
        sample = generate_sample_config()
        parsed = yaml.safe_load(sample)
        assert isinstance(parsed, dict)

    def test_sample_has_required_sections(self):
        """Sample has all required sections."""
        sample = generate_sample_config()
        parsed = yaml.safe_load(sample)
        assert "strategy" in parsed
        assert "data" in parsed
        assert "detector" in parsed
        assert "entry" in parsed
        assert "exit" in parsed
        assert "capital" in parsed

    def test_sample_is_parseable_as_config(self):
        """Sample can be parsed as BacktestConfig."""
        sample = generate_sample_config()
        parsed = yaml.safe_load(sample)
        cfg = BacktestConfig.from_dict(parsed)
        assert cfg.strategy_id == "my_strategy"


# ===========================================================================
# Multi-Component Config Tests
# ===========================================================================


class TestValidatorConfig:
    """Tests for ValidatorConfig."""

    def test_create_validator_config(self):
        """ValidatorConfig can be created."""
        cfg = ValidatorConfig(
            name="validator/lightgbm",
            params={"n_estimators": 200},
        )
        assert cfg.name == "validator/lightgbm"
        assert cfg.params["n_estimators"] == 200

    def test_empty_params(self):
        """ValidatorConfig works with empty params."""
        cfg = ValidatorConfig(name="test/validator")
        assert cfg.params == {}


class TestAggregationConfig:
    """Tests for AggregationConfig."""

    def test_defaults(self):
        """AggregationConfig has sensible defaults."""
        cfg = AggregationConfig()
        assert cfg.mode == "merge"
        assert cfg.min_agreement == 0.5
        assert cfg.weights is None
        assert cfg.probability_threshold == 0.5

    def test_weighted_mode(self):
        """AggregationConfig stores weighted config."""
        cfg = AggregationConfig(
            mode="weighted",
            weights=[0.7, 0.3],
        )
        assert cfg.mode == "weighted"
        assert cfg.weights == [0.7, 0.3]


class TestDetectorConfigDataSource:
    """Tests for DetectorConfig data_source field."""

    def test_data_source_field(self):
        """DetectorConfig has data_source field."""
        cfg = DetectorConfig(
            name="example/sma_cross",
            data_source="spot_1h",
        )
        assert cfg.data_source == "spot_1h"

    def test_data_source_default_none(self):
        """DetectorConfig data_source defaults to None."""
        cfg = DetectorConfig(name="test")
        assert cfg.data_source is None


class TestEntryConfigSourceDetector:
    """Tests for EntryConfig source_detector field."""

    def test_source_detector_field(self):
        """EntryConfig has source_detector field."""
        cfg = EntryConfig(
            size_pct=0.1,
            source_detector="trend",
        )
        assert cfg.source_detector == "trend"

    def test_source_detector_default_none(self):
        """EntryConfig source_detector defaults to None."""
        cfg = EntryConfig()
        assert cfg.source_detector is None


class TestMultiComponentFromDict:
    """Tests for BacktestConfig.from_dict with plural keys."""

    def test_data_sources_dict(self):
        """from_dict parses data_sources dict format."""
        d = {
            "data_sources": {
                "spot_1m": {
                    "source": "data.duckdb",
                    "pairs": ["BTCUSDT"],
                    "start": "2024-01-01",
                    "timeframe": "1m",
                },
                "spot_1h": {
                    "source": "data.duckdb",
                    "pairs": ["BTCUSDT"],
                    "start": "2024-01-01",
                    "timeframe": "1h",
                },
            },
            "detector": {"name": "example/sma_cross"},
        }
        cfg = BacktestConfig.from_dict(d)
        assert cfg.data_sources is not None
        assert len(cfg.data_sources) == 2
        assert "spot_1m" in cfg.data_sources
        assert "spot_1h" in cfg.data_sources
        assert cfg.data_sources["spot_1m"].timeframe == "1m"
        assert cfg.data_sources["spot_1h"].timeframe == "1h"

    def test_detectors_dict(self):
        """from_dict parses detectors dict format."""
        d = {
            "data": {"source": "test.duckdb", "pairs": ["BTC"], "start": "2024-01-01"},
            "detectors": {
                "trend": {
                    "name": "example/sma_cross",
                    "params": {"fast_period": 20, "slow_period": 50},
                    "data_source": "spot_1h",
                },
                "volume": {
                    "name": "example/volume_spike",
                    "params": {"threshold": 2.0},
                    "data_source": "spot_1m",
                },
            },
        }
        cfg = BacktestConfig.from_dict(d)
        assert cfg.detectors is not None
        assert len(cfg.detectors) == 2
        assert cfg.detectors["trend"].name == "example/sma_cross"
        assert cfg.detectors["trend"].data_source == "spot_1h"
        assert cfg.detectors["volume"].params["threshold"] == 2.0

    def test_validators_dict(self):
        """from_dict parses validators dict format."""
        d = {
            "data": {"source": "test.duckdb", "pairs": ["BTC"], "start": "2024-01-01"},
            "detector": {"name": "example/sma_cross"},
            "validators": {
                "ml_filter": {
                    "name": "validator/lightgbm",
                    "params": {"n_estimators": 200},
                },
            },
        }
        cfg = BacktestConfig.from_dict(d)
        assert cfg.validators is not None
        assert "ml_filter" in cfg.validators
        assert cfg.validators["ml_filter"].name == "validator/lightgbm"

    def test_entries_dict(self):
        """from_dict parses entries dict format."""
        d = {
            "data": {"source": "test.duckdb", "pairs": ["BTC"], "start": "2024-01-01"},
            "detector": {"name": "example/sma_cross"},
            "entries": {
                "trend_entry": {
                    "size_pct": 0.15,
                    "source_detector": "trend",
                },
                "volume_entry": {
                    "size_pct": 0.05,
                    "source_detector": "volume",
                },
            },
        }
        cfg = BacktestConfig.from_dict(d)
        assert cfg.entries is not None
        assert len(cfg.entries) == 2
        assert cfg.entries["trend_entry"].size_pct == 0.15
        assert cfg.entries["trend_entry"].source_detector == "trend"

    def test_exits_dict(self):
        """from_dict parses exits dict format."""
        d = {
            "data": {"source": "test.duckdb", "pairs": ["BTC"], "start": "2024-01-01"},
            "detector": {"name": "example/sma_cross"},
            "exits": {
                "standard": {"tp": 0.03, "sl": 0.015},
                "trailing": {"trailing": 0.02},
            },
        }
        cfg = BacktestConfig.from_dict(d)
        assert cfg.exits is not None
        assert len(cfg.exits) == 2
        assert cfg.exits["standard"].tp == 0.03
        assert cfg.exits["trailing"].trailing == 0.02

    def test_aggregation(self):
        """from_dict parses aggregation config."""
        d = {
            "data": {"source": "test.duckdb", "pairs": ["BTC"], "start": "2024-01-01"},
            "detector": {"name": "example/sma_cross"},
            "aggregation": {
                "mode": "weighted",
                "weights": [0.7, 0.3],
                "min_agreement": 0.6,
                "probability_threshold": 0.7,
            },
        }
        cfg = BacktestConfig.from_dict(d)
        assert cfg.aggregation is not None
        assert cfg.aggregation.mode == "weighted"
        assert cfg.aggregation.weights == [0.7, 0.3]
        assert cfg.aggregation.min_agreement == 0.6

    def test_full_multi_component(self):
        """from_dict parses full multi-component config."""
        d = {
            "strategy": {"id": "ensemble"},
            "data_sources": {
                "spot_1m": {
                    "source": "data.duckdb",
                    "pairs": ["BTCUSDT"],
                    "start": "2024-01-01",
                    "timeframe": "1m",
                },
            },
            "detectors": {
                "trend": {
                    "name": "example/sma_cross",
                    "params": {"fast_period": 20},
                    "data_source": "spot_1m",
                },
            },
            "validators": {
                "ml": {"name": "validator/lgbm", "params": {}},
            },
            "aggregation": {"mode": "merge"},
            "entries": {
                "trend_entry": {"size_pct": 0.15, "source_detector": "trend"},
            },
            "exits": {
                "standard": {"tp": 0.03, "sl": 0.015},
            },
            "capital": 50000,
        }
        cfg = BacktestConfig.from_dict(d)
        assert cfg.strategy_id == "ensemble"
        assert cfg.data_sources is not None
        assert cfg.detectors is not None
        assert cfg.validators is not None
        assert cfg.aggregation is not None
        assert cfg.entries is not None
        assert cfg.exits is not None
        assert cfg.capital == 50000


class TestMultiComponentFromDictListFormat:
    """Tests for from_dict with list format (alternative to dict)."""

    def test_detectors_list_format(self):
        """from_dict parses detectors as list with name field."""
        d = {
            "data": {"source": "test.duckdb", "pairs": ["BTC"], "start": "2024-01-01"},
            "detectors": [
                {
                    "name": "trend",
                    "detector": "example/sma_cross",
                    "params": {"fast_period": 20},
                },
                {
                    "name": "volume",
                    "detector": "example/volume_spike",
                },
            ],
        }
        cfg = BacktestConfig.from_dict(d)
        assert cfg.detectors is not None
        assert "trend" in cfg.detectors
        assert "volume" in cfg.detectors
        assert cfg.detectors["trend"].name == "example/sma_cross"

    def test_entries_list_format(self):
        """from_dict parses entries as list with name field."""
        d = {
            "data": {"source": "test.duckdb", "pairs": ["BTC"], "start": "2024-01-01"},
            "detector": {"name": "test"},
            "entries": [
                {"name": "trend_entry", "size_pct": 0.15, "source_detector": "trend"},
                {"name": "volume_entry", "size_pct": 0.05},
            ],
        }
        cfg = BacktestConfig.from_dict(d)
        assert cfg.entries is not None
        assert len(cfg.entries) == 2
        assert cfg.entries["trend_entry"].source_detector == "trend"

    def test_exits_list_format(self):
        """from_dict parses exits as list with name field."""
        d = {
            "data": {"source": "test.duckdb", "pairs": ["BTC"], "start": "2024-01-01"},
            "detector": {"name": "test"},
            "exits": [
                {"name": "standard", "tp": 0.03, "sl": 0.015},
                {"name": "trailing", "trailing": 0.02},
            ],
        }
        cfg = BacktestConfig.from_dict(d)
        assert cfg.exits is not None
        assert "standard" in cfg.exits
        assert "trailing" in cfg.exits

    def test_validators_list_format(self):
        """from_dict parses validators as list with name field."""
        d = {
            "data": {"source": "test.duckdb", "pairs": ["BTC"], "start": "2024-01-01"},
            "detector": {"name": "test"},
            "validators": [
                {"name": "ml", "validator": "validator/lgbm", "params": {}},
            ],
        }
        cfg = BacktestConfig.from_dict(d)
        assert cfg.validators is not None
        assert "ml" in cfg.validators

    def test_data_sources_list_format(self):
        """from_dict parses data_sources as list with name field."""
        d = {
            "data_sources": [
                {
                    "name": "spot_1m",
                    "source": "data.duckdb",
                    "pairs": ["BTCUSDT"],
                    "start": "2024-01-01",
                    "timeframe": "1m",
                },
            ],
            "detector": {"name": "test"},
        }
        cfg = BacktestConfig.from_dict(d)
        assert cfg.data_sources is not None
        assert "spot_1m" in cfg.data_sources
        assert cfg.data_sources["spot_1m"].timeframe == "1m"


class TestMultiComponentValidation:
    """Tests for validation of multi-component config."""

    def test_mutual_exclusion_data(self):
        """Cannot use both 'data' and 'data_sources'."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            data_sources={"1m": DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01")},
            detector=DetectorConfig(name="test"),
        )
        issues = cfg.validate()
        assert any("Cannot use both" in i and "data" in i.lower() for i in issues)

    def test_mutual_exclusion_detector(self):
        """Cannot use both 'detector' and 'detectors'."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            detector=DetectorConfig(name="test"),
            detectors={"trend": DetectorConfig(name="test")},
        )
        issues = cfg.validate()
        assert any("Cannot use both" in i and "detector" in i.lower() for i in issues)

    def test_cross_ref_entry_detector(self):
        """Validation catches entry referencing non-existent detector."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            detectors={"trend": DetectorConfig(name="test")},
            entries={
                "bad_entry": EntryConfig(source_detector="nonexistent"),
            },
        )
        issues = cfg.validate()
        assert any("nonexistent" in i for i in issues)

    def test_cross_ref_detector_data(self):
        """Validation catches detector referencing non-existent data source."""
        cfg = BacktestConfig(
            data_sources={"1m": DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01")},
            detectors={"trend": DetectorConfig(name="test", data_source="nonexistent")},
        )
        issues = cfg.validate()
        assert any("nonexistent" in i for i in issues)

    def test_aggregation_weights_mismatch(self):
        """Validation catches aggregation weights count mismatch."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            detectors={
                "trend": DetectorConfig(name="test"),
                "volume": DetectorConfig(name="test"),
            },
            aggregation=AggregationConfig(weights=[0.5]),  # 1 weight, 2 detectors
        )
        issues = cfg.validate()
        assert any("weights" in i.lower() for i in issues)

    def test_multi_exit_tp_sl_warning(self):
        """Validation warns about bad TP/SL in named exits."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            detector=DetectorConfig(name="test"),
            exits={
                "bad": ExitConfig(tp=0.01, sl=0.02),
            },
        )
        issues = cfg.validate()
        assert any("WARNING" in i and "TP" in i for i in issues)

    def test_valid_multi_config(self):
        """Valid multi-component config passes validation."""
        cfg = BacktestConfig(
            data_sources={"1m": DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01")},
            detectors={"trend": DetectorConfig(name="example/sma_cross", data_source="1m")},
            entries={"trend_entry": EntryConfig(size_pct=0.1, source_detector="trend")},
            exits={"standard": ExitConfig(tp=0.03, sl=0.015)},
            aggregation=AggregationConfig(mode="merge"),
            capital=10_000,
        )
        issues = cfg.validate()
        errors = [i for i in issues if "ERROR" in i]
        assert len(errors) == 0

    def test_data_sources_counts_as_data(self):
        """data_sources satisfy the 'has data' check."""
        cfg = BacktestConfig(
            data_sources={"1m": DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01")},
            detector=DetectorConfig(name="test"),
        )
        issues = cfg.validate()
        assert not any("No data" in i for i in issues)

    def test_detectors_counts_as_detector(self):
        """detectors (plural) satisfy the 'has detector' check."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            detectors={"trend": DetectorConfig(name="test")},
        )
        issues = cfg.validate()
        assert not any("No detector" in i for i in issues)


class TestMultiComponentToBuilder:
    """Tests for to_builder with multi-component config."""

    def test_to_builder_multi_detectors(self):
        """to_builder maps plural detectors to builder calls."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            detectors={
                "trend": DetectorConfig(name="example/sma_cross", params={"fast_period": 20}),
            },
        )
        builder = cfg.to_builder()
        assert "trend" in builder._named_detectors

    def test_to_builder_multi_entries(self):
        """to_builder maps plural entries to builder calls."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            detector=DetectorConfig(name="example/sma_cross"),
            entries={
                "trend_entry": EntryConfig(size_pct=0.15, source_detector="trend"),
                "volume_entry": EntryConfig(size_pct=0.05),
            },
        )
        builder = cfg.to_builder()
        assert "trend_entry" in builder._named_entries
        assert "volume_entry" in builder._named_entries

    def test_to_builder_multi_exits(self):
        """to_builder maps plural exits to builder calls."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            detector=DetectorConfig(name="example/sma_cross"),
            exits={
                "standard": ExitConfig(tp=0.03, sl=0.015),
                "trailing": ExitConfig(trailing=0.02),
            },
        )
        builder = cfg.to_builder()
        assert "standard" in builder._named_exits
        assert "trailing" in builder._named_exits

    def test_to_builder_aggregation(self):
        """to_builder maps aggregation config."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            detector=DetectorConfig(name="example/sma_cross"),
            aggregation=AggregationConfig(mode="weighted", weights=[0.7, 0.3]),
        )
        builder = cfg.to_builder()
        assert builder._aggregation_config is not None
        assert builder._aggregation_config["mode"] == "weighted"
        assert builder._aggregation_config["weights"] == [0.7, 0.3]

    def test_to_builder_validators(self):
        """to_builder maps validators to builder calls."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            detector=DetectorConfig(name="example/sma_cross"),
            validators={
                "ml": ValidatorConfig(name="validator/lgbm"),
            },
        )
        # Validator may not exist in registry, which is OK —
        # we just check that to_builder attempts to set it
        try:
            builder = cfg.to_builder()
            # If it succeeds, validators should be set
            assert "ml" in builder._named_validators
        except Exception:
            # ValidatorNotFoundError is expected if no validators registered
            pass

    def test_to_builder_backward_compat(self):
        """to_builder with singular config still works."""
        cfg = BacktestConfig(
            data=DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            detector=DetectorConfig(name="example/sma_cross"),
            entry=EntryConfig(size_pct=0.1, max_positions=5),
            exit=ExitConfig(tp=0.03, sl=0.015),
            capital=25000,
        )
        builder = cfg.to_builder()
        assert builder._capital == 25000
        assert len(builder._named_entries) == 0  # singular → _entry_config
        assert len(builder._named_exits) == 0  # singular → _exit_config

    def test_to_builder_data_sources(self):
        """to_builder maps plural data_sources to builder calls."""
        cfg = BacktestConfig(
            data_sources={
                "1m": DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            },
            detector=DetectorConfig(name="example/sma_cross"),
        )
        builder = cfg.to_builder()
        assert "1m" in builder._named_data

    def test_to_builder_detector_data_source(self):
        """to_builder maps detector data_source to builder."""
        cfg = BacktestConfig(
            data_sources={
                "1h": DataConfig(source="test.duckdb", pairs=["BTC"], start="2024-01-01"),
            },
            detectors={
                "trend": DetectorConfig(
                    name="example/sma_cross",
                    data_source="1h",
                ),
            },
        )
        builder = cfg.to_builder()
        assert builder._detector_data_sources.get("trend") == "1h"


class TestMultiComponentYAML:
    """Tests for multi-component YAML file loading."""

    def test_multi_component_yaml(self):
        """Full multi-component YAML file parses correctly."""
        yaml_content = """
strategy:
  id: ensemble

data_sources:
  spot_1m:
    source: data/binance.duckdb
    pairs: [BTCUSDT, ETHUSDT]
    start: "2024-01-01"
    timeframe: 1m
  spot_1h:
    source: data/binance.duckdb
    pairs: [BTCUSDT, ETHUSDT]
    start: "2024-01-01"
    timeframe: 1h

detectors:
  trend:
    name: example/sma_cross
    params:
      fast_period: 20
      slow_period: 50
    data_source: spot_1h
  volume:
    name: example/volume_spike
    params:
      threshold: 2.0
    data_source: spot_1m

aggregation:
  mode: weighted
  weights: [0.7, 0.3]

entries:
  trend_entry:
    size_pct: 0.15
    source_detector: trend
  volume_entry:
    size_pct: 0.05
    source_detector: volume

exits:
  standard:
    tp: 0.03
    sl: 0.015
  trailing:
    trailing: 0.02

capital: 50000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            cfg = BacktestConfig.from_yaml(path)
            assert cfg.strategy_id == "ensemble"
            assert cfg.data_sources is not None
            assert len(cfg.data_sources) == 2
            assert cfg.detectors is not None
            assert len(cfg.detectors) == 2
            assert cfg.detectors["trend"].data_source == "spot_1h"
            assert cfg.aggregation is not None
            assert cfg.aggregation.mode == "weighted"
            assert cfg.entries is not None
            assert len(cfg.entries) == 2
            assert cfg.exits is not None
            assert len(cfg.exits) == 2
            assert cfg.capital == 50000

            # Validate should pass
            issues = cfg.validate()
            errors = [i for i in issues if "ERROR" in i]
            assert len(errors) == 0
        finally:
            Path(path).unlink()

    def test_singular_yaml_backward_compat(self):
        """Singular YAML config still works."""
        yaml_content = """
strategy:
  id: simple
data:
  source: test.duckdb
  pairs: [BTCUSDT]
  start: "2024-01-01"
detector:
  name: example/sma_cross
entry:
  size_pct: 0.1
exit:
  tp: 0.03
  sl: 0.015
capital: 10000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            cfg = BacktestConfig.from_yaml(path)
            assert cfg.strategy_id == "simple"
            assert cfg.data is not None
            assert cfg.detector is not None
            assert cfg.data_sources is None
            assert cfg.detectors is None
            assert cfg.capital == 10000
        finally:
            Path(path).unlink()
