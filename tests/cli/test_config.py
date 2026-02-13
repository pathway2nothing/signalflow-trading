"""Tests for CLI configuration loader."""

import tempfile
from pathlib import Path

import pytest
import yaml

from signalflow.cli.config import (
    BacktestConfig,
    DataConfig,
    DetectorConfig,
    EntryConfig,
    ExitConfig,
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
        assert cfg.timeframe == "1h"
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
