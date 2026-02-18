"""Tests for signalflow.config module."""

import os
import tempfile
from pathlib import Path

import pytest

from signalflow.config import deep_merge, get_flow_info, list_flows, load_flow_config, load_yaml


class TestDeepMerge:
    """Tests for deep_merge function."""

    def test_simple_merge(self):
        """Merge two flat dicts."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Merge nested dicts."""
        base = {"a": 1, "nested": {"x": 10, "y": 20}}
        override = {"nested": {"y": 25, "z": 30}}
        result = deep_merge(base, override)
        assert result == {"a": 1, "nested": {"x": 10, "y": 25, "z": 30}}

    def test_override_replaces_non_dict(self):
        """Override replaces non-dict value."""
        base = {"a": {"nested": 1}}
        override = {"a": "string"}
        result = deep_merge(base, override)
        assert result == {"a": "string"}

    def test_does_not_mutate_original(self):
        """Original dicts are not mutated."""
        base = {"a": 1}
        override = {"b": 2}
        deep_merge(base, override)
        assert base == {"a": 1}
        assert override == {"b": 2}


class TestLoadYaml:
    """Tests for load_yaml function."""

    def test_load_yaml_file(self, tmp_path: Path):
        """Load a simple YAML file."""
        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text("key: value\nnested:\n  a: 1\n")

        result = load_yaml(yaml_file)
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_load_yaml_not_found(self, tmp_path: Path):
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "nonexistent.yml")

    def test_load_empty_yaml(self, tmp_path: Path):
        """Empty YAML returns empty dict."""
        yaml_file = tmp_path / "empty.yml"
        yaml_file.write_text("")

        result = load_yaml(yaml_file)
        assert result == {}


class TestLoadFlowConfig:
    """Tests for load_flow_config function."""

    @pytest.fixture
    def conf_dir(self, tmp_path: Path) -> Path:
        """Create a test conf directory structure."""
        conf = tmp_path / "conf" / "base"
        conf.mkdir(parents=True)

        # Create parameters directory
        params = conf / "parameters"
        params.mkdir()

        # Create common.yml
        (params / "common.yml").write_text("""
defaults:
  capital: 10000
  fee: 0.001
telegram:
  enabled: false
output:
  signals: "data/{flow_id}/signals"
  strategy: "data/{flow_id}/strategy"
  db: "data/{flow_id}.duckdb"
""")

        # Create flows directory
        flows = conf / "flows"
        flows.mkdir()

        # Create test flow
        (flows / "test_flow.yml").write_text("""
flow_id: test_flow
flow_name: Test Flow
description: A test flow for unit tests
detector:
  type: example/sma_cross
  fast_period: 10
  slow_period: 20
capital: 50000
""")

        return conf

    def test_load_flow_config(self, conf_dir: Path):
        """Load a flow config and merge with defaults."""
        config = load_flow_config("test_flow", conf_dir)

        assert config["flow_id"] == "test_flow"
        assert config["flow_name"] == "Test Flow"
        assert config["capital"] == 50000  # Overridden
        assert config["fee"] == 0.001  # From defaults
        assert config["detector"]["type"] == "example/sma_cross"
        assert config["telegram"]["enabled"] is False

    def test_load_flow_config_not_found(self, conf_dir: Path):
        """Raise FileNotFoundError for missing flow."""
        with pytest.raises(FileNotFoundError, match="Flow config not found"):
            load_flow_config("nonexistent", conf_dir)

    def test_output_paths_resolved(self, conf_dir: Path):
        """Output paths have flow_id substituted."""
        config = load_flow_config("test_flow", conf_dir)

        assert config["output"]["signals"] == "data/test_flow/signals"
        assert config["output"]["db"] == "data/test_flow.duckdb"


class TestListFlows:
    """Tests for list_flows function."""

    def test_list_flows(self, tmp_path: Path):
        """List available flows."""
        flows_dir = tmp_path / "flows"
        flows_dir.mkdir()

        (flows_dir / "alpha.yml").write_text("flow_id: alpha")
        (flows_dir / "beta.yml").write_text("flow_id: beta")
        (flows_dir / "gamma.yml").write_text("flow_id: gamma")

        result = list_flows(tmp_path)
        assert result == ["alpha", "beta", "gamma"]

    def test_list_flows_empty(self, tmp_path: Path):
        """Empty list for missing flows dir."""
        result = list_flows(tmp_path)
        assert result == []


class TestGetFlowInfo:
    """Tests for get_flow_info function."""

    def test_get_flow_info(self, tmp_path: Path):
        """Get flow metadata."""
        conf = tmp_path / "conf"
        conf.mkdir()

        flows = conf / "flows"
        flows.mkdir()

        (flows / "my_flow.yml").write_text("""
flow_id: my_flow
flow_name: My Flow
description: Description of my flow
""")

        info = get_flow_info("my_flow", conf)

        assert info["flow_id"] == "my_flow"
        assert info["flow_name"] == "My Flow"
        assert info["description"] == "Description of my flow"


class TestEnvVarResolution:
    """Tests for environment variable resolution."""

    def test_env_var_resolved(self, tmp_path: Path):
        """Environment variables are resolved in config."""
        conf = tmp_path
        flows = conf / "flows"
        flows.mkdir()

        (flows / "env_test.yml").write_text("""
flow_id: env_test
api_key: ${TEST_API_KEY}
""")

        os.environ["TEST_API_KEY"] = "secret123"
        try:
            config = load_flow_config("env_test", conf)
            assert config["api_key"] == "secret123"
        finally:
            del os.environ["TEST_API_KEY"]

    def test_unset_env_var_kept(self, tmp_path: Path):
        """Unset env vars are kept as-is."""
        conf = tmp_path
        flows = conf / "flows"
        flows.mkdir()

        (flows / "env_test2.yml").write_text("""
flow_id: env_test2
api_key: ${UNSET_VAR}
""")

        config = load_flow_config("env_test2", conf)
        assert config["api_key"] == "${UNSET_VAR}"


class TestFlowConfig:
    """Tests for FlowConfig dataclass."""

    def test_from_dict_basic(self):
        """Create FlowConfig from basic dict."""
        from signalflow.config import FlowConfig

        config = FlowConfig.from_dict({
            "flow_id": "test",
            "flow_name": "Test Flow",
            "description": "A test flow",
            "capital": 50000,
            "fee": 0.002,
        })

        assert config.flow_id == "test"
        assert config.flow_name == "Test Flow"
        assert config.description == "A test flow"
        assert config.capital == 50000
        assert config.fee == 0.002

    def test_from_dict_with_detector(self):
        """Create FlowConfig with detector."""
        from signalflow.config import FlowConfig

        config = FlowConfig.from_dict({
            "flow_id": "sma_test",
            "detector": {
                "type": "example/sma_cross",
                "fast_period": 10,
                "slow_period": 20,
            },
        })

        assert config.detector is not None
        assert config.detector.type == "example/sma_cross"
        assert config.detector.params["fast_period"] == 10
        assert config.detector.params["slow_period"] == 20

    def test_from_dict_with_strategy(self):
        """Create FlowConfig with strategy config."""
        from signalflow.config import FlowConfig

        config = FlowConfig.from_dict({
            "flow_id": "grid_test",
            "strategy": {
                "strategy_id": "grid_sma",
                "entry_rules": [
                    {
                        "type": "signal",
                        "base_position_size": 200.0,
                        "max_positions_per_pair": 5,
                        "max_total_positions": 25,
                        "entry_filters": [
                            {
                                "type": "price_distance_filter",
                                "min_distance_pct": 0.02,
                            }
                        ],
                    }
                ],
                "exit_rules": [
                    {
                        "type": "tp_sl",
                        "take_profit_pct": 0.03,
                        "stop_loss_pct": 0.01,
                    }
                ],
            },
        })

        assert config.strategy.strategy_id == "grid_sma"
        assert len(config.strategy.entry_rules) == 1
        assert config.strategy.entry_rules[0].base_position_size == 200.0
        assert len(config.strategy.entry_rules[0].entry_filters) == 1
        assert config.strategy.entry_rules[0].entry_filters[0].type == "price_distance_filter"
        assert len(config.strategy.exit_rules) == 1
        assert config.strategy.exit_rules[0].type == "tp_sl"

    def test_from_dict_with_data(self):
        """Create FlowConfig with data config."""
        from signalflow.config import FlowConfig

        config = FlowConfig.from_dict({
            "flow_id": "data_test",
            "data": {
                "pairs": ["BTCUSDT", "ETHUSDT"],
                "timeframe": "4h",
            },
        })

        assert config.data.pairs == ["BTCUSDT", "ETHUSDT"]
        assert config.data.timeframe == "4h"

    def test_to_backtest_config(self):
        """Convert FlowConfig to BacktestBuilder config format."""
        from signalflow.config import FlowConfig

        flow = FlowConfig.from_dict({
            "flow_id": "grid_sma",
            "detector": {
                "type": "example/sma_cross",
                "fast_period": 60,
            },
            "strategy": {
                "entry_rules": [
                    {
                        "base_position_size": 100.0,
                        "max_total_positions": 10,
                        "entry_filters": [
                            {"type": "price_distance_filter", "min_distance_pct": 0.02}
                        ],
                    }
                ],
                "exit_rules": [
                    {"type": "tp_sl", "take_profit_pct": 0.03, "stop_loss_pct": 0.01}
                ],
            },
            "capital": 10000,
        })

        config = flow.to_backtest_config()

        assert config["strategy_id"] == "grid_sma"
        assert config["capital"] == 10000
        assert "detectors" in config
        assert config["detectors"]["main"]["class_name"] == "example/sma_cross"
        assert "entry" in config
        assert config["entry"]["size"] == 100.0
        assert config["entry"]["entry_filters"][0]["type"] == "price_distance_filter"
        assert "exit" in config
        assert config["exit"]["tp"] == 0.03

    def test_raw_preserved(self):
        """Original raw config is preserved."""
        from signalflow.config import FlowConfig

        raw = {"flow_id": "test", "custom_field": "value"}
        config = FlowConfig.from_dict(raw)

        assert config.raw == raw
        assert config.raw["custom_field"] == "value"


class TestBacktestBuilderFromDict:
    """Tests for BacktestBuilder.from_dict() with flow config format."""

    def test_from_flow_config_detector(self):
        """Parse detector from flow config format."""
        from signalflow.api.builder import BacktestBuilder

        config = {
            "flow_id": "test",
            "detector": {
                "type": "example/sma_cross",
                "fast_period": 10,
                "slow_period": 20,
            },
        }

        builder = BacktestBuilder.from_dict(config)

        # Strategy ID should come from flow_id
        assert builder.strategy_id == "test"
        # Detector should be configured
        assert len(builder._named_detectors) == 1 or builder._detector is not None

    def test_from_flow_config_strategy(self):
        """Parse strategy from flow config format."""
        from signalflow.api.builder import BacktestBuilder

        config = {
            "flow_id": "grid_test",
            "strategy": {
                "strategy_id": "grid_sma",
                "entry_rules": [
                    {
                        "base_position_size": 200.0,
                        "max_total_positions": 25,
                        "max_positions_per_pair": 5,
                    }
                ],
                "exit_rules": [
                    {
                        "type": "tp_sl",
                        "take_profit_pct": 0.03,
                        "stop_loss_pct": 0.01,
                    }
                ],
            },
        }

        builder = BacktestBuilder.from_dict(config)

        assert builder.strategy_id == "grid_sma"
        # Entry config should be set
        assert builder._entry_config.get("size") == 200.0
        assert builder._entry_config.get("max_positions") == 25
        # Exit config should be set
        assert builder._exit_config.get("tp") == 0.03
        assert builder._exit_config.get("sl") == 0.01

    def test_from_builder_config(self):
        """Parse original builder config format (backward compat)."""
        from signalflow.api.builder import BacktestBuilder

        config = {
            "strategy_id": "my_strategy",
            "entry": {
                "size": 100,
                "max_positions": 5,
            },
            "exit": {
                "tp": 0.05,
                "sl": 0.02,
            },
            "capital": 50000,
            "fee": 0.001,
        }

        builder = BacktestBuilder.from_dict(config)

        assert builder.strategy_id == "my_strategy"
        assert builder._entry_config["size"] == 100
        assert builder._exit_config["tp"] == 0.05
        assert builder._capital == 50000
        assert builder._fee == 0.001
