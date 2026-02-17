"""Tests for signalflow.cli.main module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from signalflow.cli.main import (
    _display_result,
    _save_result,
    _show_plots,
    cli,
    init,
    list_all,
    list_detectors,
    list_features,
    list_metrics,
    main,
    run,
    validate,
)


@pytest.fixture
def runner():
    """Create CLI runner for testing."""
    return CliRunner()


class TestCliGroup:
    """Tests for main CLI group."""

    def test_cli_help(self, runner):
        """CLI shows help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "SignalFlow" in result.output

    def test_cli_version(self, runner):
        """CLI shows version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0


class TestRunCommand:
    """Tests for run command."""

    def test_run_config_not_found(self, runner):
        """run fails if config file not found."""
        result = runner.invoke(run, ["nonexistent.yaml"])
        assert result.exit_code != 0

    def test_run_with_valid_config(self, runner, tmp_path):
        """run executes with valid config."""
        # Create minimal config file
        config_content = """
strategy_id: test
data:
  source: test.duckdb
  pairs: [BTCUSDT]
  start: "2024-01-01"
detector:
  name: example/sma_cross
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        # Mock the execution
        mock_config = MagicMock()
        mock_config.validate.return_value = []
        mock_config.strategy_id = "test"
        mock_builder = MagicMock()
        mock_result = MagicMock()
        mock_result.total_return = 0.1
        mock_result.n_trades = 10
        mock_result.win_rate = 0.6
        mock_result.profit_factor = 1.5
        mock_result.initial_capital = 10000
        mock_result.final_capital = 11000
        mock_result.metrics = {}
        mock_builder.run.return_value = mock_result
        mock_config.to_builder.return_value = mock_builder

        with patch("signalflow.cli.config.BacktestConfig") as mock_cls:
            mock_cls.from_yaml.return_value = mock_config
            result = runner.invoke(run, [str(config_path)])

        # Should succeed
        assert result.exit_code == 0

    def test_run_with_errors_in_config(self, runner, tmp_path):
        """run fails if config has errors."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("strategy_id: test")

        mock_config = MagicMock()
        mock_config.validate.return_value = ["ERROR: Missing data"]

        with patch("signalflow.cli.config.BacktestConfig") as mock_cls:
            mock_cls.from_yaml.return_value = mock_config
            result = runner.invoke(run, [str(config_path)])

        assert result.exit_code == 1
        assert "ERROR" in result.output

    def test_run_with_warnings(self, runner, tmp_path):
        """run shows warnings but continues."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("strategy_id: test")

        mock_config = MagicMock()
        mock_config.validate.return_value = ["WARNING: TP/SL ratio low"]
        mock_config.strategy_id = "test"
        mock_builder = MagicMock()
        mock_result = MagicMock()
        mock_result.total_return = 0.1
        mock_result.n_trades = 10
        mock_result.win_rate = 0.6
        mock_result.profit_factor = 1.5
        mock_result.initial_capital = 10000
        mock_result.final_capital = 11000
        mock_result.metrics = {}
        mock_builder.run.return_value = mock_result
        mock_config.to_builder.return_value = mock_builder

        with patch("signalflow.cli.config.BacktestConfig") as mock_cls:
            mock_cls.from_yaml.return_value = mock_config
            result = runner.invoke(run, [str(config_path)])

        assert result.exit_code == 0
        assert "WARNING" in result.output

    def test_run_quiet_mode(self, runner, tmp_path):
        """run --quiet suppresses progress."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("strategy_id: test")

        mock_config = MagicMock()
        mock_config.validate.return_value = []
        mock_config.strategy_id = "test"
        mock_config.show_progress = True
        mock_builder = MagicMock()
        mock_result = MagicMock()
        mock_result.total_return = 0.1
        mock_result.n_trades = 10
        mock_result.win_rate = 0.6
        mock_result.profit_factor = 1.5
        mock_result.initial_capital = 10000
        mock_result.final_capital = 11000
        mock_result.metrics = {}
        mock_builder.run.return_value = mock_result
        mock_config.to_builder.return_value = mock_builder

        with patch("signalflow.cli.config.BacktestConfig") as mock_cls:
            mock_cls.from_yaml.return_value = mock_config
            result = runner.invoke(run, [str(config_path), "--quiet"])

        assert mock_config.show_progress is False

    def test_run_with_output(self, runner, tmp_path):
        """run --output saves results to file."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("strategy_id: test")
        output_path = tmp_path / "results.json"

        mock_config = MagicMock()
        mock_config.validate.return_value = []
        mock_config.strategy_id = "test"
        mock_builder = MagicMock()
        mock_result = MagicMock()
        mock_result.total_return = 0.1
        mock_result.n_trades = 10
        mock_result.win_rate = 0.6
        mock_result.profit_factor = 1.5
        mock_result.initial_capital = 10000
        mock_result.final_capital = 11000
        mock_result.metrics = {}
        mock_result.to_dict.return_value = {"test": "data"}
        mock_builder.run.return_value = mock_result
        mock_config.to_builder.return_value = mock_builder

        with patch("signalflow.cli.config.BacktestConfig") as mock_cls:
            mock_cls.from_yaml.return_value = mock_config
            result = runner.invoke(run, [str(config_path), "-o", str(output_path)])

        assert result.exit_code == 0
        assert output_path.exists()


class TestDisplayResult:
    """Tests for _display_result helper."""

    def test_positive_return(self):
        """Displays positive return in green."""
        mock_result = MagicMock()
        mock_result.total_return = 0.15
        mock_result.n_trades = 10
        mock_result.win_rate = 0.6
        mock_result.profit_factor = 1.5
        mock_result.initial_capital = 10000
        mock_result.final_capital = 11500
        mock_result.metrics = {}

        # Should not raise
        _display_result(mock_result)

    def test_negative_return(self):
        """Displays negative return in red."""
        mock_result = MagicMock()
        mock_result.total_return = -0.10
        mock_result.n_trades = 10
        mock_result.win_rate = 0.4
        mock_result.profit_factor = 0.8
        mock_result.initial_capital = 10000
        mock_result.final_capital = 9000
        mock_result.metrics = {}

        # Should not raise
        _display_result(mock_result)

    def test_with_additional_metrics(self):
        """Displays additional metrics if present."""
        mock_result = MagicMock()
        mock_result.total_return = 0.15
        mock_result.n_trades = 10
        mock_result.win_rate = 0.6
        mock_result.profit_factor = 1.5
        mock_result.initial_capital = 10000
        mock_result.final_capital = 11500
        mock_result.metrics = {"max_drawdown": 0.05, "sharpe_ratio": 1.5}

        # Should not raise
        _display_result(mock_result)


class TestSaveResult:
    """Tests for _save_result helper."""

    def test_save_to_json(self, tmp_path):
        """Saves result to JSON file."""
        from datetime import datetime

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "trades": [],
            "timestamp": datetime(2024, 1, 1),
        }

        output_path = tmp_path / "result.json"
        _save_result(mock_result, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "trades" in content


class TestShowPlots:
    """Tests for _show_plots helper."""

    def test_no_plots_available(self):
        """Shows message when no plots."""
        mock_result = MagicMock()
        mock_result.plot.return_value = None

        # Should not raise
        _show_plots(mock_result)

    def test_shows_figures(self):
        """Shows figures when available."""
        mock_fig = MagicMock()
        mock_result = MagicMock()
        mock_result.plot.return_value = [mock_fig]

        _show_plots(mock_result)
        mock_fig.show.assert_called_once()


class TestListCommands:
    """Tests for list commands."""

    def test_list_detectors_empty(self, runner):
        """list detectors shows message when empty."""
        with patch("signalflow.core.default_registry") as mock_reg:
            mock_reg.list.return_value = []
            result = runner.invoke(cli, ["list", "detectors"])

        assert result.exit_code == 0
        assert "No detectors registered" in result.output

    def test_list_detectors_with_items(self, runner):
        """list detectors shows available detectors."""
        with patch("signalflow.core.default_registry") as mock_reg:
            mock_reg.list.return_value = ["example/sma_cross", "example/rsi"]
            result = runner.invoke(cli, ["list", "detectors"])

        assert result.exit_code == 0
        assert "example/sma_cross" in result.output

    def test_list_detectors_verbose(self, runner):
        """list detectors --verbose shows docstrings."""
        mock_cls = MagicMock()
        mock_cls.__doc__ = "SMA Crossover detector."

        with patch("signalflow.core.default_registry") as mock_reg:
            mock_reg.list.return_value = ["example/sma_cross"]
            mock_reg.get.return_value = mock_cls
            result = runner.invoke(cli, ["list", "detectors", "-v"])

        assert result.exit_code == 0

    def test_list_metrics(self, runner):
        """list metrics shows available metrics."""
        with patch("signalflow.core.default_registry") as mock_reg:
            mock_reg.list.return_value = ["sharpe", "sortino"]
            result = runner.invoke(cli, ["list", "metrics"])

        assert result.exit_code == 0
        assert "sharpe" in result.output or "sortino" in result.output

    def test_list_metrics_empty(self, runner):
        """list metrics shows message when empty."""
        with patch("signalflow.core.default_registry") as mock_reg:
            mock_reg.list.return_value = []
            result = runner.invoke(cli, ["list", "metrics"])

        assert result.exit_code == 0
        assert "No metrics registered" in result.output

    def test_list_features(self, runner):
        """list features shows available features."""
        with patch("signalflow.core.default_registry") as mock_reg:
            mock_reg.list.return_value = ["rsi", "macd"]
            result = runner.invoke(cli, ["list", "features"])

        assert result.exit_code == 0

    def test_list_features_empty(self, runner):
        """list features shows message when empty."""
        with patch("signalflow.core.default_registry") as mock_reg:
            mock_reg.list.return_value = []
            result = runner.invoke(cli, ["list", "features"])

        assert result.exit_code == 0
        assert "No features registered" in result.output

    def test_list_all(self, runner):
        """list all shows all component types."""
        from signalflow.core import SfComponentType

        with patch("signalflow.core.default_registry") as mock_reg:
            mock_reg.list.return_value = ["item1", "item2"]
            result = runner.invoke(cli, ["list", "all"])

        assert result.exit_code == 0
        assert "Registry" in result.output


class TestValidateCommand:
    """Tests for validate command."""

    def test_validate_config_not_found(self, runner):
        """validate fails if config file not found."""
        result = runner.invoke(validate, ["nonexistent.yaml"])
        assert result.exit_code != 0

    def test_validate_valid_config(self, runner, tmp_path):
        """validate shows success for valid config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("strategy_id: test")

        mock_config = MagicMock()
        mock_config.validate.return_value = []

        with patch("signalflow.cli.config.BacktestConfig") as mock_cls:
            mock_cls.from_yaml.return_value = mock_config
            result = runner.invoke(validate, [str(config_path)])

        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_with_errors(self, runner, tmp_path):
        """validate shows errors and exits 1."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("strategy_id: test")

        mock_config = MagicMock()
        mock_config.validate.return_value = ["ERROR: Missing data"]

        with patch("signalflow.cli.config.BacktestConfig") as mock_cls:
            mock_cls.from_yaml.return_value = mock_config
            result = runner.invoke(validate, [str(config_path)])

        assert result.exit_code == 1
        assert "ERROR" in result.output

    def test_validate_with_warnings_only(self, runner, tmp_path):
        """validate shows warnings but exits 0."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("strategy_id: test")

        mock_config = MagicMock()
        mock_config.validate.return_value = ["WARNING: TP/SL ratio low"]

        with patch("signalflow.cli.config.BacktestConfig") as mock_cls:
            mock_cls.from_yaml.return_value = mock_config
            result = runner.invoke(validate, [str(config_path)])

        assert result.exit_code == 0
        assert "WARNING" in result.output

    def test_validate_yaml_parse_error(self, runner, tmp_path):
        """validate handles YAML parse errors."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("invalid: yaml: content")

        with patch("signalflow.cli.config.BacktestConfig") as mock_cls:
            mock_cls.from_yaml.side_effect = Exception("YAML parse error")
            result = runner.invoke(validate, [str(config_path)])

        assert result.exit_code == 1
        assert "parse error" in result.output.lower()


class TestInitCommand:
    """Tests for init command."""

    def test_init_creates_file(self, runner, tmp_path):
        """init creates sample config file."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("signalflow.cli.config.generate_sample_config") as mock_gen:
                mock_gen.return_value = "# Sample config\nstrategy_id: sample"
                result = runner.invoke(init, ["--output", "test.yaml"])

        assert result.exit_code == 0
        assert "Created" in result.output

    def test_init_file_exists_no_force(self, runner, tmp_path):
        """init fails if file exists without --force."""
        config_path = tmp_path / "backtest.yaml"
        config_path.write_text("existing content")

        result = runner.invoke(init, ["--output", str(config_path)])

        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_init_file_exists_with_force(self, runner, tmp_path):
        """init overwrites with --force."""
        config_path = tmp_path / "backtest.yaml"
        config_path.write_text("existing content")

        with patch("signalflow.cli.config.generate_sample_config") as mock_gen:
            mock_gen.return_value = "# New content"
            result = runner.invoke(init, ["--output", str(config_path), "--force"])

        assert result.exit_code == 0
        assert "Created" in result.output

    def test_init_default_filename(self, runner, tmp_path):
        """init uses default filename."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("signalflow.cli.config.generate_sample_config") as mock_gen:
                mock_gen.return_value = "# Sample"
                result = runner.invoke(init)

        assert result.exit_code == 0


class TestMainEntryPoint:
    """Tests for main() entry point."""

    def test_main_calls_cli(self):
        """main() invokes cli group."""
        with patch("signalflow.cli.main.cli") as mock_cli:
            main()
            mock_cli.assert_called_once()
