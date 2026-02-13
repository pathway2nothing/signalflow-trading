"""Tests for CLI commands."""

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from signalflow.cli.main import cli


@pytest.fixture
def runner():
    """Create CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_config_file():
    """Create a sample config file for testing."""
    content = """
strategy:
  id: test_strategy
data:
  source: test.duckdb
  pairs: [BTCUSDT]
  start: "2024-01-01"
detector:
  name: example/sma_cross
  params:
    fast_period: 20
    slow_period: 50
capital: 10000
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        path = f.name

    yield path

    # Cleanup
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def invalid_config_file():
    """Create an invalid config file for testing."""
    content = """
strategy:
  id: invalid_test
# Missing data and detector
capital: 10000
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        path = f.name

    yield path

    Path(path).unlink(missing_ok=True)


class TestCLIHelp:
    """Tests for CLI help commands."""

    def test_main_help(self, runner):
        """Main help shows commands."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "SignalFlow" in result.output
        assert "run" in result.output
        assert "list" in result.output
        assert "validate" in result.output
        assert "init" in result.output

    def test_version(self, runner):
        """Version flag works."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        # Should show version number
        assert "." in result.output


class TestListCommand:
    """Tests for list command."""

    def test_list_detectors(self, runner):
        """List detectors works."""
        result = runner.invoke(cli, ["list", "detectors"])
        assert result.exit_code == 0
        assert "detector" in result.output.lower() or "Available" in result.output

    def test_list_detectors_verbose(self, runner):
        """List detectors verbose works."""
        result = runner.invoke(cli, ["list", "detectors", "-v"])
        assert result.exit_code == 0

    def test_list_metrics(self, runner):
        """List metrics works."""
        result = runner.invoke(cli, ["list", "metrics"])
        assert result.exit_code == 0

    def test_list_features(self, runner):
        """List features works."""
        result = runner.invoke(cli, ["list", "features"])
        assert result.exit_code == 0

    def test_list_all(self, runner):
        """List all works."""
        result = runner.invoke(cli, ["list", "all"])
        assert result.exit_code == 0
        assert "Registry" in result.output


class TestValidateCommand:
    """Tests for validate command."""

    def test_validate_valid_config(self, runner, sample_config_file):
        """Validate passes for valid config."""
        result = runner.invoke(cli, ["validate", sample_config_file])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_invalid_config(self, runner, invalid_config_file):
        """Validate fails for invalid config."""
        result = runner.invoke(cli, ["validate", invalid_config_file])
        assert result.exit_code != 0
        assert "ERROR" in result.output

    def test_validate_nonexistent_file(self, runner):
        """Validate fails for nonexistent file."""
        result = runner.invoke(cli, ["validate", "/nonexistent/file.yaml"])
        assert result.exit_code != 0


class TestInitCommand:
    """Tests for init command."""

    def test_init_creates_file(self, runner):
        """Init creates config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "backtest.yaml"
            result = runner.invoke(cli, ["init", "--output", str(output)])
            assert result.exit_code == 0
            assert output.exists()
            assert "Created" in result.output

    def test_init_default_filename(self, runner):
        """Init uses default filename."""
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert Path("backtest.yaml").exists()

    def test_init_no_overwrite(self, runner):
        """Init refuses to overwrite without force."""
        with runner.isolated_filesystem():
            # Create file first
            Path("backtest.yaml").write_text("existing")
            result = runner.invoke(cli, ["init"])
            assert result.exit_code != 0
            assert "exists" in result.output.lower()

    def test_init_force_overwrite(self, runner):
        """Init overwrites with force flag."""
        with runner.isolated_filesystem():
            # Create file first
            Path("backtest.yaml").write_text("existing")
            result = runner.invoke(cli, ["init", "--force"])
            assert result.exit_code == 0
            content = Path("backtest.yaml").read_text()
            assert "SignalFlow" in content  # New content, not "existing"

    def test_init_generates_valid_config(self, runner):
        """Init generates parseable config."""
        import yaml

        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            content = Path("backtest.yaml").read_text()
            parsed = yaml.safe_load(content)
            assert "strategy" in parsed
            assert "data" in parsed
            assert "detector" in parsed


class TestRunCommand:
    """Tests for run command."""

    def test_run_invalid_config(self, runner, invalid_config_file):
        """Run fails for invalid config."""
        result = runner.invoke(cli, ["run", invalid_config_file])
        assert result.exit_code != 0
        assert "ERROR" in result.output

    def test_run_nonexistent_file(self, runner):
        """Run fails for nonexistent file."""
        result = runner.invoke(cli, ["run", "/nonexistent/file.yaml"])
        assert result.exit_code != 0

    def test_run_help(self, runner):
        """Run help shows options."""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output
        assert "--quiet" in result.output
        assert "--plot" in result.output
