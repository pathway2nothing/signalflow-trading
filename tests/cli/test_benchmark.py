"""Tests for the benchmark-ta CLI command."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from signalflow.cli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestBenchmarkCommand:
    """Test sf benchmark-ta command."""

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["benchmark-ta", "--help"])
        assert result.exit_code == 0
        assert "Benchmark signalflow-ta" in result.output
        assert "--rows" in result.output
        assert "--runs" in result.output

    def test_runs_with_small_data(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["benchmark-ta", "--rows", "1000", "--runs", "1"])
        assert result.exit_code == 0
        assert "Indicator Benchmark" in result.output
        assert "RSI" in result.output

    def test_verbose_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["benchmark-ta", "--rows", "1000", "--runs", "1", "--verbose"])
        assert result.exit_code == 0

    def test_command_registered(self, runner: CliRunner) -> None:
        """benchmark-ta should appear in sf --help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "benchmark-ta" in result.output
