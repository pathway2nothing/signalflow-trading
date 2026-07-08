"""CLI smoke tests - exercise the ``sf`` click group via ``CliRunner``."""

from click.testing import CliRunner

import signalflow as sf
from signalflow.cli.main import main


def test_version() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["version"])
    assert result.exit_code == 0
    assert sf.__version__ in result.output


def test_list_all() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["list"])
    assert result.exit_code == 0

    assert "transform" in result.output
    assert "source" in result.output


def test_list_transform_lists_components() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["list", "transform"])
    assert result.exit_code == 0

    assert "sma" in result.output
    assert "threshold" in result.output


def test_list_target_lists_components() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["list", "target"])
    assert result.exit_code == 0

    assert "fixed_horizon" in result.output
    assert "triple_barrier" in result.output


def test_list_all_includes_targets() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["list"])
    assert result.exit_code == 0
    assert "target" in result.output


def test_list_unknown_type_errors() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["list", "nonsense"])
    assert result.exit_code != 0


def test_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0


def test_list_transform_contains_iv_selector_and_is_cp1252_safe() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["list", "transform"])
    assert result.exit_code == 0
    assert "iv_selector" in result.output
    result.output.encode("cp1252", errors="replace")


def test_info_transform_sma_shows_param_default() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["info", "transform", "sma"])
    assert result.exit_code == 0
    assert "length" in result.output
    assert "20" in result.output
    assert "sma_20" in result.output


def test_info_unknown_type_errors() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["info", "nonsense", "sma"])
    assert result.exit_code != 0
