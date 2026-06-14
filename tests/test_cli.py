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


def test_list_unknown_type_errors() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["list", "nonsense"])
    assert result.exit_code != 0


def test_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
