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


def test_list_target_marks_legacy_section() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["list", "target"])
    assert result.exit_code == 0
    assert "legacy" in result.output


def test_target_legacy_flags() -> None:
    assert sf.registry.get_info(sf.ComponentType.TARGET, "fixed_horizon").legacy is False
    assert sf.registry.get_info(sf.ComponentType.TARGET, "triple_barrier").legacy is False
    assert sf.registry.get_info(sf.ComponentType.TARGET, "fixed_horizon_labeler").legacy is True


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


def _make_flow_yaml(tmp_path, scorecard=None):
    import json

    flow = sf.Flow(name="promo", detectors=[sf.SmaCrossDetector(fast=3, slow=8)])
    path = tmp_path / "flow.yaml"
    flow.save(str(path))
    if scorecard is not None:
        (tmp_path / "scorecard.json").write_text(json.dumps(scorecard), encoding="utf-8")
    return str(path)


def test_flow_save_with_run_writes_scorecard(tmp_path) -> None:
    import json

    ds = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-02-01", interval="1h")
    flow = sf.Flow(
        name="ev",
        detectors=[sf.SmaCrossDetector(fast=3, slow=8)],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.1)),
    )
    run = flow.backtest(ds, capital=10_000)
    flow.save(str(tmp_path / "flow.yaml"), run=run)
    sc = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))
    assert "promotable" in sc and sc["name"] == "ev"


def test_promote_without_scorecard_exits(tmp_path) -> None:
    result = CliRunner().invoke(main, ["promote", _make_flow_yaml(tmp_path), "--to", "shadow"])
    assert result.exit_code == 1
    assert "no scorecard.json" in result.output


def test_promote_not_promotable_exits(tmp_path) -> None:
    path = _make_flow_yaml(tmp_path, scorecard={"promotable": False, "oos": True})
    result = CliRunner().invoke(main, ["promote", path, "--to", "shadow"])
    assert result.exit_code == 1
    assert "not promotable" in result.output


def test_promote_generates_conf(tmp_path) -> None:
    to_dir = tmp_path / "sfprod"
    path = _make_flow_yaml(tmp_path, scorecard={"promotable": True, "oos": True, "oos_coverage": 0.97})
    result = CliRunner().invoke(main, ["promote", path, "--to", str(to_dir)])
    assert result.exit_code == 0
    conf = to_dir / "services" / "strategy" / "conf" / "strategy" / "promo.yaml"
    assert conf.exists()
    text = conf.read_text(encoding="utf-8")
    assert "use_v5_flow: true" in text
    assert "flow_path: /app/flows/promo/flow.yaml" in text


def test_promote_dry_run_writes_nothing(tmp_path) -> None:
    to_dir = tmp_path / "sfprod2"
    path = _make_flow_yaml(tmp_path, scorecard={"promotable": True, "oos": True})
    result = CliRunner().invoke(main, ["promote", path, "--to", str(to_dir), "--dry-run"])
    assert result.exit_code == 0
    assert not to_dir.exists()
    assert "use_v5_flow: true" in result.output
