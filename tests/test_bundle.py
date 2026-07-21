"""FlowBundle: write/read/validate the promotion artifact + manifest, and the promote CLI gate."""

import json
import warnings

import pytest
from click.testing import CliRunner

import signalflow as sf
from signalflow.cli.main import main
from signalflow.flow.bundle import read_manifest, validate_bundle, write_bundle

warnings.filterwarnings("ignore", message="X does not have valid feature names")

pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names")


@pytest.fixture(scope="module")
def promotable():
    ds = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-05-01", interval="1h")
    model = sf.ForecastModel(target=sf.FixedHorizon(bars=6), features=sf.FeaturePipe(sf.SMA(20)), n_folds=3).fit(ds)
    flow = sf.Flow(
        name="bundleflow",
        forecasts={"m": model},
        detectors=[sf.ThresholdDetector(forecast="m", p_min=0.5)],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.1)),
    )
    run = flow.backtest(ds, capital=10_000, oos=True)
    assert run.promotable is True
    return flow, run


def test_write_bundle_round_trip(promotable, tmp_path):
    flow, run = promotable
    bundle = tmp_path / "flows" / "bundleflow"
    write_bundle(flow, run, bundle)
    for name in ("flow.yaml", "scorecard.json", "manifest.json"):
        assert (bundle / name).exists()
    assert (bundle / "models").is_dir()
    manifest = read_manifest(bundle)
    assert set(manifest) >= {"manifest_version", "name", "signalflow_version", "models", "run"}
    assert manifest["run"]["promotable"] is True
    assert validate_bundle(bundle) == []


def test_validate_bundle_missing_scorecard(promotable, tmp_path):
    flow, run = promotable
    bundle = tmp_path / "b2"
    write_bundle(flow, run, bundle)
    (bundle / "scorecard.json").unlink()
    assert any("scorecard.json" in p for p in validate_bundle(bundle))


def test_validate_bundle_not_promotable(promotable, tmp_path):
    flow, run = promotable
    bundle = tmp_path / "b3"
    write_bundle(flow, run, bundle)
    manifest = read_manifest(bundle)
    manifest["run"]["promotable"] = False
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("promotable" in p for p in validate_bundle(bundle))


def test_validate_bundle_dropped_model(promotable, tmp_path):
    flow, run = promotable
    bundle = tmp_path / "b4"
    write_bundle(flow, run, bundle)
    manifest = read_manifest(bundle)
    manifest["models"] = {}
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("absent from manifest models" in p for p in validate_bundle(bundle))


def test_promote_valid_bundle_generates_conf(promotable, tmp_path):
    flow, run = promotable
    bundle = tmp_path / "flows" / "bundleflow"
    write_bundle(flow, run, bundle)
    to_dir = tmp_path / "sfprod"
    result = CliRunner().invoke(main, ["promote", str(bundle / "flow.yaml"), "--to", str(to_dir)])
    assert result.exit_code == 0
    assert (to_dir / "services" / "strategy" / "conf" / "strategy" / "bundleflow.yaml").exists()


def test_promote_invalid_bundle_refuses(promotable, tmp_path):
    flow, run = promotable
    bundle = tmp_path / "b5"
    write_bundle(flow, run, bundle)
    manifest = read_manifest(bundle)
    manifest["run"]["promotable"] = False
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = CliRunner().invoke(main, ["promote", str(bundle / "flow.yaml"), "--to", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "promotable" in result.output


def test_promote_force_overrides_invalid(promotable, tmp_path):
    flow, run = promotable
    bundle = tmp_path / "b6"
    write_bundle(flow, run, bundle)
    manifest = read_manifest(bundle)
    manifest["run"]["promotable"] = False
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    to_dir = tmp_path / "out6"
    result = CliRunner().invoke(main, ["promote", str(bundle / "flow.yaml"), "--to", str(to_dir), "--force"])
    assert result.exit_code == 0
    assert (to_dir / "services" / "strategy" / "conf" / "strategy" / "bundleflow.yaml").exists()
