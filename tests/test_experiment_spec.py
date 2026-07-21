"""experiment.yaml: declarative run through run_experiment (walk-forward, fit, validation)."""

import json
import warnings

import pytest

import signalflow as sf
from signalflow.experiment.spec import load_spec, run_experiment

warnings.filterwarnings("ignore", message="X does not have valid feature names")

pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names")

_SPEC = """
kind: experiment
name: exp_spec_test
seed: 7
data:
  source: memory
  pairs: [BTCUSDT]
  start: "2023-01-01"
  end: "2023-06-01"
  interval: 1h
  cache_dir: null
model:
  backend: lightgbm
  output: p_rise
  target: {name: fixed_horizon, params: {bars: 12}}
  features:
    - {transform: sma, params: {length: 10}}
    - {transform: sma, params: {length: 50}}
  encode: null
  backend_params: {seed: 0, deterministic: true, num_threads: 1, verbosity: -1}
scheme:
  walk_forward: {train: 90d, step: 30d}
metrics: [auc, brier]
backtest:
  capital: 10000
  oos: true
  slot: rise
  detectors:
    - {transform: threshold, params: {forecast: rise, p_min: 0.6}}
  strategy: {name: rules, params: {}}
tracking:
  mlflow: null
"""


def _write(tmp_path, text):
    path = tmp_path / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_run_experiment_end_to_end(tmp_path):
    path = _write(tmp_path, _SPEC)
    result = run_experiment(path)
    assert set(result) == {"folds", "model_scorecard", "run_scorecard", "spec"}
    assert result["folds"] and "auc" in result["folds"][0] and "brier" in result["folds"][0]
    assert result["model_scorecard"]["n"] > 0
    assert "promotable" in result["run_scorecard"]
    assert (tmp_path / "results.json").exists()
    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert saved["spec"]["name"] == "exp_spec_test"


def test_run_experiment_is_deterministic(tmp_path):
    path = _write(tmp_path, _SPEC)
    a = run_experiment(path)["folds"]
    b = run_experiment(path)["folds"]
    assert a == b


def test_unknown_top_level_key_raises(tmp_path):
    path = _write(tmp_path, _SPEC + "\nbogus_key: 1\n")
    with pytest.raises(sf.FlowConfigError, match="bogus_key"):
        load_spec(path)


def test_fit_scheme_returns_model_scorecard(tmp_path):
    fit_spec = _SPEC.replace("scheme:\n  walk_forward: {train: 90d, step: 30d}", "scheme:\n  fit: {}").replace(
        """backtest:
  capital: 10000
  oos: true
  slot: rise
  detectors:
    - {transform: threshold, params: {forecast: rise, p_min: 0.6}}
  strategy: {name: rules, params: {}}
""",
        "",
    )
    path = _write(tmp_path, fit_spec)
    result = run_experiment(path)
    assert result["folds"] is None
    assert result["model_scorecard"]["n"] > 0
