# Feature Module

Feature extraction for technical indicators and derived metrics.

!!! note "v2: where features live"
    `flow` no longer constructs features - the `.features()` builder method was
    removed. Features now live **inside a forecast artefact** (pinned with its
    weights) or as primitive parameters on a detector. The `FeaturePipeline`
    class itself is unchanged: it remains the single **computation engine** and
    can still be used directly to turn a DataFrame into feature columns. The new
    `FeatureSpec` and `ModelFeaturesPipeline` add a reproducibility layer (recipe
    + hash) *around* that engine without duplicating any computation.

## Base Classes

::: signalflow.feature.base.Feature
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.feature.feature_pipeline.FeaturePipeline
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.feature.base.GlobalFeature
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.feature.offset_feature.OffsetFeature
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.feature.lin_reg_forecast.LinRegForecastFeature
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.feature.atr.ATRFeature
    options:
      show_root_heading: true
      show_source: true
      members: true

## Examples

::: signalflow.feature.examples.ExampleRsiFeature
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.feature.examples.ExampleSmaFeature
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.feature.examples.ExampleGlobalMeanRsiFeature
    options:
      show_root_heading: true
      show_source: true
      members: true

## Warmup Reproducibility Contract

Every `Feature` declares whether it can be reproduced identically in live and
backtest after warmup. Two `ClassVar` flags on `Feature` express this:

| Flag | Meaning |
|------|---------|
| `is_recursive` (`ClassVar[bool]`, default `False`) | The indicator is stateful / entry-point dependent: its value at bar *N* depends on where the series starts unless correctly initialized. |
| `warmup_invariant` (`ClassVar[bool]`, default `True`) | The indicator guarantees entry-point invariance after warmup (e.g. deterministic SMA-seeded initialization). For a recursive feature that does **not** re-seed deterministically this must be `False`. |

The existing `warmup` property remains - the minimum number of bars before the
output is stable (default `0`).

`Feature.assert_reproducible()` raises `RuntimeError` for exactly the dangerous
combination `is_recursive and not warmup_invariant` - a feature whose value
depends on the warmup start point, which would diverge between live and
backtest and break parity:

```python
feat.assert_reproducible()   # raises if recursive and not warmup-invariant
```

`FeaturePipeline.assert_reproducible()` delegates to each nested feature and
aggregates all offending feature names into a single error.

---

## FeatureSpec

`FeatureSpec` is the serializable, hashable **recipe** for a `FeaturePipeline`
- it captures *how* to rebuild a pipeline (ordered features + params +
`ta_version` + `raw_data_type`), not the computed values.

```python
from signalflow.feature import FeaturePipeline, ExampleRsiFeature, ExampleSmaFeature
from signalflow.feature.spec import FeatureSpec

pipeline = FeaturePipeline(
    features=[ExampleRsiFeature(period=14), ExampleSmaFeature(period=20)],
    raw_data_type="spot",
)

spec = FeatureSpec.from_pipeline(pipeline, ta_version="1.0.0")
spec.feature_hash()          # stable SHA-256 of the recipe
config = spec.to_config()    # plain dict: {"features": [...], "meta": {...}}
spec2 = FeatureSpec.from_config(config)
assert spec2.feature_hash() == spec.feature_hash()   # round-trips
rebuilt = spec.build()       # -> FeaturePipeline
```

Attributes: `features` (ordered list of `{"name", "params", "scope"}` dicts -
order is significant), `ta_version`, `raw_data_type` (default `"spot"`),
`order_significant` (default `True`, provenance metadata; features are never
reordered).

| Method | Purpose |
|--------|---------|
| `from_pipeline(pipeline, *, ta_version=None)` | Extract a spec from a live pipeline (feature names via registry reverse-lookup). |
| `from_config(data)` / `to_config()` | Round-trip a plain dict (flat or YAML `meta:` nested form). |
| `build()` | Reconstruct a `FeaturePipeline` in the recorded order. |
| `to_yaml(path)` / `from_yaml(path)` | Persist / load the recipe as YAML (survives class refactors, unlike pickle). |
| `feature_hash()` | Stable SHA-256 of the canonical recipe. |

### feature_hash

`canonical_feature_hash(features, ta_version, raw_data_type)` is a pure
function (no I/O, no global state) backing `FeatureSpec.feature_hash()`. The
hash is:

- **identical** for two logically-equal recipes - dict key order is irrelevant,
  float jitter is normalized (e.g. `0.1 == 0.1000000001`), and omitted defaults
  are resolved explicitly (`rsi()` == `rsi(period=14)`);
- **different** whenever something meaningful changes - a param value, the
  **order** of features (never sorted), or the `ta_version` (the same feature
  name across TA library versions is not the same implementation).

It is a configuration-drift detector: recompute it when loading a model
artefact and refuse to continue on mismatch.

```python
from signalflow.feature.spec import canonical_feature_hash

h = canonical_feature_hash(
    features=[{"name": "rsi", "params": {"period": 14}, "scope": "pair"}],
    ta_version="1.0.0",
    raw_data_type="spot",
)
```

::: signalflow.feature.spec.FeatureSpec
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.feature.spec.canonical_feature_hash
    options:
      show_root_heading: true
      show_source: true

---

## ModelFeaturesPipeline

`ModelFeaturesPipeline` is the reproducibility wrapper for train↔serve parity.
It is a **composition** (not a subclass) over a `FeaturePipeline` (the one and
only compute engine) plus a `FeatureSpec` (the serializable recipe). There is
**zero** duplicated feature computation: `compute()` delegates straight into the
wrapped pipeline.

```python
from signalflow.feature import FeaturePipeline, ExampleRsiFeature
from signalflow.feature.model_features import ModelFeaturesPipeline

pipeline = FeaturePipeline(features=[ExampleRsiFeature(period=14)], raw_data_type="spot")
mfp = ModelFeaturesPipeline.from_pipeline(pipeline, ta_version="1.0.0")

# Bundle recipe + hash to store alongside the trained model:
artifact = mfp.to_artifact_dict()    # {"features_config": {...}, "feature_hash": "..."}

# At serve time, rebuild from config and guard against drift:
served = ModelFeaturesPipeline.from_config(artifact["features_config"])
served.verify_hash(artifact["feature_hash"])   # raises RuntimeError on mismatch
served.validate_reproducible()                 # raises if a recursive non-invariant feature is present
features_df = served.compute(df)               # pure delegation to FeaturePipeline.compute
```

| Member | Purpose |
|--------|---------|
| `from_config(data)` | Build the recipe and instantiate the engine from it (single source of truth). |
| `from_pipeline(pipeline, *, ta_version=None)` | Wrap an already-built pipeline, deriving its recipe. |
| `to_config()` | Serialize the recipe (delegates to the spec). |
| `to_artifact_dict()` | Bundle `features_config` + `feature_hash` for storing with a model. |
| `feature_hash` (property) | Stable SHA-256 of the recipe. |
| `validate_reproducible()` | Assert every nested feature honours the warmup-invariance contract. |
| `verify_hash(expected)` | Recompute the hash and raise `RuntimeError` if it differs (drift detector). |
| `compute(df, context=None)` | Compute features by delegating to the wrapped engine (no math here). |

::: signalflow.feature.model_features.ModelFeaturesPipeline
    options:
      show_root_heading: true
      show_source: true
      members: true

---

## Feature Informativeness

Measures how informative each feature is relative to multiple targets at multiple prediction
horizons. Combines MI magnitude with temporal stability into a composite score.

### Usage

```python
from signalflow.feature.informativeness import FeatureInformativenessAnalyzer
from signalflow.detector.market import MarketZScoreDetector

analyzer = FeatureInformativenessAnalyzer(
    event_detector=MarketZScoreDetector(z_threshold=3.0),
)
report = analyzer.analyze(df, feature_columns=["rsi_14", "sma_20", "volume_ratio"])

print(report.top_features(10))      # best features by composite score
print(report.score_matrix)          # NMI heatmap: feature x (horizon, target)
report.feature_detail("rsi_14")     # per-target breakdown for one feature
```

::: signalflow.feature.informativeness.FeatureInformativenessAnalyzer
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.feature.informativeness.InformativenessReport
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.feature.informativeness.RollingMIConfig
    options:
      show_root_heading: true
      show_source: true

::: signalflow.feature.informativeness.CompositeWeights
    options:
      show_root_heading: true
      show_source: true

### Mutual Information Functions

::: signalflow.feature.mutual_information
    options:
      show_root_heading: true
      show_source: true
      members: true
