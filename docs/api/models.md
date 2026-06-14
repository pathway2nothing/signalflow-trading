# Models Module

`signalflow.models` is the **pinned-inference delivery layer**: declarative, versioned references to forecast-model artefacts plus lazy resolution of their weights.

Forecast models are trained elsewhere and arrive in the trading pipeline as
versioned, reproducible artefacts. This package keeps the trading pipeline
decoupled from training - a reference carries no weights, only enough metadata
to resolve the artefact later.

!!! note "Lazy by design"
    Importing `signalflow.models` does **not** require `mlflow`. Weights load
    only on an explicit `resolve` / `get` call.

---

## Overview

| Component | Role |
|-----------|------|
| `ModelRef` | Declarative, versioned pointer to an artefact (frozen, hashable). |
| `Resolver` (Protocol) | Port: turn a `ModelRef` into a loaded model. |
| `MlflowResolver` | MLflow-backed `Resolver` (lazy `mlflow` import). |
| `ModelRegistry` (Protocol) | Port: fetch loaded models by `ModelRef`. |
| `CachingModelRegistry` | In-process registry that resolves once and caches. |

```python
from signalflow.models import (
    ModelRef,
    Resolver,
    MlflowResolver,
    ModelRegistry,
    CachingModelRegistry,
)
```

---

## ModelRef

A `ModelRef` is a frozen dataclass: `ModelRef(name, version, source="mlflow")`.

- `name` - registered model name (non-empty).
- `version` - **mandatory**. Usually a numeric string/int (`"3"`).
- `source` - backing registry, one of `{"mlflow", "hf"}` (default `"mlflow"`).

### Why version is mandatory

A floating `version="latest"` silently breaks parity and reproducibility
between training and live inference, so it is **rejected** unless the
environment variable `SF_ALLOW_LATEST=1` is set (dev opt-in only).

### Parsing

`ModelRef.parse(spec, *, source="mlflow")` accepts two compact forms:

```python
from signalflow.models import ModelRef

ModelRef.parse("models:/revert/3")   # -> ModelRef(name='revert', version='3', source='mlflow')
ModelRef.parse("revert@3")           # -> ModelRef(name='revert', version='3', source='mlflow')
ModelRef.parse("revert@3", source="hf")  # at-spec uses the given source
```

- `models:/<name>/<version>` always forces `source="mlflow"`.
- `<name>@<version>` uses the `source` argument.

### URI

```python
ModelRef(name="revert", version="3").uri   # "models:/revert/3"
```

---

## Resolver

`Resolver` is a `runtime_checkable` Protocol with one method:

```python
def resolve(self, ref: ModelRef) -> Any: ...
```

`MlflowResolver(tracking_uri=None)` is the MLflow-backed implementation.
Loading is fully lazy: `mlflow` is imported only inside `resolve`, and the
underlying loader is isolated in `_load` so tests can override it without a
real MLflow server. If `tracking_uri` is set it is applied on the first
resolve call. `resolve` raises `ValueError` if `ref.source != "mlflow"`.

```python
from signalflow.models import ModelRef, MlflowResolver

resolver = MlflowResolver(tracking_uri="http://mlflow:5000")
model = resolver.resolve(ModelRef.parse("models:/revert/3"))   # loads weights here
```

---

## ModelRegistry

`ModelRegistry` is the consumer-facing Protocol:

```python
def get(self, ref: ModelRef) -> Any: ...   # resolve lazily if needed
def has(self, ref: ModelRef) -> bool: ...  # already cached?
```

`CachingModelRegistry(resolver)` is a simple, lazy, in-process implementation.
It holds a `Resolver` and a cache keyed by `ModelRef` (frozen → hashable). The
first `get` for a ref triggers resolution; subsequent calls return the cached
artefact without re-resolving. `has` never triggers resolution.

```python
from signalflow.models import ModelRef, MlflowResolver, CachingModelRegistry

registry = CachingModelRegistry(MlflowResolver())
ref = ModelRef.parse("models:/revert/3")

registry.has(ref)   # False
model = registry.get(ref)   # cache miss -> resolves and caches
registry.has(ref)   # True
model is registry.get(ref)  # True (served from cache)
```

---

## See Also

- [Model Integration guide](../guide/model-integration.md) - registering forecast artefacts in a flow via `.forecast()` and consuming them with `forecasts=` / `forecast_window=`.
- [Feature API](feature.md) - `FeatureSpec` / `ModelFeaturesPipeline` and the `feature_hash` drift detector that guards train↔serve reproducibility.

---

## API Reference

::: signalflow.models.model_ref.ModelRef
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.models.resolver.Resolver
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.models.resolver.MlflowResolver
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.models.registry.ModelRegistry
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.models.registry.CachingModelRegistry
    options:
      show_root_heading: true
      show_source: true
      members: true
