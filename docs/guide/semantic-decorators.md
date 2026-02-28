---
title: Semantic Decorators
description: Type-safe component registration with @sf.detector, @sf.feature, @sf.entry, and more
---

# Semantic Decorators

Semantic decorators provide type-safe component registration with clear intent.
Instead of the generic `@sf_component`, each decorator maps directly to a
component type — improving readability, IDE support, and discoverability.

---

## Overview

```python
import signalflow as sf

@sf.detector("my/sma_cross")
class SmaCross(SignalDetector):
    def detect(self, data):
        return signals

@sf.feature("my/rsi")
class RsiFeature(Feature):
    def compute_pair(self, df):
        return df_with_rsi

@sf.entry("my/signal_entry")
class MyEntry(SignalEntryRule):
    def should_enter(self, signal):
        return True, size

@sf.exit("my/tp_sl")
class MyExit(ExitRule):
    def should_exit(self, position):
        return exit_signal
```

All decorators share the same signature:

```python
@sf.decorator_name(name: str, *, override: bool = True)
```

- **`name`** — Registry key for the component (case-insensitive)
- **`override`** — Allow overriding an existing registration (default: `True`)

---

## Available Decorators

### Signal Pipeline

| Decorator | Component Type | Base Class | Purpose |
|-----------|---------------|------------|---------|
| `@sf.detector()` | `DETECTOR` | `SignalDetector` | Signal generation |
| `@sf.feature()` | `FEATURE` | `Feature` / `FeatureExtractor` | Feature extraction |
| `@sf.validator()` | `VALIDATOR` | `SignalValidator` | ML signal filtering |
| `@sf.labeler()` | `LABELER` | `Labeler` | Training label generation |

### Strategy Execution

| Decorator | Component Type | Base Class | Purpose |
|-----------|---------------|------------|---------|
| `@sf.entry()` | `STRATEGY_ENTRY_RULE` | `EntryRule` | Position entry logic |
| `@sf.exit()` | `STRATEGY_EXIT_RULE` | `ExitRule` | Position exit logic |
| `@sf.executor()` | `STRATEGY_EXECUTOR` | `Executor` | Backtest/live runner |
| `@sf.risk()` | `STRATEGY_RISK` | `RiskLimit` | Risk constraints |

### Metrics & Monitoring

| Decorator | Component Type | Base Class | Purpose |
|-----------|---------------|------------|---------|
| `@sf.signal_metric()` | `SIGNAL_METRIC` | `SignalMetric` | Signal quality metrics |
| `@sf.strategy_metric()` | `STRATEGY_METRIC` | `StrategyMetric` | Performance metrics |
| `@sf.alert()` | `STRATEGY_ALERT` | `Alert` | Live trading alerts |

### Data Infrastructure

| Decorator | Component Type | Base Class | Purpose |
|-----------|---------------|------------|---------|
| `@sf.data_source()` | `RAW_DATA_SOURCE` | `DataSource` | Exchange APIs, feeds |
| `@sf.data_store()` | `RAW_DATA_STORE` | `DataStore` | OHLCV storage backends |
| `@sf.strategy_store()` | `STRATEGY_STORE` | `StrategyStore` | State persistence |

### Generic

| Decorator | Component Type | Base Class | Purpose |
|-----------|---------------|------------|---------|
| `@sf.register()` | Auto-detected | Any | Infers type from base class |

---

## Examples

### Custom Detector

```python
import signalflow as sf
from signalflow.detector import SignalDetector
from signalflow.core import Signals

@sf.detector("my/momentum_burst")
class MomentumBurst(SignalDetector):
    """Detects sudden momentum acceleration."""

    rsi_threshold: float = 70.0
    volume_multiplier: float = 2.0

    def detect(self, data) -> Signals:
        # Your detection logic
        ...
        return signals
```

### Custom Feature

```python
import polars as pl
import signalflow as sf
from signalflow.feature.base import Feature

@sf.feature("my/spread")
class SpreadFeature(Feature):
    period: int = 20

    requires = ["high", "low", "close"]
    outputs = ["spread_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        spread = (df["high"] - df["low"]) / df["close"]
        return df.with_columns(
            spread.rolling_mean(self.period).alias(f"spread_{self.period}")
        )

    @property
    def warmup(self) -> int:
        return self.period * 2
```

### Custom Entry Rule

```python
import signalflow as sf
from signalflow.strategy.entry import SignalEntryRule

@sf.entry("my/scaled_entry")
class ScaledEntry(SignalEntryRule):
    """Scale position size by signal confidence."""

    base_size: float = 0.1
    confidence_col: str = "probability_rise"

    def calculate_size(self, signal, portfolio):
        confidence = signal.get(self.confidence_col, 0.5)
        return self.base_size * confidence
```

### Custom Data Source

```python
import signalflow as sf
from signalflow.data.source.base import DataSource

@sf.data_source("exchange/my_exchange")
class MyExchangeSource(DataSource):
    """Custom exchange data loader."""

    async def fetch_ohlcv(self, pair, interval, start, end):
        # Fetch from API
        ...
        return raw_data
```

---

## Using Registered Components

Components registered via decorators are accessible through the registry:

```python
from signalflow.core import default_registry, SfComponentType

# Look up by type + name
detector_cls = default_registry.get(SfComponentType.DETECTOR, "my/momentum_burst")
detector = detector_cls(rsi_threshold=65.0)

# List all registered detectors
all_detectors = default_registry.list(SfComponentType.DETECTOR)
```

Or use them directly in the builder API:

```python
result = (
    sf.Backtest("test")
    .data(raw=my_data)
    .detector("my/momentum_burst", rsi_threshold=65.0)
    .entry("my/scaled_entry", base_size=0.15)
    .exit(tp=0.03, sl=0.015)
    .run()
)
```

---

## Entry-Point Autodiscovery

External packages can register components automatically via Python entry points.
Add to your `pyproject.toml`:

```toml
[project.entry-points."signalflow.components"]
my_package = "my_package.components"
```

When SignalFlow loads, it imports `my_package.components`, which triggers
decorator registration. No manual imports needed.

---

## Migration from `@sf_component`

**Before (deprecated):**

```python
from signalflow.core.decorators import sf_component
from signalflow.core.enums import SfComponentType

@sf_component(name="sma_cross")
class SmaCross(SignalDetector):
    component_type = SfComponentType.DETECTOR  # manual, redundant
    ...
```

**After (recommended):**

```python
import signalflow as sf

@sf.detector("sma_cross")
class SmaCross(SignalDetector):
    # component_type set automatically by the decorator
    ...
```

The old `@sf_component` still works but is deprecated. Semantic decorators
automatically set the `component_type` class attribute and register the class
in the global registry.
