# Custom Detector Development Guide

This guide covers how to develop custom signal detectors in SignalFlow.

## Overview

A **SignalDetector** is a component that analyzes market data and generates trading signals. The detector pipeline:

```
RawDataView → preprocess() → detect() → Signals
```

1. **preprocess()**: Extract features from raw OHLCV data
2. **detect()**: Apply detection logic and generate signals
3. **Signals**: Container with detected signals

## Quick Start

### Minimal Detector

```python
from dataclasses import dataclass
import polars as pl

from signalflow.core import Signals, SignalType, sf_component
from signalflow.detector import SignalDetector


@dataclass
@sf_component(name="my/rsi_oversold")
class RsiOversoldDetector(SignalDetector):
    """Detect oversold conditions using RSI."""

    rsi_period: int = 14
    oversold_threshold: float = 30.0

    def __post_init__(self):
        # Setup features for RSI computation
        from signalflow.feature import ExampleRsiFeature
        self.features = ExampleRsiFeature(period=self.rsi_period)
        self.rsi_col = f"RSI_{self.rsi_period}"

    def detect(self, features: pl.DataFrame, context=None) -> Signals:
        signals_df = (
            features
            .filter(pl.col(self.rsi_col) < self.oversold_threshold)
            .select([
                self.pair_col,
                self.ts_col,
                pl.lit(SignalType.RISE.value).alias("signal_type"),
                pl.lit(1).alias("signal"),
            ])
        )
        return Signals(signals_df)


# Usage
detector = RsiOversoldDetector(rsi_period=14, oversold_threshold=25)
signals = detector.run(raw_data_view)
```

## Core Concepts

### Signal Categories

Every detector declares which category of signals it produces:

```python
from signalflow.core.enums import SignalCategory

class MyDetector(SignalDetector):
    signal_category = SignalCategory.PRICE_DIRECTION  # default
```

Available categories:

| Category | Description | Example signal_type values |
|----------|-------------|---------------------------|
| `PRICE_DIRECTION` | Price movement direction | `rise`, `fall`, `flat` |
| `PRICE_STRUCTURE` | Price patterns | `local_top`, `local_bottom`, `breakout` |
| `TREND_MOMENTUM` | Trend state | `trend_start`, `trend_reversal`, `overbought` |
| `VOLATILITY` | Volatility regime | `vol_high`, `vol_low`, `vol_expansion` |
| `VOLUME_LIQUIDITY` | Volume patterns | `volume_spike`, `accumulation` |
| `MARKET_WIDE` | Cross-pair events | `market_crash`, `correlation_shift` |
| `ANOMALY` | Anomalous events | `black_swan`, `flash_crash` |

### Signal Types

For `PRICE_DIRECTION` category, use the `SignalType` enum:

```python
from signalflow.core import SignalType

pl.lit(SignalType.RISE.value).alias("signal_type")  # "rise"
pl.lit(SignalType.FALL.value).alias("signal_type")  # "fall"
pl.lit(SignalType.FLAT.value).alias("signal_type")  # "flat"
```

For other categories, use string values:

```python
# Custom signal types for VOLATILITY category
pl.lit("vol_high").alias("signal_type")
pl.lit("vol_low").alias("signal_type")
```

### Allowed Signal Types

Declare which signal_type values your detector produces:

```python
from dataclasses import field

@dataclass
class MyVolatilityDetector(SignalDetector):
    signal_category = SignalCategory.VOLATILITY

    # Declare allowed values (for validation)
    allowed_signal_types: set[str] | None = field(
        default_factory=lambda: {"vol_high", "vol_low"}
    )
```

## Features Integration

### Using Built-in Features

```python
from signalflow.feature import ExampleSmaFeature, ExampleRsiFeature, FeaturePipeline

@dataclass
class SmaCrossDetector(SignalDetector):
    fast_period: int = 10
    slow_period: int = 20

    def __post_init__(self):
        # Option 1: List of features
        self.features = [
            ExampleSmaFeature(period=self.fast_period),
            ExampleSmaFeature(period=self.slow_period),
        ]

        # Option 2: FeaturePipeline
        self.features = FeaturePipeline([
            ExampleSmaFeature(period=self.fast_period),
            ExampleSmaFeature(period=self.slow_period),
        ])

        # Option 3: Single feature
        self.features = ExampleRsiFeature(period=14)
```

### No Features (Raw OHLCV)

If `features=None` (default), `preprocess()` returns raw OHLCV data:

```python
@dataclass
class SimplePriceDetector(SignalDetector):
    # features = None  # default - raw OHLCV

    def detect(self, features: pl.DataFrame, context=None) -> Signals:
        # features has: pair, timestamp, open, high, low, close, volume
        ...
```

## Advanced Patterns

### Override preprocess()

Add helper columns for your detection method:

```python
@dataclass
class ZScoreDetector(SignalDetector):
    target_feature: str = "close"
    rolling_window: int = 100
    threshold: float = 3.0

    def preprocess(self, raw_data_view, context=None) -> pl.DataFrame:
        # 1. Base preprocessing (OHLCV + features)
        df = super().preprocess(raw_data_view, context)

        # 2. Add helper columns
        df = df.with_columns([
            pl.col(self.target_feature)
                .rolling_mean(window_size=self.rolling_window)
                .over(self.pair_col)
                .alias("_rolling_mean"),
            pl.col(self.target_feature)
                .rolling_std(window_size=self.rolling_window)
                .over(self.pair_col)
                .alias("_rolling_std"),
        ])

        return df

    def detect(self, features: pl.DataFrame, context=None) -> Signals:
        z_score = (pl.col(self.target_feature) - pl.col("_rolling_mean")) / pl.col("_rolling_std")

        signals_df = (
            features
            .filter(z_score.abs() > self.threshold)
            .select([
                self.pair_col,
                self.ts_col,
                pl.when(z_score > self.threshold)
                    .then(pl.lit("anomaly_high"))
                    .otherwise(pl.lit("anomaly_low"))
                    .alias("signal_type"),
                pl.lit(1).alias("signal"),
                (z_score.abs() / self.threshold).clip(0, 1).alias("probability"),
            ])
        )
        return Signals(signals_df)
```

### Market-Wide Detectors

Detect cross-pair events (all pairs affected simultaneously):

```python
@dataclass
@sf_component(name="my/market_panic")
class MarketPanicDetector(SignalDetector):
    signal_category = SignalCategory.MARKET_WIDE
    allowed_signal_types: set[str] | None = field(
        default_factory=lambda: {"market_panic"}
    )

    agreement_threshold: float = 0.9
    min_pairs: int = 5

    def preprocess(self, raw_data_view, context=None) -> pl.DataFrame:
        df = super().preprocess(raw_data_view, context)

        # Add log returns
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1))
            .log()
            .over(self.pair_col)
            .alias("_ret")
        )
        return df

    def detect(self, features: pl.DataFrame, context=None) -> Signals:
        # Compute cross-pair agreement per timestamp
        agreement = (
            features
            .filter(pl.col("_ret").is_not_null())
            .group_by(self.ts_col)
            .agg([
                pl.col("_ret").count().alias("_n_pairs"),
                (pl.col("_ret") < -0.01).sum().alias("_n_falling"),
            ])
            .filter(pl.col("_n_pairs") >= self.min_pairs)
            .with_columns(
                (pl.col("_n_falling") / pl.col("_n_pairs")).alias("_agreement")
            )
            .filter(pl.col("_agreement") >= self.agreement_threshold)
        )

        # Create signals with synthetic "ALL" pair
        signals_df = agreement.select([
            pl.lit("ALL").alias(self.pair_col),
            self.ts_col,
            pl.lit("market_panic").alias("signal_type"),
            pl.lit(1).alias("signal"),
            pl.col("_agreement").alias("probability"),
        ])

        return Signals(signals_df)
```

### Probability Column

Add probability (confidence) to signals:

```python
def detect(self, features: pl.DataFrame, context=None) -> Signals:
    rsi = pl.col("RSI_14")

    signals_df = (
        features
        .filter((rsi < 30) | (rsi > 70))
        .select([
            self.pair_col,
            self.ts_col,
            pl.when(rsi < 30)
                .then(pl.lit(SignalType.RISE.value))
                .otherwise(pl.lit(SignalType.FALL.value))
                .alias("signal_type"),
            pl.lit(1).alias("signal"),
            # Probability: distance from threshold normalized
            pl.when(rsi < 30)
                .then((30 - rsi) / 30)  # 0 at 30, 1 at 0
                .otherwise((rsi - 70) / 30)  # 0 at 70, 1 at 100
                .clip(0, 1)
                .alias("probability"),
        ])
    )
    return Signals(signals_df)
```

To require probability:

```python
@dataclass
class MyDetector(SignalDetector):
    require_probability: bool = True  # Validation will fail without probability
```

### Keep Latest Signal Per Pair

For real-time trading, keep only the most recent signal:

```python
@dataclass
class RealtimeDetector(SignalDetector):
    keep_only_latest_per_pair: bool = True
```

## Signals Output Schema

Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `pair` | str | Trading pair (e.g., "BTCUSDT") |
| `timestamp` | datetime | Signal timestamp (timezone-naive) |
| `signal_type` | str | Signal type value |

Optional columns:

| Column | Type | Description |
|--------|------|-------------|
| `signal` | int/float | Signal value (e.g., 1 for long, -1 for short) |
| `probability` | float | Confidence score [0, 1] |

## Best Practices

### 1. Use @sf_component decorator

Register your detector for serialization:

```python
@dataclass
@sf_component(name="my_namespace/detector_name")
class MyDetector(SignalDetector):
    ...
```

### 2. Validate parameters in __post_init__

```python
def __post_init__(self):
    if self.threshold <= 0:
        raise ValueError("threshold must be positive")
    if self.window < 2:
        raise ValueError("window must be >= 2")
```

### 3. Filter null values

```python
def detect(self, features: pl.DataFrame, context=None) -> Signals:
    # Always filter nulls before detection
    df = features.filter(
        pl.col("RSI_14").is_not_null() &
        pl.col("close").is_not_null()
    )
    ...
```

### 4. Use .over(pair_col) for per-pair calculations

```python
# Rolling stats per pair
df.with_columns(
    pl.col("close")
        .rolling_mean(window_size=20)
        .over(self.pair_col)  # Important!
        .alias("sma_20")
)
```

### 5. Return empty Signals if no signals

```python
def detect(self, features: pl.DataFrame, context=None) -> Signals:
    signals_df = features.filter(...)  # May be empty
    return Signals(signals_df)  # OK even if empty
```

## Complete Example: Bollinger Bands Detector

```python
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from signalflow.core import RawDataView, Signals, SignalType, sf_component
from signalflow.core.enums import SignalCategory
from signalflow.detector import SignalDetector


@dataclass
@sf_component(name="technical/bollinger_bands")
class BollingerBandsDetector(SignalDetector):
    """Detect overbought/oversold using Bollinger Bands.

    Signals:
        - RISE: Price touches lower band (oversold)
        - FALL: Price touches upper band (overbought)
    """

    signal_category = SignalCategory.PRICE_DIRECTION

    window: int = 20
    num_std: float = 2.0
    price_col: str = "close"

    def preprocess(
        self,
        raw_data_view: RawDataView,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        df = super().preprocess(raw_data_view, context)

        price = pl.col(self.price_col)

        df = df.with_columns([
            price
                .rolling_mean(window_size=self.window)
                .over(self.pair_col)
                .alias("_bb_middle"),
            price
                .rolling_std(window_size=self.window)
                .over(self.pair_col)
                .alias("_bb_std"),
        ]).with_columns([
            (pl.col("_bb_middle") + self.num_std * pl.col("_bb_std")).alias("_bb_upper"),
            (pl.col("_bb_middle") - self.num_std * pl.col("_bb_std")).alias("_bb_lower"),
        ])

        return df

    def detect(
        self,
        features: pl.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> Signals:
        price = pl.col(self.price_col)
        upper = pl.col("_bb_upper")
        lower = pl.col("_bb_lower")

        # Filter valid rows
        df = features.filter(
            pl.col("_bb_upper").is_not_null() &
            pl.col("_bb_lower").is_not_null()
        )

        # Detect touches
        touch_lower = price <= lower
        touch_upper = price >= upper

        signals_df = (
            df
            .filter(touch_lower | touch_upper)
            .select([
                self.pair_col,
                self.ts_col,
                pl.when(touch_lower)
                    .then(pl.lit(SignalType.RISE.value))
                    .otherwise(pl.lit(SignalType.FALL.value))
                    .alias("signal_type"),
                pl.when(touch_lower)
                    .then(pl.lit(1))
                    .otherwise(pl.lit(-1))
                    .alias("signal"),
                # Probability: how far beyond band
                pl.when(touch_lower)
                    .then((lower - price) / pl.col("_bb_std"))
                    .otherwise((price - upper) / pl.col("_bb_std"))
                    .abs()
                    .clip(0, 1)
                    .alias("probability"),
            ])
        )

        return Signals(signals_df)


# Usage
detector = BollingerBandsDetector(window=20, num_std=2.5)
signals = detector.run(raw_data_view)

# Filter for bullish signals
bullish = signals.value.filter(pl.col("signal_type") == SignalType.RISE.value)
```

## Testing Your Detector

```python
import pytest
from datetime import datetime, timedelta
import numpy as np
import polars as pl

from signalflow.core import RawData, RawDataView, Signals


def _make_test_data(n: int = 100, n_pairs: int = 3) -> RawDataView:
    """Generate test OHLCV data."""
    np.random.seed(42)
    base = datetime(2024, 1, 1)
    pairs = [f"PAIR{i}" for i in range(n_pairs)]

    rows = []
    for pair in pairs:
        price = 100.0
        for i in range(n):
            price *= np.exp(np.random.randn() * 0.01)
            rows.append({
                "pair": pair,
                "timestamp": base + timedelta(minutes=i),
                "open": price * 0.999,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "volume": 1000.0,
            })

    df = pl.DataFrame(rows)
    raw = RawData(
        datetime_start=base,
        datetime_end=base + timedelta(minutes=n),
        pairs=pairs,
        data={"spot": df},
    )
    return RawDataView(raw)


class TestMyDetector:
    def test_returns_signals(self):
        raw_view = _make_test_data()
        detector = MyDetector(threshold=2.0)
        signals = detector.run(raw_view)

        assert isinstance(signals, Signals)
        assert "signal_type" in signals.value.columns

    def test_signal_schema(self):
        raw_view = _make_test_data()
        detector = MyDetector()
        signals = detector.run(raw_view)

        df = signals.value
        assert "pair" in df.columns
        assert "timestamp" in df.columns
        assert "signal_type" in df.columns

    def test_no_duplicate_signals(self):
        raw_view = _make_test_data()
        detector = MyDetector()
        signals = detector.run(raw_view)

        df = signals.value
        dups = df.group_by(["pair", "timestamp"]).len().filter(pl.col("len") > 1)
        assert dups.height == 0
```

## See Also

- [SignalDetector API Reference](../api/detector.md)
- [Feature Extraction Guide](./custom_features.md)
- [Market-Wide Detectors](../api/detector.md#market-wide-detection)
