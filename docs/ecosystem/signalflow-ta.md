---
title: signalflow-ta
description: Technical analysis extension with 199+ indicators for SignalFlow
---

# signalflow-ta - Technical Analysis

**signalflow-ta** extends SignalFlow with 199+ technical analysis indicators
organized into 8 modules. Each indicator is a standard `Feature` class that
integrates directly with `FeaturePipeline` and the component registry.

---

## Installation

```bash
pip install signalflow-ta
```

Requires `signalflow-trading >= 0.3.5`.

---

## Indicator Modules

| Module | Count | Description |
|--------|------:|-------------|
| **Momentum** | 18 | RSI, MACD, Stochastic, ROC, CCI, Williams %R, and kinematics analogs |
| **Overlap** | 26 | SMA, EMA, DEMA, TEMA, HMA, KAMA, ALMA, JMA, and price transforms |
| **Volatility** | 40 | ATR, Bollinger, Keltner, Donchian bands, and energy-based indicators |
| **Volume** | 16 | OBV, A/D, MFI, CMF, KVO, and market dynamics analogs |
| **Trend** | 22 | ADX, Aroon, Supertrend, PSAR, Ichimoku, and physics-based strength |
| **Statistics** | 73 | Dispersion, distribution, memory, cycles, complexity, DSP, regression |
| **Performance** | 2 | Log return, percent return |
| **Divergence** | 2 | RSI divergence, MACD divergence detectors |

---

## Quick Start

### Using Individual Indicators

```python
import signalflow.ta as ta
from signalflow.feature import FeaturePipeline

# Create indicators
rsi = ta.RsiMom(period=14)
bbands = ta.BollingerVol(period=20, num_std=2.0)
atr = ta.AtrVol(period=14)

# Use in pipeline
pipeline = FeaturePipeline(features=[rsi, bbands, atr])
features_df = pipeline.compute(df)
```

### Using Preset Factories

Preset factories provide curated indicator sets:

```python
from signalflow.ta.pipes import (
    momentum_core_pipe,
    volatility_bands_pipe,
    all_ta_pipe,
)
from signalflow.feature import FeaturePipeline

# Compose a custom set
pipeline = FeaturePipeline(features=[
    *momentum_core_pipe(normalized=True),
    *volatility_bands_pipe(),
])

# Or use all indicators at once
full_pipeline = FeaturePipeline(features=all_ta_pipe(normalized=True))
```

**Available factories:**

| Factory | Indicators |
|---------|-----------|
| `smoothers_pipe()` | SMA, EMA, WMA, HMA, DEMA, TEMA, ... |
| `momentum_core_pipe()` | RSI, ROC, MOM, CMO |
| `momentum_oscillators_pipe()` | Stochastic, StochRSI, Williams %R, CCI, UO, AO |
| `momentum_pipe()` | All momentum indicators |
| `volatility_range_pipe()` | True Range, ATR, NATR |
| `volatility_bands_pipe()` | Bollinger, Keltner, Donchian, AccBands |
| `volatility_pipe()` | All volatility indicators |
| `volume_cumulative_pipe()` | OBV, A/D, PVT, NVI, PVI |
| `volume_oscillators_pipe()` | MFI, CMF, EFI, EOM, KVO |
| `volume_pipe()` | All volume indicators |
| `trend_strength_pipe()` | ADX, Aroon, Vortex, VHF, CHOP |
| `trend_stops_pipe()` | PSAR, Supertrend, Chandelier, HiLo, CKSP |
| `trend_pipe()` | All trend indicators |
| `stat_dispersion_pipe()` | Variance, Std, MAD, Z-Score, CV, Range |
| `stat_cycle_pipe()` | Instantaneous Amplitude, Phase, Frequency |
| `stat_pipe()` | All statistical indicators |
| `performance_pipe()` | LogReturn, PctReturn |
| `divergence_pipe()` | RSI divergence, MACD divergence |
| `all_ta_pipe()` | All 199+ indicators |

---

## Indicator Architecture

Every indicator follows a consistent pattern:

```python
from dataclasses import dataclass
from signalflow.core import sf_component
from signalflow.feature.base import Feature

@dataclass
@sf_component(name="momentum/rsi")
class RsiMom(Feature):
    period: int = 14

    requires = ["close"]                # input columns
    outputs = ["rsi_{period}"]          # output column template

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        # Pure NumPy computation on single pair
        ...

    @property
    def warmup(self) -> int:
        return self.period * 10         # minimum stable bars
```

Key design principles:

- **Deterministic**: Pure lookback computation, always reproducible
- **Polars-native**: DataFrame in/out, native rolling operations
- **Warmup support**: Each indicator declares minimum bars for stable output
- **Normalization**: Optional `normalized=True` parameter for value rescaling

### Normalization

Indicators support two normalization strategies:

**Bounded indicators** (RSI, Williams %R, CCI):
```python
rsi = ta.RsiMom(period=14, normalized=True)
# Rescales [0, 100] → [0, 1]
```

**Unbounded indicators** (MACD, SMA, ROC):
```python
sma = ta.SmaSmooth(period=20, normalized=True)
# Rolling z-score normalization
```

### AutoFeatureNormalizer

For automatic per-feature normalization:

```python
from signalflow.ta.auto_norm import AutoFeatureNormalizer

normalizer = AutoFeatureNormalizer(window=256, warmup=256)
df_normalized = normalizer.fit_transform(df)

# Save for production use
normalizer.artifact.save("normalizer.json")
```

The normalizer analyzes each feature's distribution (skewness, outlier ratio, CV)
and selects the best method: rolling robust, rolling z-score, rolling rank,
rolling winsorization, or signed log1p.

---

## Physics-Based Indicators

signalflow-ta includes novel indicators that model market dynamics using
physics analogs:

### Volatility Energy
- **Kinetic Energy**: `KE = 0.5 * v²` - measures price velocity
- **Potential Energy**: `PE = (ln(Close) - ln(MA))²` - measures deviation
- **Total Energy**: `TE = KE + PE` - overall market energy
- **Temperature**: `T = KE / degrees_of_freedom` - thermal regime
- **Free Energy**: `F = E - T*S` - actionable energy (entropy-adjusted)

### Volume Dynamics
- **Market Force**: `F = volume × acceleration` - buying/selling pressure
- **Impulse**: `J = Σ(F × dt)` - accumulated force
- **Market Momentum**: `p = volume × velocity` - directional volume
- **Market Capacitance**: `C = volume / ΔPrice` - liquidity measure

### Trend Physics
- **Viscosity**: Resistance to velocity change
- **Reynolds Number**: Laminar vs turbulent market regime
- **Impedance**: `Z = V/I` electrical analogy for trend resistance

---

## Module Reference

### Momentum

| Class | Component Name | Description |
|-------|---------------|-------------|
| `RsiMom` | `momentum/rsi` | Relative Strength Index |
| `RocMom` | `momentum/roc` | Rate of Change |
| `MomMom` | `momentum/mom` | Momentum |
| `CmoMom` | `momentum/cmo` | Chande Momentum Oscillator |
| `StochMom` | `momentum/stoch` | Stochastic Oscillator |
| `StochRsiMom` | `momentum/stoch_rsi` | Stochastic RSI |
| `WillrMom` | `momentum/willr` | Williams %R |
| `CciMom` | `momentum/cci` | Commodity Channel Index |
| `UoMom` | `momentum/uo` | Ultimate Oscillator |
| `AoMom` | `momentum/ao` | Awesome Oscillator |
| `MacdMom` | `momentum/macd` | MACD |
| `PpoMom` | `momentum/ppo` | Percentage Price Oscillator |
| `TsiMom` | `momentum/tsi` | True Strength Index |
| `TrixMom` | `momentum/trix` | Triple EMA Rate of Change |
| `AccelerationMom` | `momentum/acceleration` | Price Acceleration |
| `JerkMom` | `momentum/jerk` | Rate of Acceleration Change |
| `AngularMomentumMom` | `momentum/angular_momentum` | Rotational Momentum |
| `TorqueMom` | `momentum/torque` | Rate of Angular Momentum |

### Overlap (Smoothers & Price Transforms)

| Class | Component Name | Description |
|-------|---------------|-------------|
| `SmaSmooth` | `overlap/sma` | Simple Moving Average |
| `EmaSmooth` | `overlap/ema` | Exponential Moving Average |
| `WmaSmooth` | `overlap/wma` | Weighted Moving Average |
| `RmaSmooth` | `overlap/rma` | Running Moving Average |
| `DemaSmooth` | `overlap/dema` | Double EMA |
| `TemaSmooth` | `overlap/tema` | Triple EMA |
| `HmaSmooth` | `overlap/hma` | Hull Moving Average |
| `KamaSmooth` | `overlap/kama` | Kaufman Adaptive MA |
| `AlmaSmooth` | `overlap/alma` | Arnaud Legoux MA |
| `JmaSmooth` | `overlap/jma` | Jurik MA |
| `T3Smooth` | `overlap/t3` | Tillson T3 |
| `ZlmaSmooth` | `overlap/zlma` | Zero-Lag MA |
| `McGinleySmooth` | `overlap/mcginley` | McGinley Dynamic |
| `FramaSmooth` | `overlap/frama` | Fractal Adaptive MA |

### Volatility

| Class | Component Name | Description |
|-------|---------------|-------------|
| `AtrVol` | `volatility/atr` | Average True Range |
| `NatrVol` | `volatility/natr` | Normalized ATR |
| `BollingerVol` | `volatility/bollinger` | Bollinger Bands |
| `KeltnerVol` | `volatility/keltner` | Keltner Channels |
| `DonchianVol` | `volatility/donchian` | Donchian Channels |
| `MassIndexVol` | `volatility/mass_index` | Mass Index |
| `UlcerIndexVol` | `volatility/ulcer_index` | Ulcer Index |
| `KineticEnergyVol` | `volatility/kinetic_energy` | Market Kinetic Energy |
| `PotentialEnergyVol` | `volatility/potential_energy` | Market Potential Energy |
| `TemperatureVol` | `volatility/temperature` | Market Temperature |

### Trend

| Class | Component Name | Description |
|-------|---------------|-------------|
| `AdxTrend` | `trend/adx` | Average Directional Index |
| `AroonTrend` | `trend/aroon` | Aroon Indicator |
| `SupertrendTrend` | `trend/supertrend` | Supertrend |
| `PsarTrend` | `trend/psar` | Parabolic SAR |
| `IchimokuTrend` | `trend/ichimoku` | Ichimoku Cloud |
| `VortexTrend` | `trend/vortex` | Vortex Indicator |

### Statistics

| Submodule | Count | Indicators |
|-----------|------:|-----------|
| Dispersion | 9 | Variance, Std, MAD, Z-Score, CV, Range, IQR |
| Distribution | 11 | Median, Quantile, PctRank, Skew, Kurtosis, Entropy |
| Memory | 12 | Hurst, Autocorrelation, Variance Ratio, Diffusion Coefficient |
| Cycle | 9 | Instantaneous Amplitude, Phase, Frequency, Spectral Centroid |
| Complexity | 5 | Permutation Entropy, Sample Entropy, Lempel-Ziv, DFA |
| Information | 5 | KL Divergence, JS Divergence, Renyi Entropy, Mutual Info |
| Regression | 6 | Correlation, Beta, R-Squared, Linear Regression Slope |
| Realized Vol | 5 | Realized, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang |
| DSP | 10 | Spectral Flux, Zero Crossing Rate, Spectral Rolloff, MFCC |
| Control | 5 | Kalman Innovation, AR Coefficient, Lyapunov Exponent |

---

## Links

- [:material-github: GitHub Repository](https://github.com/pathway2nothing/signalflow-ta)
- [:material-package: PyPI Package](https://pypi.org/project/signalflow-ta/)
