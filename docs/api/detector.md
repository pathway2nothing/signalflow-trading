# Detector Module

Signal detectors and event detectors for real-time market analysis.

!!! info "Module Name"
    The detector functionality is implemented in the `signalflow.detector` module.

## Signal Detection

::: signalflow.detector.base.SignalDetector
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.detector.sma_cross.ExampleSmaCrossDetector
    options:
      show_root_heading: true
      show_source: true
      members: true

## Real-Time Detectors

### Anomaly Detector

::: signalflow.detector.anomaly_detector.AnomalyDetector
    options:
      show_root_heading: true
      show_source: true
      members: true

### Volatility Detector

::: signalflow.detector.volatility_detector.VolatilityDetector
    options:
      show_root_heading: true
      show_source: true
      members: true

### Structure Detector (Local Extrema)

::: signalflow.detector.structure_detector.StructureDetector
    options:
      show_root_heading: true
      show_source: true
      members: true

## Market-Wide Detection

Exogenous market-wide signals (regulatory news, rate decisions, black swans) cause correlated price moves
that no feature could predict. Market-wide detectors identify these timestamps so that labels near
them can be masked (set to null), preventing MI estimate pollution.

All detectors extend SignalDetector with `signal_category=MARKET_WIDE`.

```
SignalDetector (signal_category=MARKET_WIDE)
    ├── AgreementDetector         @sf_component("market/agreement")
    ├── MarketZScoreDetector      @sf_component("market/zscore")
    └── MarketCusumDetector       @sf_component("market/cusum")
```

### Usage

```python
from signalflow.detector.market import MarketZScoreDetector, MarketCusumDetector
from signalflow.target.utils import mask_targets_by_signals

# Z-Score: detects sudden shocks (z-score of aggregate cross-pair return)
zscore_det = MarketZScoreDetector(z_threshold=6.0, rolling_window=500)
signals = zscore_det.run(raw_data_view)
# Default signal_type_name: "aggregate_outlier"

# CUSUM: detects sustained regime shifts (cumulative sum of deviations)
cusum_det = MarketCusumDetector(drift=0.005, cusum_threshold=0.05)
signals = cusum_det.run(raw_data_view)
# Default signal_type_name: "structural_break"

# Mask labels near detected signals
df_masked = mask_targets_by_signals(
    df=df,
    signals=signals,
    mask_signal_types={"aggregate_outlier", "structural_break"},
    horizon_bars=60,
)
```

### Agreement-Based Detector

::: signalflow.detector.market.agreement_detector.AgreementDetector
    options:
      show_root_heading: true
      show_source: true
      members: true

### Z-Score Detector

::: signalflow.detector.market.zscore_detector.MarketZScoreDetector
    options:
      show_root_heading: true
      show_source: true
      members: true

### CUSUM Detector

::: signalflow.detector.market.cusum_detector.MarketCusumDetector
    options:
      show_root_heading: true
      show_source: true
      members: true

## Generic Detectors

### Z-Score Anomaly Detector

::: signalflow.detector.zscore_anomaly.ZScoreAnomalyDetector
    options:
      show_root_heading: true
      show_source: true
      members: true

### Percentile Regime Detector

::: signalflow.detector.percentile_regime.PercentileRegimeDetector
    options:
      show_root_heading: true
      show_source: true
      members: true

### Local Extrema Detector

::: signalflow.detector.local_extrema.LocalExtremaDetector
    options:
      show_root_heading: true
      show_source: true
      members: true
