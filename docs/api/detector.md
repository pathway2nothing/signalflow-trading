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

## Event Detection

Exogenous events (regulatory news, rate decisions, black swans) cause correlated price moves
that no feature could predict. Event detectors identify these timestamps so that labels near
them can be masked (set to null), preventing MI estimate pollution.

All detectors share the same base class and masking logic — only the detection algorithm differs.

```
EventDetectorBase (ABC)
    ├── GlobalEventDetector      @sf_component("event_detector/agreement")
    ├── ZScoreEventDetector      @sf_component("event_detector/zscore")
    └── CusumEventDetector       @sf_component("event_detector/cusum")
```

### Usage

```python
from signalflow.detector.event import ZScoreEventDetector, CusumEventDetector

# Z-Score: detects sudden shocks (z-score of aggregate cross-pair return)
zscore_det = ZScoreEventDetector(z_threshold=6.0, rolling_window=500)
events = zscore_det.detect(df)

# CUSUM: detects sustained regime shifts (cumulative sum of deviations)
cusum_det = CusumEventDetector(drift=0.005, cusum_threshold=0.05)
events = cusum_det.detect(df)

# Mask labels near detected events
df_masked = zscore_det.mask_targets(
    df=df,
    event_timestamps=events,
    horizon_configs=horizons,
    target_columns_by_horizon=cols_by_horizon,
)
```

### Base Class

::: signalflow.detector.event.base.EventDetectorBase
    options:
      show_root_heading: true
      show_source: true
      members: true

### Agreement-Based Detector

::: signalflow.detector.event.global_detector.GlobalEventDetector
    options:
      show_root_heading: true
      show_source: true
      members: true

### Z-Score Detector

::: signalflow.detector.event.zscore_detector.ZScoreEventDetector
    options:
      show_root_heading: true
      show_source: true
      members: true

### CUSUM Detector

::: signalflow.detector.event.cusum_detector.CusumEventDetector
    options:
      show_root_heading: true
      show_source: true
      members: true
