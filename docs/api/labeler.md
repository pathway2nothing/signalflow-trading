# Target Module

Signal labeling strategies for machine learning training. These classes generate
look-ahead labels (direction, return magnitude, volume regime) at various horizons.

!!! info "Module Name"
    The target functionality is implemented in the `signalflow.target` module.

!!! tip "Event Detection"
    Market-wide event detectors are in the [`detector.market`](detector.md#market-wide-detection) module.
    Use `mask_targets_by_signals()` to exclude labels around detected events.

## Base Class

::: signalflow.target.base.Labeler
    options:
      show_root_heading: true
      show_source: true
      members: true

## Labeling Strategies

### Fixed Horizon

::: signalflow.target.fixed_horizon_labeler.FixedHorizonLabeler
    options:
      show_root_heading: true
      show_source: true
      members: true

### Triple Barrier (Dynamic)

::: signalflow.target.triple_barrier_labeler.TripleBarrierLabeler
    options:
      show_root_heading: true
      show_source: true
      members: true

### Take Profit (Symmetric Barrier)

::: signalflow.target.take_profit_labeler.TakeProfitLabeler
    options:
      show_root_heading: true
      show_source: true
      members: true

## Non-Price-Direction Labelers

### Anomaly Labeler

::: signalflow.target.anomaly_labeler.AnomalyLabeler
    options:
      show_root_heading: true
      show_source: true
      members: true

### Volatility Regime Labeler

::: signalflow.target.volatility_labeler.VolatilityRegimeLabeler
    options:
      show_root_heading: true
      show_source: true
      members: true

### Trend Scanning Labeler

::: signalflow.target.trend_scanning.TrendScanningLabeler
    options:
      show_root_heading: true
      show_source: true
      members: true

### Structure Labeler (Local Extrema)

::: signalflow.target.structure_labeler.StructureLabeler
    options:
      show_root_heading: true
      show_source: true
      members: true

### Zigzag Structure Labeler (Global)

::: signalflow.target.structure_labeler.ZigzagStructureLabeler
    options:
      show_root_heading: true
      show_source: true
      members: true

### Volume Regime Labeler

::: signalflow.target.volume_labeler.VolumeRegimeLabeler
    options:
      show_root_heading: true
      show_source: true
      members: true

## Multi-Target Generation

::: signalflow.target.multi_target_generator.MultiTargetGenerator
    options:
      show_root_heading: true
      show_source: true
      members: true

::: signalflow.target.multi_target_generator.HorizonConfig
    options:
      show_root_heading: true
      show_source: true

::: signalflow.target.multi_target_generator.TargetType
    options:
      show_root_heading: true
      show_source: true

## Utility Functions

### mask_targets_by_signals

::: signalflow.target.utils.mask_targets_by_signals
    options:
      show_root_heading: true
      show_source: true

### mask_targets_by_timestamps

::: signalflow.target.utils.mask_targets_by_timestamps
    options:
      show_root_heading: true
      show_source: true
