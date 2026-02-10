# Feature Module

Feature extraction for technical indicators and derived metrics.

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
