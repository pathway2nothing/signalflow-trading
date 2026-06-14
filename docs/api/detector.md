# Detector

Detectors turn data and model predictions into discrete trading signals.
`SignalDetector` is the base contract; the others are ready-to-use detectors.

## Base

::: signalflow.SignalDetector
    options:
      show_root_heading: true

## Detectors

::: signalflow.SmaCrossDetector
    options:
      show_root_heading: true

::: signalflow.ThresholdDetector
    options:
      show_root_heading: true

::: signalflow.RevertDetector
    options:
      show_root_heading: true

::: signalflow.MarketDropDetector
    options:
      show_root_heading: true
