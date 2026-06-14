# Transform & Features

Transforms turn raw market data into model-ready features. `Feature` is the
base contract, `FeaturePipe` chains transforms into a pipeline, and the
selection helpers score and prune features by informativeness.

## Transform contract

::: signalflow.Transform
    options:
      show_root_heading: true

::: signalflow.Feature
    options:
      show_root_heading: true

::: signalflow.FeaturePipe
    options:
      show_root_heading: true

## Curated features

::: signalflow.SMA
    options:
      show_root_heading: true

## Encoding and selection

::: signalflow.WoE
    options:
      show_root_heading: true

::: signalflow.Binning
    options:
      show_root_heading: true

::: signalflow.IVSelector
    options:
      show_root_heading: true
