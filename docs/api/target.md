# Target

Targets define how forward returns are turned into labels, and samplers select
which observations to keep for training.

## Labelers

::: signalflow.Target
    options:
      show_root_heading: true

::: signalflow.FixedHorizon
    options:
      show_root_heading: true

::: signalflow.TripleBarrier
    options:
      show_root_heading: true

## Samplers

::: signalflow.Sampler
    options:
      show_root_heading: true

::: signalflow.SampleSet
    options:
      show_root_heading: true

::: signalflow.UniformSampler
    options:
      show_root_heading: true

::: signalflow.MetaLabelingSampler
    options:
      show_root_heading: true

::: signalflow.CUSUMSampler
    options:
      show_root_heading: true

::: signalflow.UniquenessSampler
    options:
      show_root_heading: true
