# Experiment

Experiments run flows under controlled conditions and summarize their results in
a `Scorecard`. The artifact cache reuses expensive intermediate results, and the
statistics helpers quantify uncertainty in performance metrics.

## Experiment

::: signalflow.Experiment
    options:
      show_root_heading: true

::: signalflow.Scorecard
    options:
      show_root_heading: true

## Artifact cache

::: signalflow.ArtifactCache
    options:
      show_root_heading: true

## Statistics

::: signalflow.bootstrap_ci
    options:
      show_root_heading: true

::: signalflow.monte_carlo_bounds
    options:
      show_root_heading: true
