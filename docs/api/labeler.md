# Target Module

Signal labeling strategies for machine learning training.

!!! info "Module Name"
    The target functionality is implemented in the `signalflow.target` module.

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

::: signalflow.target.triple_barrier.TripleBarrierLabeler
    options:
      show_root_heading: true
      show_source: true
      members: true

### Static Triple Barrier

::: signalflow.target.static_triple_barrier.StaticTripleBarrierLabeler
    options:
      show_root_heading: true
      show_source: true
      members: true