# Visualization

Interactive pipeline visualization for SignalFlow, inspired by Kedro-Viz.

!!! info "Module"
    `signalflow.viz` -- Available as `sf.viz.pipeline()`, `sf.viz.features()`, `sf.viz.serve()`.

## Quick Example

```python
import signalflow as sf

# Build a backtest pipeline
builder = (
    sf.Backtest("my_strategy")
    .data(raw=raw_data)
    .detector("example/sma_cross", fast_period=20, slow_period=50)
    .exit(tp=0.03, sl=0.015)
)

# Interactive HTML visualization
sf.viz.pipeline(builder)

# Mermaid diagram (for docs)
mermaid_code = sf.viz.pipeline(builder, format="mermaid", show=False)

# Local development server
sf.viz.serve(builder, port=4141)
```

## Functions

::: signalflow.viz.pipeline
    options:
      show_root_heading: true
      show_source: false

::: signalflow.viz.features
    options:
      show_root_heading: true
      show_source: false

::: signalflow.viz.data_flow
    options:
      show_root_heading: true
      show_source: false

::: signalflow.viz.serve
    options:
      show_root_heading: true
      show_source: false

## Graph

::: signalflow.viz.graph.PipelineGraph
    options:
      show_root_heading: true
      show_source: false
