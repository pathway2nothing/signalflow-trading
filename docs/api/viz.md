# Visualization

Interactive pipeline visualization for SignalFlow.

!!! info "Module"
    `signalflow.viz` - Available as `sf.viz.pipeline()`, `sf.viz.features()`, `sf.viz.serve()`.

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

### `sf.viz.pipeline(builder, format="html", show=True)`

Generate an interactive flow visualization of the pipeline.

**Parameters:**

- `builder` - A `Backtest` or `FlowBuilder` instance
- `format` - Output format: `"html"` (D3.js interactive) or `"mermaid"` (text diagram)
- `show` - Open in browser automatically (default: `True`)

**Returns:** HTML string or Mermaid code when `show=False`

### `sf.viz.features(pipeline, df)`

Visualize feature distributions and correlations.

**Parameters:**

- `pipeline` - A `FeaturePipeline` instance
- `df` - DataFrame with computed features

### `sf.viz.data_flow(builder)`

Visualize data flow through the pipeline nodes.

**Parameters:**

- `builder` - A `Backtest` or `FlowBuilder` instance

### `sf.viz.serve(builder, port=4141)`

Launch a local development server with live-reloading visualization.

**Parameters:**

- `builder` - A `Backtest` or `FlowBuilder` instance
- `port` - Local server port (default: `4141`)
