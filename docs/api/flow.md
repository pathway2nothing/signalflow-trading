# Flow & Live

`Flow` wires data, detectors, strategy, and engine into a single runnable
pipeline. A `Run` captures the result of executing a flow. Feeds drive the flow
over historical, replayed, or live data.

## Flow

::: signalflow.Flow
    options:
      show_root_heading: true

::: signalflow.Run
    options:
      show_root_heading: true

## Feeds

::: signalflow.LiveFeed
    options:
      show_root_heading: true

::: signalflow.ReplayFeed
    options:
      show_root_heading: true

::: signalflow.PollingFeed
    options:
      show_root_heading: true

## Live loop

::: signalflow.run_live_loop
    options:
      show_root_heading: true
