# High-Level API

The high-level API provides an ergonomic interface for building and running backtests.

!!! info "Module"
    `signalflow.api` -- Available as `sf.Backtest`, `sf.backtest()`, `sf.load()`.

## Builder

::: signalflow.api.builder.BacktestBuilder
    options:
      show_root_heading: true
      show_source: false
      members:
        - data
        - detector
        - aggregation
        - validator
        - entry
        - exit
        - capital
        - run
        - validate
        - visualize

## Result

::: signalflow.api.result.BacktestResult
    options:
      show_root_heading: true
      show_source: false
      members:
        - summary
        - plot
        - metrics
        - metrics_df
        - trades
        - signals
        - state
        - n_trades
        - win_rate
        - total_return

## Shortcuts

### sf.load()

::: signalflow.api.shortcuts.load
    options:
      show_root_heading: true
      show_source: true

### sf.backtest()

::: signalflow.api.shortcuts.backtest
    options:
      show_root_heading: true
      show_source: true

## Exceptions

::: signalflow.api.exceptions
    options:
      show_root_heading: true
      members: true
      show_source: false
