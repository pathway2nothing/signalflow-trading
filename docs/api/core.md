# Core Module

::: signalflow.core
    options:
      show_root_heading: true
      show_source: true
      members:
        - RawData
        - Signals
        - SignalType
        - SignalFlowRegistry
        - sf_component

## Event log (source of truth)

::: signalflow.core.eventlog
    options:
      show_root_heading: true
      members:
        - CashPolicy
        - apply_fill
        - fold
        - replay_state
        - portfolios_match

## Warmup contract

::: signalflow.core.warmup
    options:
      show_root_heading: true
      members:
        - warmup_bars_of
        - required_warmup_bars
        - assert_warmup_consistency

::: signalflow.core.enums
    options:
      show_root_heading: true
      members:
        - SfComponentType
        - DataFrameType
        - RawDataType

::: signalflow.core.registry
    options:
      show_root_heading: true
      show_source: true