# Strategy

A strategy decides what to do with signals. `RulesStrategy` composes entry,
exit, and risk rules; `StrategyModel` is the base contract for custom
strategies, and `Observation` is the input passed to each decision.

## Rules strategy

::: signalflow.RulesStrategy
    options:
      show_root_heading: true

## Rules

::: signalflow.Entry
    options:
      show_root_heading: true

::: signalflow.Exit
    options:
      show_root_heading: true

::: signalflow.Risk
    options:
      show_root_heading: true

## Strategy contract

::: signalflow.StrategyModel
    options:
      show_root_heading: true

::: signalflow.Observation
    options:
      show_root_heading: true
