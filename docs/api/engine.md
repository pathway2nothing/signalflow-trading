# Engine

The execution engine applies intents to a broker, producing orders and fills.
Brokers cover simulation and live trading; the order primitives describe what is
sent and what comes back.

## Engine

::: signalflow.Engine
    options:
      show_root_heading: true

## Brokers

::: signalflow.SimBroker
    options:
      show_root_heading: true

::: signalflow.ExchangeBroker
    options:
      show_root_heading: true

::: signalflow.BinanceBroker
    options:
      show_root_heading: true

## Order primitives

::: signalflow.Intent
    options:
      show_root_heading: true

::: signalflow.Order
    options:
      show_root_heading: true

::: signalflow.Fill
    options:
      show_root_heading: true
