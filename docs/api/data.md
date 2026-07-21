# Data

Market data is held in a `Dataset` and produced by data sources. The `data`
helper is a convenience loader that returns a ready-to-use `Dataset`.

## Dataset

::: signalflow.Dataset
    options:
      show_root_heading: true

## Loader

::: signalflow.data
    options:
      show_root_heading: true

## Sources

::: signalflow.BinanceSource
    options:
      show_root_heading: true

::: signalflow.MemorySource
    options:
      show_root_heading: true

## Disk cache

::: signalflow.data.source.cached.CachedSource
    options:
      show_root_heading: true
