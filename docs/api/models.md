# Models

`ForecastModel` produces out-of-sample predictions used by detectors. The
validator combinators aggregate several models into a single decision.

## Forecast model

::: signalflow.ForecastModel
    options:
      show_root_heading: true

## Walk-forward

::: signalflow.walk_forward
    options:
      show_root_heading: true

::: signalflow.WalkForwardResult
    options:
      show_root_heading: true

::: signalflow.WalkForwardFold
    options:
      show_root_heading: true

## Validator combinators

::: signalflow.MeanValidator
    options:
      show_root_heading: true

::: signalflow.MaxValidator
    options:
      show_root_heading: true

::: signalflow.VoteValidator
    options:
      show_root_heading: true
