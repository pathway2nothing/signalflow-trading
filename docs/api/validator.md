# Validator Module

Signal validators (meta-labelers) predict the quality or risk of trading signals.
In De Prado's terminology, this implements the meta-labeling approach - training a secondary
model to predict whether a primary signal will be successful.

## Available Validators

| Validator | Model | Best For |
|-----------|-------|----------|
| `LightGBMValidator` | LGBMClassifier | Fast training, good defaults |
| `XGBoostValidator` | XGBClassifier | Robust, regularized |
| `RandomForestValidator` | RandomForestClassifier | Ensemble, interpretable |
| `LogisticRegressionValidator` | LogisticRegression | Fast, linear relationships |
| `SVMValidator` | SVC | Small datasets, non-linear |
| `AutoSelectValidator` | Auto | Automatic model selection |

## Quick Start

```python
import polars as pl
from signalflow.validator import LightGBMValidator
from signalflow.core import Signals

# Prepare data - filter to active signals (not NONE)
train_df = train_df.filter(pl.col("signal_type") != "none")

# Create and train validator
validator = LightGBMValidator(n_estimators=200, learning_rate=0.05)
validator.fit(
    train_df.select(["pair", "timestamp"] + feature_cols),
    train_df.select("label"),
    X_val=val_df.select(["pair", "timestamp"] + feature_cols),  # Early stopping
    y_val=val_df.select("label"),
)

# Validate new signals
validated = validator.validate_signals(
    Signals(test_df.select(signal_cols)),
    test_df.select(["pair", "timestamp"] + feature_cols),
)

# Filter to high-confidence predictions
confident = validated.value.filter(pl.col("probability_rise") > 0.7)
```

## Auto Model Selection

Use `AutoSelectValidator` to automatically select the best model:

```python
from signalflow.validator import AutoSelectValidator

validator = AutoSelectValidator()
validator.fit(X_train, y_train)

# Check which model was selected
print(validator.selected_validator)  # e.g., LightGBMValidator
```

## Hyperparameter Tuning

Each validator supports Optuna-based hyperparameter tuning:

```python
from signalflow.validator import RandomForestValidator

validator = RandomForestValidator()
validator.tune_params = {"n_trials": 50, "cv_folds": 5, "timeout": 600}
best_params = validator.tune(X_train, y_train)

# Fit with best params
validator.fit(X_train, y_train)
```

## Base Class

::: signalflow.validator.base.SignalValidator
    options:
      show_root_heading: true
      show_source: true
      members: true

## Sklearn Base

::: signalflow.validator.sklearn_validator.SklearnValidatorBase
    options:
      show_root_heading: true
      show_source: true
      members: true

## LightGBM Validator

::: signalflow.validator.sklearn_validator.LightGBMValidator
    options:
      show_root_heading: true
      show_source: true
      members: true

## XGBoost Validator

::: signalflow.validator.sklearn_validator.XGBoostValidator
    options:
      show_root_heading: true
      show_source: true
      members: true

## Random Forest Validator

::: signalflow.validator.sklearn_validator.RandomForestValidator
    options:
      show_root_heading: true
      show_source: true
      members: true

## Logistic Regression Validator

::: signalflow.validator.sklearn_validator.LogisticRegressionValidator
    options:
      show_root_heading: true
      show_source: true
      members: true

## SVM Validator

::: signalflow.validator.sklearn_validator.SVMValidator
    options:
      show_root_heading: true
      show_source: true
      members: true

## Auto-Select Validator

::: signalflow.validator.sklearn_validator.AutoSelectValidator
    options:
      show_root_heading: true
      show_source: true
      members: true
