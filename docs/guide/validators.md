# Signal Validators (Meta-Labeling)

This guide covers training and using signal validators - the meta-labeling approach
from Marcos Lopez de Prado for filtering and scoring trading signals.

---

## Overview

A **validator** (meta-labeler) is a secondary model that predicts whether a primary
signal will be successful. This two-stage approach separates:

1. **Detection** (high recall) - find as many potential signals as possible
2. **Validation** (high precision) - filter out false positives

```
Detector → candidate signals → Validator → scored signals → Strategy
```

---

## Available Validators

| Validator | Model | Use Case |
|-----------|-------|----------|
| `LightGBMValidator` | LGBMClassifier | Default choice, fast training |
| `XGBoostValidator` | XGBClassifier | Robust, good regularization |
| `RandomForestValidator` | RandomForestClassifier | Interpretable, no hyperparams |
| `LogisticRegressionValidator` | LogisticRegression | Fast, linear relationships |
| `SVMValidator` | SVC | Small datasets |
| `AutoSelectValidator` | Auto | Automatic model selection |

---

## Quick Start

### 1. Prepare Data

```python
import polars as pl
from signalflow.detector import ExampleSmaCrossDetector
from signalflow.target import FixedHorizonLabeler
from signalflow.feature import FeaturePipeline

# Load your data
raw_data = ...  # RawData with OHLCV

# Generate signals (what we want to validate)
detector = ExampleSmaCrossDetector(fast_period=20, slow_period=50)
signals = detector.run(raw_data.to_view())

# Generate labels (ground truth for training)
labeler = FixedHorizonLabeler(horizon=60, threshold_pct=0.5)
labeled_df = labeler.compute(raw_data.to_polars("spot"))

# Compute features
pipeline = FeaturePipeline([...])
features_df = pipeline.run(raw_data.to_polars("spot"))
```

### 2. Split Data

```python
# Time-based split (no lookahead)
train_end = features_df["timestamp"].quantile(0.7)
val_end = features_df["timestamp"].quantile(0.85)

train_df = features_df.filter(pl.col("timestamp") <= train_end)
val_df = features_df.filter(
    (pl.col("timestamp") > train_end) & (pl.col("timestamp") <= val_end)
)
test_df = features_df.filter(pl.col("timestamp") > val_end)

# Filter to active signals only (not NONE/FLAT)
train_df = train_df.filter(pl.col("signal_type").is_in(["rise", "fall"]))
val_df = val_df.filter(pl.col("signal_type").is_in(["rise", "fall"]))
```

### 3. Train Validator

```python
from signalflow.validator import LightGBMValidator

# Define feature columns (exclude pair, timestamp, label)
feature_cols = [c for c in train_df.columns
                if c not in ["pair", "timestamp", "label", "signal_type"]]

# Create validator
validator = LightGBMValidator(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
)

# Train with early stopping
validator.fit(
    X_train=train_df.select(["pair", "timestamp"] + feature_cols),
    y_train=train_df.select("label"),
    X_val=val_df.select(["pair", "timestamp"] + feature_cols),
    y_val=val_df.select("label"),
)
```

### 4. Validate Signals

```python
from signalflow.core import Signals

# Wrap test signals
test_signals = Signals(test_df.select(["pair", "timestamp", "signal", "signal_type"]))

# Get probabilities
validated = validator.validate_signals(
    signals=test_signals,
    features=test_df.select(["pair", "timestamp"] + feature_cols),
)

# Access results
df = validated.value
print(df.columns)
# ['pair', 'timestamp', 'signal', 'signal_type', 'probability_0', 'probability_1']

# Filter high-confidence signals
confident = df.filter(
    (pl.col("signal_type") == "rise") &
    (pl.col("probability_1") > 0.7)
)
```

---

## Hyperparameter Tuning

Each validator supports Optuna-based hyperparameter tuning:

```python
from signalflow.validator import RandomForestValidator

validator = RandomForestValidator()

# Configure tuning
validator.tune_params = {
    "n_trials": 100,      # Number of Optuna trials
    "cv_folds": 5,        # Cross-validation folds
    "timeout": 1800,      # Max seconds
}
validator.tune_metric = "roc_auc"  # Optimization metric

# Run tuning
best_params = validator.tune(
    X_train=train_df.select(["pair", "timestamp"] + feature_cols),
    y_train=train_df.select("label"),
)

print(f"Best params: {best_params}")

# Fit with best params (already set)
validator.fit(
    X_train=train_df.select(["pair", "timestamp"] + feature_cols),
    y_train=train_df.select("label"),
)
```

### Tune Spaces

Each validator has a predefined tuning space:

**LightGBMValidator:**
- `n_estimators`: 50-500
- `max_depth`: 3-12
- `learning_rate`: 0.01-0.3 (log scale)
- `num_leaves`: 15-127
- `min_child_samples`: 5-100
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0

**RandomForestValidator:**
- `n_estimators`: 50-300
- `max_depth`: 5-30
- `min_samples_split`: 2-20
- `min_samples_leaf`: 1-10

---

## Auto Model Selection

Use `AutoSelectValidator` to automatically find the best model:

```python
from signalflow.validator import AutoSelectValidator

validator = AutoSelectValidator(
    auto_select_metric="roc_auc",
    auto_select_cv_folds=5,
)

validator.fit(X_train, y_train)

# Check selected model
print(f"Selected: {validator.selected_validator.__class__.__name__}")
# e.g., "LightGBMValidator"
```

The auto-selector tests LightGBM, XGBoost, Random Forest, and Logistic Regression
using cross-validation and selects the best performing model.

---

## Early Stopping (Boosting Models)

LightGBM and XGBoost support early stopping to prevent overfitting:

```python
from signalflow.validator import LightGBMValidator

validator = LightGBMValidator(
    n_estimators=1000,  # Max iterations
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
)

# Provide validation data for early stopping
validator.fit(
    X_train=train_df.select(["pair", "timestamp"] + feature_cols),
    y_train=train_df.select("label"),
    X_val=val_df.select(["pair", "timestamp"] + feature_cols),
    y_val=val_df.select("label"),
)

# Check actual iterations used
print(f"Best iteration: {validator.model.best_iteration_}")
```

---

## Save and Load

```python
# Save trained validator
validator.save("models/my_validator.pkl")

# Load later
from signalflow.validator import LightGBMValidator

loaded = LightGBMValidator.load("models/my_validator.pkl")
validated = loaded.validate_signals(signals, features)
```

---

## Integration with Strategy

Use validated signals in your trading strategy:

```python
from signalflow.strategy.component.entry import SignalEntryRule

class ValidatedEntryRule(SignalEntryRule):
    """Entry rule that filters by validator probability."""

    min_probability: float = 0.6

    def check_entries(self, signals, state, context):
        # Filter to high-confidence signals
        confident = signals.value.filter(
            pl.col("probability_1") > self.min_probability
        )

        # Proceed with filtered signals
        return super().check_entries(
            Signals(confident), state, context
        )
```

Or use with `SignalAggregator` in META_LABELING mode:

```python
from signalflow.strategy.component.entry.aggregation import (
    SignalAggregator,
    VotingMode,
)

aggregator = SignalAggregator(
    voting_mode=VotingMode.META_LABELING,
    probability_threshold=0.6,
)

# Combines detector signals with validator probabilities
combined = aggregator.aggregate([detector_signals, validated_signals])
```

---

## Best Practices

### 1. Use Time-Based Splits

Always use time-based train/val/test splits to avoid lookahead bias:

```python
# Good: Time-based split
train = df.filter(pl.col("timestamp") < cutoff)
test = df.filter(pl.col("timestamp") >= cutoff)

# Bad: Random split (causes lookahead)
train, test = train_test_split(df, test_size=0.2)  # Don't do this!
```

### 2. Filter to Active Signals

Only train on signals that require a decision (not NONE/FLAT):

```python
train_df = train_df.filter(pl.col("signal_type").is_in(["rise", "fall"]))
```

### 3. Handle Class Imbalance

If your labels are imbalanced, consider:

```python
# LightGBM with class weights
validator = LightGBMValidator()
validator.model_params["class_weight"] = "balanced"
```

### 4. Feature Engineering

Good features for meta-labeling:
- Volatility metrics (ATR, realized vol)
- Volume indicators
- Market regime features
- Signal confidence from detector
- Time-of-day features

### 5. Monitor Overfitting

Use early stopping and check validation metrics:

```python
validator.fit(X_train, y_train, X_val, y_val)

# Check train vs val performance
from sklearn.metrics import roc_auc_score

train_preds = validator.model.predict_proba(X_train_np)[:, 1]
val_preds = validator.model.predict_proba(X_val_np)[:, 1]

print(f"Train AUC: {roc_auc_score(y_train_np, train_preds):.3f}")
print(f"Val AUC: {roc_auc_score(y_val_np, val_preds):.3f}")
```

---

## See Also

- **[API Reference](../api/validator.md)**: Detailed class documentation
- **[Signal Architecture](signal-architecture.md)**: Meta-labeling theory
- **[Custom Detectors](../guides/custom_detectors.md)**: Building primary models
