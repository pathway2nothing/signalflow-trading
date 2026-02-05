---
title: signalflow-nn
description: Neural network extension for deep learning signal validation in SignalFlow
---

# signalflow-nn - Neural Networks

**signalflow-nn** extends SignalFlow with deep learning signal validators
built on PyTorch and Lightning. It provides a composable architecture
where encoders and classification heads are mixed and matched via the
component registry.

---

## Installation

```bash
pip install signalflow-nn
```

Requires `signalflow-trading >= 1.1.0`, `torch >= 2.2`, `lightning >= 2.5`.

For GPU support:
```bash
# Check CUDA version: nvidia-smi
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install signalflow-nn
```

---

## Architecture

signalflow-nn uses an **Encoder + Head** composition pattern:

```
Input [batch, seq_len, features]
  → Encoder (LSTM, GRU)
    → [batch, embedding_size]
  → Head (MLP, Attention, Residual, ...)
    → [batch, num_classes]
```

Components are loaded from the signalflow registry, making architectures
fully configurable:

```python
from signalflow.nn.model import TemporalClassificator

model = TemporalClassificator(
    encoder_type="encoder/lstm",
    encoder_params={"input_size": 10, "hidden_size": 128, "num_layers": 2},
    head_type="head/cls/mlp",
    head_params={"hidden_sizes": [64, 32]},
    num_classes=3,
)
```

---

## Quick Start

### Training a Neural Validator

```python
from pathlib import Path
from signalflow.nn.validator import TemporalValidator
from signalflow.nn.data import TimeSeriesPreprocessor, ScalerConfig

# Configure preprocessing
preprocessor = TimeSeriesPreprocessor(
    default_config=ScalerConfig(method="robust", scope="group")
)

# Create validator
validator = TemporalValidator(
    encoder_type="encoder/lstm",
    encoder_params={"input_size": 10, "hidden_size": 64, "num_layers": 2},
    head_type="head/cls/mlp",
    head_params={"hidden_sizes": [128]},
    num_classes=3,
    preprocessor=preprocessor,
    window_size=60,
)

# Train on signalflow DataFrames
validator.fit(X_train, y_train, log_dir=Path("./logs"))

# Predict - returns Signals with probability columns
validated_signals = validator.validate_signals(signals, features)

# Save/load model
validator.save("model.pkl")
loaded = TemporalValidator.load("model.pkl")
```

### Using TemporalClassificator Directly

```python
import lightning as L
from signalflow.nn.model import TemporalClassificator
from signalflow.nn.data import SignalDataModule

# Create model
model = TemporalClassificator(
    encoder_type="encoder/gru",
    encoder_params={"input_size": 15, "hidden_size": 128},
    head_type="head/cls/attention",
    head_params={"num_heads": 4},
    num_classes=3,
)

# Create data module
data_module = SignalDataModule(
    train_df=train_df,
    val_df=val_df,
    feature_cols=feature_cols,
    label_col="label",
    window_size=60,
)

# Train with Lightning
trainer = L.Trainer(max_epochs=20, accelerator="auto")
trainer.fit(model, data_module)
```

---

## Components

### Encoders

Sequence encoders that process windowed time series into fixed-size embeddings.

| Class | Registry Name | Description |
|-------|--------------|-------------|
| `LSTMEncoder` | `encoder/lstm` | Long Short-Term Memory, configurable depth and bidirectional |
| `GRUEncoder` | `encoder/gru` | Gated Recurrent Unit, faster than LSTM with similar performance |

**Parameters:**

- `input_size` - Number of input features per timestep
- `hidden_size` - Hidden state dimensionality (default: 64)
- `num_layers` - Number of stacked RNN layers (default: 2)
- `bidirectional` - Process sequence in both directions (default: False)

### Classification Heads

Output heads that convert encoder embeddings to class predictions.

| Class | Registry Name | Description |
|-------|--------------|-------------|
| `MLPClassifierHead` | `head/cls/mlp` | Standard MLP with configurable hidden layers |
| `LinearClassifierHead` | `head/cls/linear` | Single linear projection |
| `ResidualClassifierHead` | `head/cls/residual` | MLP with residual skip connections |
| `AttentionClassifierHead` | `head/cls/attention` | Multi-head self-attention based |

### Specialized Heads

| Class | Registry Name | Description |
|-------|--------------|-------------|
| `DistributionHead` | `head/cls/distribution` | Soft label output via temperature-scaled softmax |
| `OrdinalRegressionHead` | `head/cls/ordinal` | Ordered classification (fall < neutral < rise) |
| `ClassificationWithConfidenceHead` | `head/cls/confidence` | Dual output: class logits + confidence score |

**DistributionHead** is useful for KL-divergence training with soft labels.
**OrdinalRegressionHead** exploits the natural ordering of signal classes.
**ClassificationWithConfidenceHead** allows filtering predictions by model confidence.

---

## Data Pipeline

### TimeSeriesPreprocessor

Per-asset feature scaling with configurable methods:

```python
from signalflow.nn.data import TimeSeriesPreprocessor, ScalerConfig

preprocessor = TimeSeriesPreprocessor(
    default_config=ScalerConfig(
        method="robust",     # robust | standard | minmax
        scope="group",       # group (per-asset) | global
    )
)

# Fit on training data only
preprocessor.fit(train_df, asset_col="pair")

# Transform
train_scaled = preprocessor.transform(train_df)
test_scaled = preprocessor.transform(test_df)
```

### SignalWindowDataset

Creates 3D tensors `[window_size, features]` at signal timestamps only:

```python
from signalflow.nn.data import SignalWindowDataset

dataset = SignalWindowDataset(
    df=scaled_df,
    signal_timestamps=signal_timestamps,
    feature_cols=feature_cols,
    label_col="label",
    window_size=60,
    window_timeframe=1,  # 1 = every bar, 5 = dilated sampling
)
```

### SignalDataModule

Lightning DataModule with flexible splitting strategies:

```python
from signalflow.nn.data import SignalDataModule

dm = SignalDataModule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    feature_cols=feature_cols,
    label_col="label",
    window_size=60,
    batch_size=64,
)
```

Split strategies: **temporal** (chronological), **random**, **pair-based**.

---

## Hyperparameter Tuning

All components support Optuna integration:

```python
import optuna

def objective(trial):
    config = TemporalClassificator.tune(trial, model_size="medium")
    model = TemporalClassificator(**config)
    # train and evaluate...
    return val_accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=3600)
```

Model size presets (`small`, `medium`, `large`) control search ranges
for hidden dimensions, layer counts, and learning rates.

---

## Integration with SignalFlow

`TemporalValidator` inherits from `SignalValidator` and works identically
to `SklearnSignalValidator`:

```python
# Both validators share the same interface
from signalflow.validator import SklearnSignalValidator
from signalflow.nn.validator import TemporalValidator

# Sklearn-based
sklearn_val = SklearnSignalValidator(model_type="lightgbm")
sklearn_val.fit(X_train, y_train)
result = sklearn_val.validate_signals(signals, features)

# Neural network-based
nn_val = TemporalValidator(
    encoder_type="encoder/lstm",
    encoder_params={"input_size": 10, "hidden_size": 64},
    head_type="head/cls/mlp",
    head_params={"hidden_sizes": [64]},
    num_classes=3,
    window_size=60,
)
nn_val.fit(X_train, y_train)
result = nn_val.validate_signals(signals, features)
```

Both return `Signals` with `probability_none`, `probability_rise`,
and `probability_fall` columns.

---

## Links

- [:material-github: GitHub Repository](https://github.com/pathway2nothing/signalflow-nn)
- [:material-package: PyPI Package](https://pypi.org/project/signalflow-nn/)
