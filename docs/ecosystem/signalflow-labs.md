---
title: signalflow-labs
description: Deep learning extension for neural signal models in SignalFlow
---

# signalflow-labs - Neural Networks

**signalflow-labs** extends SignalFlow with deep learning models built on
PyTorch and Lightning. It provides a composable architecture where encoders and
classification heads are mixed and matched via the component registry.

---

## Installation

```bash
pip install signalflow-labs
```

Requires `signalflow-trading`, `torch >= 2.2`, and `lightning >= 2.5`.

For GPU support:
```bash
# Check CUDA version: nvidia-smi
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install signalflow-labs
```

---

## Architecture

signalflow-labs uses an **Encoder + Head** composition pattern:

```
Input [batch, seq_len, features]
  → Encoder (LSTM, GRU, Transformer, TCN, PatchTST, ...)
    → [batch, embedding_size]
  → Head (MLP, Attention, Residual, Ordinal, ...)
    → [batch, num_classes]
```

Components are loaded from the signalflow registry, making architectures fully
configurable:

```python
from signalflow.labs.model import TemporalClassificator

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

### Training a Neural Model

```python
from pathlib import Path
from signalflow.labs.model import TemporalClassificator
from signalflow.labs.data import TimeSeriesPreprocessor, ScalerConfig

# Configure preprocessing
preprocessor = TimeSeriesPreprocessor(
    default_config=ScalerConfig(method="robust", scope="group")
)

# Create model
model = TemporalClassificator(
    encoder_type="encoder/lstm",
    encoder_params={"input_size": 10, "hidden_size": 64, "num_layers": 2},
    head_type="head/cls/mlp",
    head_params={"hidden_sizes": [128]},
    num_classes=3,
    preprocessor=preprocessor,
    window_size=60,
)

# Train on signalflow DataFrames
model.fit(X_train, y_train, log_dir=Path("./logs"))

# Save/load model
model.save("model.pkl")
loaded = TemporalClassificator.load("model.pkl")
```

### Using Lightning Directly

```python
import lightning as L
from signalflow.labs.model import TemporalClassificator
from signalflow.labs.data import SignalDataModule

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

| Class | Registry Name | Architecture |
|-------|--------------|-------------|
| `LSTMEncoder` | `encoder/lstm` | Bidirectional LSTM |
| `GRUEncoder` | `encoder/gru` | Gated Recurrent Unit |
| `TransformerEncoder` | `encoder/transformer` | Self-attention + positional encoding |
| `PatchTSTEncoder` | `encoder/patchtst` | Patch-based Transformer |
| `TCNEncoder` | `encoder/tcn` | Temporal Convolutional Network |
| `TSMixerEncoder` | `encoder/tsmixer` | All-MLP mixer |
| `InceptionTimeEncoder` | `encoder/inception_time` | Multi-scale convolutions |
| `ResNet1dEncoder` | `encoder/resnet1d` | 1D ResNet |
| `XceptionTimeEncoder` | `encoder/xception_time` | Depthwise separable conv |
| `Conv1dEncoder` | `encoder/conv1d` | 1D CNN |
| `XCMEncoder` | `encoder/xcm` | Cross-Channel Mixing |
| `gMLPEncoder` | `encoder/gmlp` | Gating MLP |
| `OmniScaleCNNEncoder` | `encoder/omniscale` | Multi-scale CNN |
| `ConvTranEncoder` | `encoder/convtran` | Conv + Transformer hybrid |

**Common parameters:**

- `input_size` - Number of input features per timestep
- `hidden_size` / `d_model` - Hidden dimensionality (default: 64)
- `num_layers` - Number of stacked layers (default: 2)
- Transformer-specific: `nhead`, `dim_feedforward`, `dropout`

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
from signalflow.labs.data import TimeSeriesPreprocessor, ScalerConfig

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
from signalflow.labs.data import SignalWindowDataset

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
from signalflow.labs.data import SignalDataModule

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

Model size presets (`small`, `medium`, `large`) control search ranges for hidden
dimensions, layer counts, and learning rates.

---

## Links

- [:material-github: GitHub Repository](https://github.com/pathway2nothing/signalflow-labs)
- [:material-package: PyPI Package](https://pypi.org/project/signalflow-labs/)
