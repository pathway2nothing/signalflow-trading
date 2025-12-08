# Installation

## Requirements

- Python 3.10 or higher
- pip or poetry package manager

## Standard Installation

Install SignalFlow using pip:

```bash
pip install signal-flow
```

## Installation with GPU Support

For GPU-accelerated operations (recommended for large datasets):

```bash
pip install signal-flow[gpu]
```

This installs cuML and PyTorch with CUDA support for significantly faster computations.

## Installation with All Extras

For the complete package including development tools:

```bash
pip install signal-flow[all]
```

## Development Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/pathway2nothing/signal-flow.git
cd signal-flow
pip install -e ".[dev]"
```

## Verify Installation

```python
import signalflow
print(signalflow.__version__)
```

## Optional Dependencies

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| `cuml` | GPU acceleration | `pip install signal-flow[gpu]` |
| `mlflow` | Experiment tracking | `pip install signal-flow[mlflow]` |
| `optuna` | Hyperparameter tuning | `pip install signal-flow[optuna]` |
| `plotly` | Interactive charts | `pip install signal-flow[viz]` |

!!! tip "Recommended Setup"
    For production use, we recommend installing with GPU support and MLflow integration:
    ```bash
    pip install signal-flow[gpu,mlflow]
    ```

## Next Steps

Once installed, head to the [Quick Start](quickstart.md) guide to create your first strategy.