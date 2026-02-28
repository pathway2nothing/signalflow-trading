# Installation

Get SignalFlow up and running in minutes.

---

## Requirements

- **Python 3.12+**
- **4GB RAM** minimum (16GB recommended for backtesting)

---

## Install

### Core Framework

```bash
pip install signalflow-trading
```

### With Technical Analysis (189+ indicators)

```bash
pip install signalflow-ta
```

### With Neural Networks (14 encoders, PyTorch Lightning)

```bash
pip install signalflow-nn
```

### Full Research Stack

```bash
pip install signalflow-trading signalflow-ta signalflow-nn
```

### Virtual Environment (Recommended)

```bash
# Create environment
python -m venv signalflow-env
source signalflow-env/bin/activate  # Windows: signalflow-env\Scripts\activate

# Install core + extensions
pip install signalflow-trading
pip install signalflow-ta    # technical analysis indicators
pip install signalflow-nn    # neural network validators
```

---

## Verify Installation

```python
import signalflow as sf
from signalflow.core import RawData, Signals

print(f"SignalFlow {sf.__version__} installed")

# Check registered components
from signalflow.core import default_registry, SfComponentType
detectors = default_registry.list(SfComponentType.DETECTOR)
print(f"Detectors available: {len(detectors)}")
```

---

## Platform Notes

=== "Linux"
    Works out of the box.

=== "macOS"
    Supports both Intel and Apple Silicon (M1/M2/M3/M4).

=== "Windows"
    Works in Command Prompt or PowerShell.

---

## GPU Support (signalflow-nn)

```bash
# Check CUDA version: nvidia-smi
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install signalflow-nn
```

---

## Troubleshooting

**Import errors?**
```bash
pip install --force-reinstall signalflow-trading
```

**Python version too old?**
```bash
python --version  # Must be 3.12+
```

---

## Next Steps

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **[Quick Start](../quickstart.md)**

    ---

    Build your first strategy in 5 minutes

-   :material-tag-multiple:{ .lg .middle } **[Semantic Decorators](../guide/semantic-decorators.md)**

    ---

    Register custom components with type-safe decorators

-   :material-puzzle:{ .lg .middle } **[Ecosystem](../ecosystem/index.md)**

    ---

    signalflow-ta, signalflow-nn, sf-kedro, sf-ui

</div>
