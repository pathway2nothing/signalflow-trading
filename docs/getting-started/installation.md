# Installation

Get SignalFlow up and running in minutes.

---

## Requirements

- **Python 3.12+**
- **4GB RAM** minimum (16GB recommended for backtesting)

---

## Install

### Standard Installation

```bash
pip install signalflow-trading
```

### With Neural Networks

For deep learning validators (LSTM, Transformers):

```bash
pip install signalflow-trading[nn]
```

### Virtual Environment (Recommended)

```bash
# Create environment
python -m venv signalflow-env
source signalflow-env/bin/activate  # Windows: signalflow-env\Scripts\activate

# Install
pip install signalflow-trading[nn]
```

---

## Verify Installation

```python
import signalflow
from signalflow.core import RawData, Signals
from signalflow.detector import SmaCrossSignalDetector

print(f"SignalFlow {signalflow.__version__} installed ✓")
```

---

## Platform Notes

=== "Linux"
    Works out of the box.

=== "macOS"
    Supports both Intel and Apple Silicon (M1/M2/M3).

=== "Windows"
    Works in Command Prompt or PowerShell.

---

## Troubleshooting

**Import errors?**
```bash
pip install --force-reinstall signalflow-trading
```

**GPU not detected?**
```bash
# Check CUDA version first: nvidia-smi
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install signalflow-nn
```

**Python version too old?**
```bash
python --version  # Must be 3.12+
```

---

## Next Steps

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **[Quick Start](quickstart.md)**

    ---

    Build your first strategy in 10 minutes

-   :material-cog:{ .lg .middle } **[Configuration](configuration.md)**

    ---

    Configure components and parameters

</div>

---

**Need help?** [pathway2nothing@gmail.com](mailto:pathway2nothing@gmail.com)