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

### With Technical Analysis Indicators

199+ indicators (momentum, volatility, trend, statistics, and more):

```bash
pip install signalflow-ta
```

### With Neural Networks

Deep learning validators (LSTM, GRU, Attention heads) via PyTorch Lightning:

```bash
pip install signalflow-nn
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
import signalflow
from signalflow.core import RawData, Signals
from signalflow.detector import ExampleSmaCrossDetector

print(f"SignalFlow {signalflow.__version__} installed")
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

**GPU not detected (for signalflow-nn)?**
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

-   :material-rocket-launch:{ .lg .middle } **[Quick Start](../quickstart.md)**

    ---

    Build your first strategy in 10 minutes

-   :material-puzzle:{ .lg .middle } **[Ecosystem](../ecosystem/index.md)**

    ---

    signalflow-ta and signalflow-nn extensions

</div>

---

**Need help?** [pathway2nothing@gmail.com](mailto:pathway2nothing@gmail.com)