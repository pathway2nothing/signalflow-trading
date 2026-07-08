# Installation

---

## Requirements

- **Python 3.12+**
- **4GB RAM** minimum (16GB recommended for larger backtests)

---

## Install

```bash
pip install signalflow-trading
```

### Extras

Install extras with `pip install "signalflow-trading[<extra>]"`.

| Extra | Installs | For |
|-------|----------|-----|
| `ta` | signalflow-ta | 248 technical-indicator features + 21 detectors |
| `labs` | signalflow-labs[rl] | neural encoders, RL strategy, torch backends |
| `live` | mlflow, huggingface_hub | model artifact tracking / deploy |
| `llm` | httpx, pydantic | LLM-assisted strategy (any OpenAI-compatible server) |
| `all` | ta + labs + live + llm | everything |
| `dev` | pytest, ruff, mypy | development |
| `test` | pytest | test run only |
| `docs` | mkdocs, mkdocs-material, mkdocstrings | building this site |

```bash
pip install "signalflow-trading[ta]"          # core + technical-analysis plugin
pip install "signalflow-trading[all]"         # everything
```

### Virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install "signalflow-trading[ta]"
```

---

## Verify installation

```python
import signalflow as sf

print(sf.__version__)
print(sf.registry.snapshot())   # {type: [names]} across every installed package
```

---

## Platform notes

=== "Linux"
    Works out of the box.

=== "macOS"
    Supports both Intel and Apple Silicon (M1/M2/M3/M4).

=== "Windows"
    Works in Command Prompt or PowerShell. Set `PYTHONUTF8=1` if Polars table
    output errors on a non-UTF-8 code page.

---

## Next steps

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **[Quick Start](../quickstart.md)**

    ---

    Build and round-trip your first Flow.

-   :material-sitemap:{ .lg .middle } **[Concepts](../concepts.md)**

    ---

    The tier stack and the invariants.

-   :material-puzzle:{ .lg .middle } **[Ecosystem](../ecosystem/index.md)**

    ---

    signalflow-ta and signalflow-labs.

</div>
