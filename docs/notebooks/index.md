# Tutorials

Interactive Jupyter notebooks demonstrating SignalFlow capabilities.

## Available Tutorials

| Tutorial | Description |
|----------|-------------|
| [Full Tutorial](tutorial.ipynb) | Complete walkthrough of the SignalFlow pipeline: data loading, feature engineering, signal detection, labeling, validation, and backtesting |
| [Strategy Tutorial](strategy_tutorial.ipynb) | Building and optimizing trading strategies |
| [Labeling Tutorial](labeling_tutorial.ipynb) | Signal labeling techniques and meta-labeling |

## Running Locally

To run these notebooks interactively:

```bash
# Install dependencies
pip install -e ".[dev]"

# Launch Jupyter
jupyter lab notebooks/
```

All notebooks use `VirtualDataProvider` by default, so no exchange API access is required.
