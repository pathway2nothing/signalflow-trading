# Tutorials

Interactive Jupyter notebooks demonstrating SignalFlow capabilities.
All tutorials use synthetic data via `VirtualDataProvider` -- no API keys required.

## Notebooks

### [01 - Quick Start](01_quickstart.ipynb)
Build and run your first backtest in 5 minutes using the fluent builder API.
Covers `sf.Backtest()`, `sf.backtest()`, and `BacktestResult`.

### [02 - Custom Detector](02_custom_detector.ipynb)
Create your own signal detector from scratch. Covers the `SignalDetector` base class,
`@sf_component` registration, multi-detector strategies, and signal aggregation.

### [03 - Data Loading & Resampling](03_data_loading.ipynb)
Load market data from DuckDB stores, auto-detect timeframes, resample OHLCV data,
and work with exchange-specific timeframe support.

### [04 - Pipeline Visualization](04_visualization.ipynb)
Visualize your backtest pipeline as an interactive D3.js graph or Mermaid diagram.
Covers `sf.viz.pipeline()`, `sf.viz.features()`, and the local development server.

### [05 - Advanced Strategies](05_advanced_strategies.ipynb)
Build multi-detector ensemble strategies with signal aggregation, named entry/exit rules,
position sizing, and YAML configuration.

## Running Locally

```bash
# Install dependencies
pip install -e ".[dev]"

# Launch Jupyter
jupyter lab
```

All cell outputs (charts, tables, images) will be rendered automatically on the documentation site.
