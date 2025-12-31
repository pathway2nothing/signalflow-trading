# SignalFlow

**SignalFlow** is a high-performance Python framework for algorithmic trading, designed to manage the full strategy lifecycle from signal detection to execution. It bridges the gap between research and production by providing a robust pipeline for signal generation, meta-labeling validation, and automated trading.

## Core Architecture: The Signal Pipeline

The framework implements a modular three-stage processing logic:

1. 
**🕵️ Signal Detector**: Scans market data (OHLCV or tick) to identify potential market events. Detectors can range from simple SMA crossovers to complex deep learning models.


2. 
**⚖️ Signal Validator (Meta-Labeling)**: Based on Lopez de Prado's methodology, this stage assesses the quality and risk of detected signals using classification models (e.g., LightGBM, XGBoost).


3. 
**♟️ Trading Strategy**: Converts validated signals into actionable trade positions, managing entry, exit, and risk.



## Key Features

* 
**Polars-First Performance**: Core data processing utilizes `polars` for extreme efficiency with large datasets.


* 
**Production Ready**: Code written for research and backtesting is designed for direct deployment to live trading.


* 
**Advanced Labeling**: Native support for Triple-Barrier Method and Fixed-Horizon labeling for ML training.


* 
**Kedro Integration**: Fully compatible with Kedro for reproducible R&D and automated data pipelines.


* 
**Flexible Extensibility**: Easily add custom features via the `@sf_component` registry.



## Quick Start

### Installation

```bash
pip install signalflow-trading

```

### Signal Detection Example

```python
from signalflow.core import RawDataView
from signalflow.detector import SmaCrossSignalDetector

# Initialize a detector (SMA 20/50 crossover)
detector = SmaCrossSignalDetector(fast_period=20, slow_period=50)

# Run detection on a data snapshot
signals = detector.run(raw_data_view)

```

### Signal Validation (Meta-Labeling)

```python
from signalflow.validator import SklearnSignalValidator

# Create a validator using LightGBM
validator = SklearnSignalValidator(model_type="lightgbm")

# Fit the model on labeled historical signals
validator.fit(X_train, y_train)

# Validate new signals to get success probabilities
validated_signals = validator.validate_signals(signals, features)

```

## Tech Stack

* 
**Data**: `polars`, `pandas`, `duckdb`.


* 
**ML/Compute**: `pytorch`, `lightning`, `scikit-learn`, `numba`, `optuna`.


* 
**Technical Analysis**: `pandas-ta`.



## Package Structure

* 
`signalflow.core`: Core data containers (`RawData`, `Signals`) and registries.


* 
`signalflow.data`: Binance API loaders and DuckDB storage.


* 
`signalflow.feature`: Feature extractors and technical indicator adapters.


* 
`signalflow.labeler`: Advanced labeling techniques for machine learning.


* 
`signalflow.detector`: Ready-to-use signal detection algorithms.



---

**License:** MIT 

**Author:** pathway2nothing 

---