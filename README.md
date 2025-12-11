# SignalFlow

**SignalFlow** is an algorithmic trading framework designed to streamline the lifecycle of trading strategies through advanced signal detection and processing. It bridges the gap between research and production by providing a robust pipeline for signal generation, validation (Meta-Labeling), and execution.

## Core Concept: The Signal Pipeline

SignalFlow treats trading as a signal-processing problem based on three key entities:

* **🕵️ Signal Detector:** Scans raw market data (tick or OHLCV) in real-time to identify potential market events. Whether it's a simple SMA cross or a pattern detected by a Transformer model, the Detector captures the moment of potential price change.
* **⚖️ Signal Validator:** Implements **Meta-Labeling** techniques (inspired by Marcos Lopez de Prado). It uses classification models to assess the probability of a signal's success, filtering out false positives before capital is committed.
* **♟️ Trading Strategy:** Converts high-confidence signals into actionable trade positions, handling entry/exit logic and risk management.

## Key Features

* **Deployment Ready:** Seamless transition from backtesting to production; the logic you write in research is the logic that runs live.
* **Kedro Integration:** Includes a fully configured `signalflow-kedro` project template for reproducible R&D, data pipelines, and experiment tracking.
* **High Frequency Capable:** Optimized architecture handles short timeframes (1m) and tick data efficiently.
* **Extensible & Modular:** Easily plug in custom feature generators, deep learning models (Torch/Lightning), or validation logic.

## Tech Stack

Built on top of the modern Python data science stack:
`polars`, `pandas`, `pytorch`, `lightning`, `scikit-learn`, `plotly`, `pandas-ta`.
