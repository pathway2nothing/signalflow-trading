---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# SignalFlow

**Modern signal processing and algorithmic trading framework for Python**

Build, test, and deploy trading strategies with confidence.

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/pathway2nothing/signal-flow){ .md-button }

</div>

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### :material-lightning-bolt: High Performance

Optimized for speed with GPU acceleration support. Process millions of data points in seconds with cuML and PyTorch backends.

</div>

<div class="feature-card" markdown>

### :material-chart-line: Advanced Signals

Comprehensive library of technical indicators and custom signal generation. From simple moving averages to complex ML-based predictions.

</div>

<div class="feature-card" markdown>

### :material-flask: Backtesting Engine

Robust backtesting with realistic market simulation. Account for slippage, fees, and market impact in your strategy evaluation.

</div>

<div class="feature-card" markdown>

### :material-robot: ML Integration

Seamless integration with scikit-learn, PyTorch, and Optuna. Build adaptive strategies that learn from market data.

</div>

<div class="feature-card" markdown>

### :material-pipe: Pipeline Architecture

Built on Kedro patterns for reproducible, maintainable workflows. Track experiments with MLflow integration.

</div>

<div class="feature-card" markdown>

### :material-clock-fast: Real-time Ready

Deploy strategies to live markets with minimal code changes. Support for multiple exchanges and data providers.

</div>

</div>

## Quick Example

```python
from signalflow import Strategy, Signal
from signalflow.indicators import RSI, MACD

class MomentumStrategy(Strategy):
    def setup(self):
        self.rsi = RSI(period=14)
        self.macd = MACD(fast=12, slow=26, signal=9)
    
    def generate_signals(self, data):
        rsi_signal = Signal.when(self.rsi(data) < 30, action="buy")
        macd_signal = Signal.when(self.macd.crossover(data), action="buy")
        
        return rsi_signal & macd_signal

# Backtest the strategy
from signalflow import Backtest

bt = Backtest(
    strategy=MomentumStrategy(),
    data="BTC/USDT",
    start="2024-01-01",
    end="2024-12-01"
)

results = bt.run()
print(results.summary())
```

## Installation

=== "pip"

    ```bash
    pip install signal-flow
    ```

=== "pip (with GPU)"

    ```bash
    pip install signal-flow[gpu]
    ```

=== "poetry"

    ```bash
    poetry add signal-flow
    ```

## What's Next?

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### :material-book-open-variant: User Guide

Learn the core concepts and build your first strategy.

[Read the Guide →](guide/overview.md)

</div>

<div class="feature-card" markdown>

### :material-api: API Reference

Detailed documentation for all modules and classes.

[Browse API →](api/index.md)

</div>

<div class="feature-card" markdown>

### :material-code-tags: Examples

Real-world examples and use cases to get you started.

[View Examples →](examples/index.md)

</div>

</div>
