# Quick Start

Get up and running with SignalFlow in under 5 minutes.

## Your First Strategy

Create a simple moving average crossover strategy:

```python
from signalflow import Strategy, Backtest
from signalflow.indicators import SMA

class SMACrossover(Strategy):
    """Simple Moving Average Crossover Strategy."""
    
    def setup(self):
        self.fast_sma = SMA(period=10)
        self.slow_sma = SMA(period=30)
    
    def generate_signals(self, data):
        fast = self.fast_sma(data)
        slow = self.slow_sma(data)
        
        # Buy when fast crosses above slow
        buy_signal = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        
        # Sell when fast crosses below slow
        sell_signal = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        
        return self.signals(buy=buy_signal, sell=sell_signal)
```

## Running a Backtest

Test your strategy on historical data:

```python
# Initialize backtest
bt = Backtest(
    strategy=SMACrossover(),
    data="BTC/USDT",
    start="2024-01-01",
    end="2024-06-01",
    initial_capital=10000
)

# Run backtest
results = bt.run()

# View performance metrics
print(results.summary())
```

Output:
```
═══════════════════════════════════════════════════
                  Backtest Results                 
═══════════════════════════════════════════════════
Total Return:        +23.45%
Sharpe Ratio:        1.82
Max Drawdown:        -8.32%
Win Rate:            62.5%
Total Trades:        24
═══════════════════════════════════════════════════
```

## Visualize Results

```python
# Plot equity curve and trades
results.plot()

# Save interactive chart
results.plot(save="backtest_results.html")
```

## Adding Risk Management

```python
from signalflow.risk import StopLoss, TakeProfit, PositionSizer

class RiskManagedStrategy(SMACrossover):
    def setup(self):
        super().setup()
        
        # Add risk management
        self.stop_loss = StopLoss(percent=2.0)
        self.take_profit = TakeProfit(percent=5.0)
        self.position_sizer = PositionSizer(
            method="kelly",
            max_position=0.2  # Max 20% per trade
        )
```

## What's Next?

- [Configuration](configuration.md) - Customize SignalFlow settings
- [User Guide](../guide/overview.md) - Deep dive into all features
- [Examples](../examples/index.md) - Real-world strategy examples