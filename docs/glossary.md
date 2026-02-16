---
title: Glossary
description: SignalFlow terminology explained in simple terms
---

# Glossary

A quick reference guide to SignalFlow terminology. Each term is explained in plain language with practical examples.

---

## Core Concepts

### RawData
**What it is**: A container for OHLCV (Open, High, Low, Close, Volume) market data.

**Think of it as**: A spreadsheet with price history for one or more trading pairs.

```python
# RawData holds your market data
raw_data = sf.load("binance", pairs=["BTC/USDT"], timeframe="1h")
# raw_data.df is a Polars DataFrame with columns: timestamp, open, high, low, close, volume, pair
```

---

### Signals
**What it is**: A DataFrame marking timestamps where trading opportunities were detected.

**Think of it as**: A list of "buy here" or "sell here" recommendations from your detector.

| Column | Description |
|--------|-------------|
| `timestamp` | When the signal occurred |
| `pair` | Which trading pair |
| `signal` | Direction: 1 (long), -1 (short), 0 (neutral) |
| `probability` | Optional confidence score (0.0 to 1.0) |

```python
# Signals tell you WHEN to potentially trade
signals = detector.run(raw_data.view())
# signals.df might show: timestamp=2024-01-15 14:00, pair=BTC/USDT, signal=1
```

---

### Detector
**What it is**: A component that analyzes market data and outputs trading signals.

**Think of it as**: A "signal finder" that scans price data for patterns.

**Examples**:
- `sma_cross` — Signal when fast SMA crosses slow SMA
- `rsi_threshold` — Signal when RSI goes below 30 (oversold) or above 70 (overbought)
- `macd_cross` — Signal when MACD line crosses signal line

```python
# Detector finds trading opportunities
result = sf.Backtest("my_strategy").detector("sma_cross", fast=20, slow=50).run()
```

---

### Labeler (Target Generator)
**What it is**: A component that labels signals as "successful" or "unsuccessful" based on future price movement.

**Think of it as**: A "hindsight checker" that looks at what happened AFTER each signal.

**Common labelers**:

| Labeler | How it works |
|---------|--------------|
| `triple_barrier` | Success if price hits take-profit before stop-loss or time limit |
| `fixed_horizon` | Success if price is higher after N bars (for long signals) |

```python
# Labeler tells you if signals would have been profitable
result = sf.Backtest("my_strategy").labeler("triple_barrier", tp=0.02, sl=0.01).run()
```

---

### Validator
**What it is**: A machine learning model that predicts whether a signal will be successful.

**Think of it as**: An "AI filter" that rejects bad signals before you trade them.

**How it works**:
1. Train on historical signals + their labels (success/failure)
2. Predict probability of success for new signals
3. Only trade signals with high predicted probability

```python
# Validator filters signals using ML
result = (
    sf.Backtest("ml_strategy")
    .detector("sma_cross")
    .labeler("triple_barrier", tp=0.02, sl=0.01)
    .validator("lightgbm")  # ML model filters signals
    .run()
)
```

---

### Feature / FeaturePipeline
**What it is**: Calculated values derived from raw price data that help ML models make decisions.

**Think of it as**: "Clues" that the ML model uses to predict signal success.

**Examples**:
- RSI value at signal time
- Volatility over last 20 bars
- Distance from 200-day moving average

```python
# Features provide context for ML validation
result = (
    sf.Backtest("ml_strategy")
    .features("momentum")  # Adds RSI, MACD, etc. as features
    .validator("lightgbm")
    .run()
)
```

---

### Entry Rule
**What it is**: Logic that determines HOW to enter a trade when a signal is received.

**Think of it as**: "How much to buy and at what price."

**Common entry rules**:
- `market` — Enter immediately at market price
- `limit` — Enter only at a specific price
- `signal` — Use signal's suggested size

---

### Exit Rule
**What it is**: Logic that determines WHEN and HOW to exit a trade.

**Think of it as**: "When to sell and take profit or cut losses."

**Common exit rules**:

| Exit Rule | Description |
|-----------|-------------|
| `tp_sl` | Exit at take-profit or stop-loss price |
| `trailing_stop` | Stop-loss that follows price up |
| `time_exit` | Exit after N bars regardless of profit |
| `volatility_exit` | Exit based on ATR multiplier |

```python
result = sf.Backtest("my_strategy").exit(tp=0.03, sl=0.015).run()
```

---

### Position
**What it is**: An active trade from entry to exit.

**Think of it as**: "I bought X amount at price Y, and I'm holding it."

**SignalFlow model**: Each position has exactly ONE entry and ONE exit. For multiple buys/sells, create multiple positions.

---

### Strategy
**What it is**: The complete trading system combining detector, validator, entry rules, and exit rules.

**Think of it as**: Your complete trading plan: "Find signals → Filter them → Enter trades → Exit trades."

---

## Pipeline Stages

```
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌───────────┐    ┌──────────┐
│  Data   │───▶│ Features │───▶│ Detector│───▶│ Validator │───▶│ Strategy │
└─────────┘    └──────────┘    └─────────┘    └───────────┘    └──────────┘
   OHLCV         RSI, ATR       Signals      Filtered sigs     Entry/Exit
```

---

## Backtest Metrics

### Sharpe Ratio
**What it is**: Risk-adjusted return. Higher is better.

**Rule of thumb**:
- < 0: Losing money
- 0-1: Mediocre
- 1-2: Good
- > 2: Excellent

---

### Max Drawdown
**What it is**: Largest peak-to-trough decline during the backtest.

**Think of it as**: "The worst losing streak."

**Example**: -15% drawdown means at some point you lost 15% from your peak equity.

---

### Win Rate
**What it is**: Percentage of profitable trades.

**Warning**: High win rate doesn't mean profitable strategy! A 90% win rate with tiny wins and huge losses = net loss.

---

### Profit Factor
**What it is**: Gross profit divided by gross loss.

**Rule of thumb**:
- < 1: Losing money
- 1-1.5: Break-even to weak
- 1.5-2: Good
- > 2: Excellent

---

## Meta-Labeling

**What it is**: A technique from Marcos López de Prado where ML predicts signal success rather than direction.

**Traditional ML**: Model predicts "buy" or "sell"

**Meta-labeling**: Detector predicts direction → ML predicts "will this signal succeed?"

**Why it's better**:
1. Detector handles the hard part (finding patterns)
2. ML handles filtering (rejecting bad signals)
3. Easier to train, more robust results

---

## Validation Methods

### Temporal Cross-Validation
**What it is**: Train/test split that respects time order.

**Why needed**: You can't train on future data (look-ahead bias).

```
[=====TRAIN=====][TEST]   Fold 1
[========TRAIN========][TEST]   Fold 2
[===========TRAIN===========][TEST]   Fold 3
```

---

### Walk-Forward Validation
**What it is**: Rolling window training that simulates real trading.

**How it works**:
1. Train on window 1 → Test on next period
2. Move window forward → Retrain → Test
3. Combine all test results

```
Window 1: [====TRAIN====][TEST]
Window 2:      [====TRAIN====][TEST]
Window 3:           [====TRAIN====][TEST]
```

---

## Common Abbreviations

| Abbreviation | Meaning |
|--------------|---------|
| OHLCV | Open, High, Low, Close, Volume |
| SMA | Simple Moving Average |
| EMA | Exponential Moving Average |
| RSI | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |
| ATR | Average True Range |
| TP | Take Profit |
| SL | Stop Loss |
| OOS | Out-of-Sample (test data) |
| IS | In-Sample (training data) |
| ML | Machine Learning |
| CV | Cross-Validation |
| DD | Drawdown |

---

## See Also

- [Quick Start](quickstart.md) — Build your first strategy
- [API Reference](api/index.md) — Detailed class documentation
- [Advanced Strategies](guide/advanced-strategies.md) — Position sizing, filters, aggregation
