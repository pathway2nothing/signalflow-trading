# SignalFlow Development Roadmap

> **Last Updated:** 2024-02-13
> **Current Version:** v0.4.3

## Vision

Transform SignalFlow from a powerful but verbose trading framework into a developer-friendly platform that competes with Freqtrade's ease of use while maintaining superior ML/signal validation capabilities.

## Competitive Analysis Summary

| Feature | SignalFlow | VectorBT | Backtrader | Freqtrade | NautilusTrader |
|---------|------------|----------|------------|-----------|----------------|
| **Ease of Use** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **ML Integration** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Performance** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Documentation** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **CLI/Config** | ❌ | ❌ | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**SignalFlow's Unique Strengths:**
- Meta-labeling pipeline (Lopez de Prado methodology)
- Triple-barrier labeling with Numba optimization
- Protocol-based ML model integration
- Polars-first performance

**Key Gaps to Address:**
- API verbosity
- Missing CLI/config file support
- Limited documentation/tutorials
- No notifications integration

---

## Q1 2024: Developer Experience

### 1. API Ergonomics (v0.5.0) ✨ PRIORITY
**Status:** Planning
**Plan:** [api-ergonomics-improvement.md](./api-ergonomics-improvement.md)
**Time:** 10-12 days

Goals:
- [ ] `sf.backtest()` one-liner shortcut
- [ ] `sf.load()` data loading shortcut
- [ ] `sf.Backtest()` fluent builder
- [ ] `BacktestResult` with `.summary()` and `.plot()`
- [ ] Sensible defaults for all parameters

**Impact:** Reduce minimal backtest from 43 lines to 5 lines

### 2. CLI Interface (v0.5.1)
**Status:** Not Started
**Time:** 1-2 weeks

Goals:
- [ ] `sf backtest --config strategy.yaml`
- [ ] `sf download --exchange binance --pairs BTCUSDT`
- [ ] `sf report results.parquet --format html`
- [ ] `sf paper-trade --config strategy.yaml`

**Dependencies:** Phase 1 (API Ergonomics)

### 3. YAML Configuration (v0.5.1)
**Status:** Not Started
**Time:** 3-5 days

Goals:
- [ ] Strategy definition in YAML
- [ ] Config validation with helpful errors
- [ ] Config inheritance/composition
- [ ] Environment variable support

Example:
```yaml
name: sma_crossover
data:
  exchange: binance
  pairs: [BTCUSDT, ETHUSDT]
  timeframe: 1h
detector:
  type: SmaCrossDetector
  fast_period: 20
  slow_period: 50
exit:
  tp: 0.03
  sl: 0.015
```

---

## Q2 2024: Ecosystem & Integrations

### 4. Notifications (v0.6.0)
**Status:** Not Started
**Time:** 1 week

Goals:
- [ ] Telegram notifications
- [ ] Discord webhook support
- [ ] Slack integration
- [ ] Email alerts (optional)

```python
from signalflow.notify import TelegramNotifier
notifier = TelegramNotifier(token="...", chat_id="...")
runner = BacktestRunner(alerts=[MaxDrawdownAlert(notifier=notifier)])
```

### 5. Streamlit Dashboard (v0.6.1)
**Status:** Not Started
**Time:** 1-2 weeks

Goals:
- [ ] Backtest results visualization
- [ ] Equity curve & drawdown charts
- [ ] Trade analysis dashboard
- [ ] Parameter sensitivity heatmaps
- [ ] Multi-run comparison

```bash
sf dashboard --results backtest.parquet
# Opens http://localhost:8501
```

### 6. Docker Support (v0.6.2)
**Status:** Not Started
**Time:** 3-5 days

Goals:
- [ ] Official Dockerfile
- [ ] Docker Compose for full stack
- [ ] Pre-built images on Docker Hub
- [ ] Cloud deployment guides (AWS, GCP)

---

## Q3 2024: Documentation & Community

### 7. Tutorial Notebooks
**Status:** 1/6 complete
**Time:** 2-3 weeks

Goals:
- [x] `funding_rate_detector.ipynb`
- [ ] `01_quickstart.ipynb` - First backtest in 5 minutes
- [ ] `02_custom_detector.ipynb` - Create your own detector
- [ ] `03_ml_validation.ipynb` - Meta-labeling with LightGBM
- [ ] `04_multi_asset.ipynb` - Portfolio of 10+ pairs
- [ ] `05_live_paper.ipynb` - Paper trading on Binance

### 8. Video Tutorials
**Status:** Not Started

Goals:
- [ ] YouTube: "SignalFlow in 10 minutes"
- [ ] YouTube: "ML-powered trading signals"
- [ ] YouTube: "From backtest to live trading"

### 9. Benchmarks & Comparisons
**Status:** Not Started
**Time:** 1 week

Goals:
- [ ] Performance benchmarks vs VectorBT
- [ ] Feature comparison table
- [ ] Migration guides from Backtrader/VectorBT

---

## Q4 2024: Performance & Scale

### 10. Rust Core (v1.0.0) 🦀
**Status:** Research
**Time:** 2-3 months

Priority components for Rust rewrite:
1. Triple Barrier Labeler (`_find_first_hit`)
2. Trend Scanning Labeler (`_trend_scan_numba`)
3. Backtest Runner core loop
4. Mutual Information calculator

Expected speedup: 5-10x for labeling, 3-4x for backtesting

### 11. Distributed Backtesting
**Status:** Not Started

Goals:
- [ ] Ray/Dask integration for parallel backtests
- [ ] Parameter optimization at scale
- [ ] Cloud-native execution

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| v0.4.3 | 2024-02 | Multi-exchange architecture, RawDataLazy |
| v0.4.2 | 2024-02 | Classification, profile, distribution metrics |
| v0.4.0 | 2024-01 | Advanced signals, parallel backtesting |
| v0.3.6 | 2024-01 | Virtual trading foundation |

---

## Priority Matrix

```
                     IMPACT
                       ▲
                       │
   HIGH IMPACT    ┌────┼────────────────────┐
                  │    │                    │
                  │ API Ergonomics   CLI    │
                  │ Notifications    YAML   │
                  │    │                    │
   ───────────────┼────┼────────────────────┤
                  │    │                    │
   LOW IMPACT     │ Benchmarks    Dashboard │
                  │ Docker        Rust Core │
                  │    │                    │
                  └────┴────────────────────┘
                       │
          LOW ◄────────┴────────► HIGH
                    EFFORT
```

**Recommended order:**
1. API Ergonomics ← Start here
2. CLI + YAML
3. Notifications
4. Tutorials
5. Dashboard
6. Rust Core (long-term)

---

## Contributing

See individual plan documents for implementation details:
- [API Ergonomics Plan](./api-ergonomics-improvement.md)
- CLI Plan (TBD)
- Notifications Plan (TBD)

## Ecosystem

Remember that SignalFlow has companion packages:
- **signalflow-ta** - Technical indicators and detectors (199+ indicators)
- **signalflow-nn** - Neural network validators (LSTM, GRU, Attention)

Core `sf` package contains only examples; real detectors/indicators are in `signalflow-ta`.
