# SignalFlow Roadmap

**Focus**: Strategy-level improvements, ML/RL-ready architecture, virtual trading validation
**Current version**: v0.3.6 | **Target**: v1.0.0

---

## Vision

```
Backtest → Virtual Trading → Live Trading
    ↓           ↓               ↓
  Signals   Validation      Production
    ↓           ↓               ↓
   ML/RL    Real-time        Capital
  Models    Simulation       at Risk
```

**Principle**: No real money until virtual trading proves consistent profitability.

---

## Phase 1: Advanced Exit Strategies

Improve exit logic beyond simple TP/SL. Enable dynamic exits based on market conditions.

### 1.1 Trailing Stop Exit

**File**: `strategy/component/exit/trailing_stop.py`

- [ ] `TrailingStopExit(ExitRule)`:
  - `trail_pct`: Trailing distance as percentage (e.g., 0.02 = 2%)
  - `activation_pct`: Optional - start trailing only after X% profit
  - Track `highest_price` (LONG) / `lowest_price` (SHORT) in `position.meta`
  - Exit when price retraces by `trail_pct` from peak
- [ ] Support both percentage and ATR-based trailing distance
- [ ] Tests with various market conditions (trending, choppy)

### 1.2 Volatility-Based Exit

**File**: `strategy/component/exit/volatility_exit.py`

- [ ] `VolatilityExit(ExitRule)`:
  - Dynamic TP/SL based on recent volatility (ATR, std dev)
  - `tp_atr_mult`: TP = entry_price + N × ATR
  - `sl_atr_mult`: SL = entry_price - N × ATR
  - Recalculate levels on each bar or use entry-time levels
- [ ] Integration with feature pipeline for ATR computation

### 1.3 Composite Exit Manager

**File**: `strategy/component/exit/composite.py`

- [ ] `CompositeExit(ExitRule)`:
  - Combine multiple exit rules with priority
  - First triggered exit wins
  - Example: TP/SL + Trailing + Time-based
- [ ] `ExitPriority` enum: FIRST_TRIGGERED, HIGHEST_PRIORITY

**Design constraint**: Position = atomic unit with fixed size. No partial exits. If scaling is needed, open multiple positions.

---

## Phase 2: Advanced Entry & Position Sizing

Improve capital allocation and entry timing.

### 2.1 Position Sizing Strategies

**File**: `strategy/component/sizing/`

- [ ] `PositionSizer` base class:
  ```python
  def compute_size(signal: Signal, state: StrategyState, prices: dict) -> float
  ```

- [ ] Implementations:
  - `FixedFractionSizer`: Fixed % of equity per trade
  - `KellyCriterionSizer`: Kelly formula based on win rate and payoff ratio
  - `VolatilityTargetSizer`: Target specific portfolio volatility
  - `RiskParitySizer`: Equal risk contribution across positions
  - `SignalStrengthSizer`: Size proportional to signal probability

- [ ] Integration with `SignalEntryRule` - inject `PositionSizer`

### 2.2 Entry Filters

**File**: `strategy/component/entry/filters.py`

- [ ] `EntryFilter` protocol:
  ```python
  def allow_entry(signal: Signal, state: StrategyState, prices: dict) -> bool
  ```

- [ ] Implementations:
  - `RegimeFilter`: Only enter in favorable market regime (trend/mean-reversion)
  - `VolatilityFilter`: Skip entries in extreme volatility
  - `DrawdownFilter`: Pause trading after X% drawdown
  - `CorrelationFilter`: Avoid concentrated positions in correlated assets
  - `TimeOfDayFilter`: Restrict trading hours

### 2.3 Signal Aggregation

**File**: `strategy/component/entry/aggregation.py`

- [ ] `SignalAggregator`:
  - Combine signals from multiple detectors
  - Voting: majority, weighted, unanimous
  - Meta-labeling integration: detector signal × validator probability

---

## Phase 3: External Model Integration

**Note**: ML/RL models will be implemented in a separate repository. Here we only define integration interfaces.

### 3.1 Strategy Model Protocol

**File**: `strategy/model/protocol.py`

**Design principle**: Strategy-level models do NOT see raw price features. They only see:
- Signals from detector/validator (already price-derived)
- Signal metrics (hit rate, accuracy, timing)
- Strategy metrics (equity, drawdown, win rate, position stats)

- [ ] `StrategyModel` protocol:
  ```python
  class StrategyModel(Protocol):
      def decide(
          self,
          signals: Signals,
          signal_metrics: dict,
          strategy_metrics: dict,
          state: StrategyState,
      ) -> StrategyDecision: ...

      def load(self, path: Path) -> None: ...
  ```

- [ ] `StrategyDecision` dataclass:
  - `action`: ENTER, SKIP, REDUCE_SIZE, WAIT
  - `size_multiplier`: 0.0 - 1.0
  - `confidence`: model confidence

### 3.2 Model-Aware Entry Rule

**File**: `strategy/component/entry/model_entry.py`

- [ ] `ModelEntryRule(EntryRule)`:
  - Wraps `StrategyModel` and calls `decide()` before entry
  - Respects model's `action` and `size_multiplier`
  - Falls back to base entry if no model provided

### 3.3 Metrics Export for Training

**File**: `strategy/export/`

- [ ] `BacktestExporter`:
  - Export backtest results for external model training
  - Format: Parquet with signals + metrics per bar
  - No raw prices - only derived metrics
  - `export(results, path)` → ready for external RL repo

---

## Phase 4: Virtual Trading (Paper Trading)

Real-time simulation with virtual execution. **Must prove profitability before live trading.**

### 4.1 RealtimeRunner Implementation

**File**: `strategy/runner/realtime_runner.py` (currently stub)

- [ ] Async main loop:
  ```python
  async def run(self):
      while not self._shutdown:
          await self._sync_new_bars()
          for bar in new_bars:
              self._process_bar(bar)
          await self._persist_state()
          await asyncio.sleep(poll_interval)
  ```

- [ ] Data ingestion:
  - Poll DuckDB for new candles (from `BinanceSpotLoader.sync()`)
  - Maintain rolling window for feature warmup
  - Detect and handle gaps in data

- [ ] Signal pipeline:
  - `FeaturePipeline.compute()` → `Detector.detect()` → `Validator.validate()`
  - Full backtest logic adapted for real-time

- [ ] Idempotency:
  - `state.last_ts` for deduplication
  - Resume from last checkpoint after restart

- [ ] Graceful shutdown:
  - SIGINT/SIGTERM handler
  - Persist state before exit

### 4.2 Data Sync Integration

**File**: `data/source/binance.py`, `strategy/runner/realtime_runner.py`

- [ ] Run data sync as background task alongside runner
- [ ] Wait for sync to write new candle before processing
- [ ] Handle sync failures gracefully

### 4.3 Virtual Execution Mode

**File**: `strategy/broker/virtual_broker.py`

- [ ] `VirtualRealtimeBroker`:
  - Uses `VirtualSpotExecutor` for instant fills
  - Simulates slippage (optional)
  - Logs all orders/fills for analysis

- [ ] Order logging:
  - Save all orders to DB for post-analysis
  - Compare virtual vs. actual market prices

### 4.4 Monitoring & Alerts

**File**: `strategy/monitoring/`

- [ ] Per-bar metrics:
  - Equity curve (current vs. expected)
  - Open positions
  - Signal hit rate

- [ ] Alert thresholds:
  - Max drawdown exceeded
  - No signals for N bars
  - Position stuck (not exiting)

- [ ] Structured logging (JSON) for parsing
- [ ] Optional: Telegram/Slack notifications

### 4.5 Paper Trading Validation

**Criteria before going live:**

- [ ] Minimum 2 weeks of virtual trading
- [ ] Performance matches backtest expectations (±10%)
- [ ] No bugs or unexpected behaviors
- [ ] Successful restart/recovery test
- [ ] Drawdown within acceptable limits

---

## Phase 5: Risk Management Module

**File**: `strategy/risk/`

### 5.1 Pre-Trade Checks

- [ ] `PreTradeValidator`:
  - Max position size (% of equity)
  - Max concentration per asset
  - Daily loss limit (circuit breaker)
  - Binance LOT_SIZE/MIN_NOTIONAL filters

### 5.2 Runtime Guards

- [ ] `RuntimeRiskManager`:
  - Max drawdown circuit breaker (pause trading)
  - Cooldown between trades
  - Equity monitoring with alerts
  - Position age limits

### 5.3 Portfolio-Level Risk

- [ ] `PortfolioRiskManager`:
  - Max total exposure
  - Correlation-based limits
  - Sector/market cap concentration

---

## Phase 6: Live Trading

**Only after Phase 4 validation is complete.**

### 6.1 BinanceSpotExecutor

**File**: `strategy/broker/executor/binance_spot.py` (currently stub)

- [ ] Trading API methods:
  - `post_order()` - MARKET/LIMIT orders
  - `get_order()` - Order status
  - `cancel_order()` - Cancel open orders
  - HMAC-SHA256 signing

- [ ] Order handling:
  - Binance response → `OrderFill`
  - Partial fills support
  - Rejection handling

- [ ] User Data Stream (WebSocket):
  - Real-time order updates
  - Auto-renew listenKey

### 6.2 RealtimeSpotBroker

**File**: `strategy/broker/realtime_spot.py` (currently stub)

- [ ] Lifecycle: `async start()` / `async stop()`
- [ ] Fill processing from WebSocket
- [ ] Order reconciliation on startup
- [ ] Retry logic with exponential backoff

### 6.3 Production Safeguards

- [ ] Testnet validation first (`testnet.binance.vision`)
- [ ] Small position sizes initially
- [ ] Kill switch (manual shutdown)
- [ ] Daily reports and alerts

---

## Phase 7: WebSocket Streaming

Optional optimization for lower latency.

**File**: `data/source/binance_ws.py`

- [ ] WebSocket manager with auto-reconnect
- [ ] `@kline_{interval}` stream for real-time candles
- [ ] `@bookTicker` for best bid/ask
- [ ] Fallback to polling on disconnect
- [ ] Integration with `RealtimeRunner`

---

## Priority Order

```
Phase 1 (exits)        --- Trailing stop, volatility exits
    │
Phase 2 (entries)      --- Position sizing, entry filters
    │
Phase 3 (integration)  --- Model protocol, metrics export
    │
Phase 4 (virtual)      --- Paper trading, validation ← GATE
    │
Phase 5 (risk)         --- Risk management module
    │
Phase 6 (live)         --- Real execution (only after Phase 4 ✓)
    │
Phase 7 (websocket)    --- Optional latency optimization
```

---

## Completed Work

### v0.3.6
- [x] Parallel backtest runners (isolated, unlimited modes)
- [x] `create_backtest_runner(mode="isolated"|"unlimited")`
- [x] Component registry with autodiscovery

### v0.3.5
- [x] SQLite + PostgreSQL storage backends
- [x] Store factory pattern

### v0.3.4
- [x] Extensible RawDataType with registry
- [x] Pre-commit hooks (ruff, mypy)

---

## Key Design Decisions

1. **Virtual trading is mandatory** before live trading
2. **Position = atomic unit** - fixed size, single exit point, no partial exits
3. **Strategy models see signals, not prices** - detector/validator handles price prediction
4. **Two-tier model architecture**:
   - Detector/Validator → price prediction (sees raw OHLCV features)
   - Strategy Model → execution decisions (sees signals + metrics only)
5. **ML/RL models in separate repo** - signalflow provides Protocol interface only
6. **Position sizing is separate from entry rules** for flexibility
7. **Exit rules are composable** (combine trailing + TP/SL + time)
8. **Risk management is a separate module** enforced at broker level

---

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| 1-2 | Backtest Sharpe improvement | +0.3 vs. simple TP/SL |
| 3 | Protocol compatibility | Works with external models |
| 4 | Virtual trading alignment | ±10% vs. backtest |
| 4 | Virtual trading period | 14+ days |
| 5 | Max drawdown | <15% |
| 6 | Live trading PnL | Positive after 30 days |
