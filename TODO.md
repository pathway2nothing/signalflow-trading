# SignalFlow Roadmap

**Focus**: Production-ready live trading on Binance Spot
**Current version**: v0.3.4 | **Target**: v1.0.0
**Breaking changes**: allowed (pre-v1.0, documented in changelog)

---

## Phase 0: Housekeeping

Close technical debt before the main work.

### 0.1 Extensible RawDataType ~~(StrEnum + registry)~~

> **Done** in `e368956` — `register_raw_data_type()`, `get_raw_data_columns()`, `list_raw_data_types()`

- [x] Keep built-in types as `StrEnum`: `SPOT`, `FUTURES`, `PERPETUAL`
- [x] Add column registry in `SignalFlowRegistry` with built-in defaults
- [x] `RawDataType.columns` delegates to registry
- [x] All components accept `RawDataType | str` for custom types
- [x] 16 tests for registration, lookup, override, validation
- [x] Documentation: `docs/guide/custom-data-types.md`

### 0.2 Component Autodiscovery

**Files**: `core/registry.py`

- [x] Implement `SignalFlowRegistry.autodiscover()`:
  - Scan `signalflow.*` packages via `importlib` + `pkgutil`
  - Find all classes decorated with `@sf_component`
  - Auto-register on first registry access
- [x] Support external packages via `entry_points` group `signalflow.components`
- [x] Lazy loading — import modules only on first `registry.get()` / `registry.list()`
- [x] Tests for autodiscovery

### 0.3 Pre-commit Hooks

**Files**: `.pre-commit-config.yaml` (new), `pyproject.toml` (ruff config)

- [x] Configure `.pre-commit-config.yaml`:
  - `ruff` — linter + formatter
  - `ruff format --check`
  - `mypy --strict` on core modules
  - `pytest tests/ -x --timeout=30` (quick smoke test)
- [x] Add ruff config to `pyproject.toml`
- [x] Add `py.typed` marker (PEP 561)

---

## Phase 1: Pluggable Data Store

Add alternative storage backends so users can choose what fits their setup: local DuckDB (already done), local SQLite, or external PostgreSQL.

### 1.1 SQLite Raw + Strategy Store

**Files**: `data/raw_store/sqlite_stores.py` (new), `data/strategy_store/sqlite.py` (new)

- [ ] `SqliteSpotStore(RawDataStore)` — same interface as `DuckDbSpotStore`:
  - `load()`, `load_many()`, `load_many_pandas()`, `insert_klines()`, `close()`
  - Schema: `ohlcv` table with `(pair, timestamp)` primary key
  - Use `sqlite3` from stdlib — zero extra dependencies
- [ ] `SqliteStrategyStore(StrategyStore)` — same interface as `DuckDbStrategyStore`:
  - `init()`, `load_state()`, `save_state()`, `upsert_positions()`, `append_trade()`, `append_metrics()`
  - Reuse `schema.py` SQL (SQLite-compatible subset)
- [ ] Tests: mirror existing DuckDB test suite, parametrize over `[duckdb, sqlite]`

### 1.2 PostgreSQL Raw + Strategy Store

**Files**: `data/raw_store/pg_stores.py` (new), `data/strategy_store/pg.py` (new)

- [ ] `PgSpotStore(RawDataStore)` — async via `asyncpg` or sync via `psycopg`:
  - Same public interface as `DuckDbSpotStore`
  - Connection string from config / env var `SIGNALFLOW_PG_DSN`
  - Schema auto-creation on `init()` (idempotent `CREATE TABLE IF NOT EXISTS`)
- [ ] `PgStrategyStore(StrategyStore)` — same interface as `DuckDbStrategyStore`:
  - Reuse JSON-based serialization pattern
  - Connection pooling for concurrent access
- [ ] Add `psycopg[binary]` or `asyncpg` as optional dependency (`pip install signalflow[postgres]`)
- [ ] Tests: use `pytest-postgresql` fixture or testcontainers; CI runs with `services: postgres`

### 1.3 Store Factory & Configuration

**Files**: `data/store_factory.py` (new), `core/config.py`

- [ ] `StoreFactory.create_raw_store(backend=..., **kwargs) -> RawDataStore`
  - `backend="duckdb"` (default), `"sqlite"`, `"postgres"`
  - Raise clear error if optional deps missing (`asyncpg` / `psycopg`)
- [ ] `StoreFactory.create_strategy_store(backend=..., **kwargs) -> StrategyStore`
- [ ] Register all backends in `SignalFlowRegistry` for autodiscovery
- [ ] Docs: `docs/guide/storage-backends.md` — comparison table (features, deps, use-case)

---

## Phase 2: Paper Trading (Real-Time Simulation)

Real-time data via existing `BinanceSpotLoader.sync()` (REST polling), orders via `VirtualSpotExecutor`. Full pipeline with zero risk.

### 2.1 RealtimeRunner

**Files**: `strategy/runner/realtime_runner.py` (empty -> full implementation)

- [ ] Async main loop driven by `BinanceSpotLoader.sync()`:
  - Poll DuckDB for new bars since `state.last_ts`
  - Process each new bar through strategy pipeline
  - Persist state after each cycle
- [ ] `process_bar()` — adapted from `BacktestRunner._process_bar()`:
  - Mark positions to market
  - Check exit rules -> submit exit orders
  - Compute features -> detect signals -> check entry rules -> submit entry orders
  - Process fills (`VirtualSpotExecutor` — instant fills)
  - Compute metrics
- [ ] Signal pipeline integration:
  - `FeaturePipeline.compute()` -> `Detector.detect()` -> `Validator.validate()` (optional)
  - Maintain rolling data window in memory for feature warmup
- [ ] Graceful shutdown:
  - SIGINT/SIGTERM handler
  - Save state to DuckDB
  - Log final portfolio state
- [ ] Idempotency:
  - `state.last_ts` for deduplication — skip already-processed bars
  - Safe restart from last checkpoint

### 2.2 Data Sync Integration

**Files**: `data/source/binance.py`, `data/raw_store/duckdb_spot.py`

- [ ] Run `BinanceSpotLoader.sync()` as background asyncio task alongside runner
- [ ] Method to fetch last N bars from DuckDB (for feature warmup)
- [ ] Coordination: runner waits until sync writes a new candle

### 2.3 Metrics & Monitoring

- [ ] Per-bar logging: equity, PnL, open positions, signals
- [ ] Periodic summary (every N bars): Sharpe, drawdown, win rate
- [ ] Structured loguru output for parsing

### 2.4 Tests

- [ ] Unit test: `RealtimeRunner.process_bar()` with mock data
- [ ] Integration test: full cycle data -> features -> signals -> orders -> fills -> state
- [ ] Restart/recovery test: save state -> create new runner -> verify continuation

---

## Phase 3: WebSocket Streaming

Replace REST polling with real-time WebSocket streams for lower latency.

**Files**: `data/source/binance_ws.py` (new)

- [ ] WebSocket manager with auto-reconnect:
  - `aiohttp` or `websockets` library
  - Heartbeat / ping-pong
  - Exponential backoff on reconnect
- [ ] Market data streams:
  - `@kline_{interval}` — real-time OHLCV candles
  - `@bookTicker` — best bid/ask (for future LIMIT orders)
- [ ] Callback-based architecture: `on_kline()`, `on_tick()`
- [ ] Integrate with `RealtimeRunner` — switch from polling to WS events
- [ ] Fallback: if WS disconnects -> automatic switch to polling

---

## Phase 4: BinanceSpotExecutor (Real Orders)

Implement order submission and tracking on Binance Spot.

**Files**: `strategy/broker/executor/binance_spot.py` (stub -> full), `data/source/binance.py`

- [ ] Add Trading API methods to `BinanceClient`:
  - `post_order()` — `POST /api/v3/order`
  - `get_order()` — `GET /api/v3/order`
  - `cancel_order()` — `DELETE /api/v3/order`
  - `get_open_orders()`, `get_account()`
- [ ] HMAC-SHA256 request signing (API key + secret)
- [ ] `BinanceSpotExecutor.execute()`:
  - `Order -> Binance API params` conversion
  - MARKET / LIMIT orders
  - Binance response -> `OrderFill`
  - Partial fills, rejections
- [ ] User Data Stream (WebSocket) for fills:
  - `POST /api/v3/userDataStream` -> listenKey
  - Auto-renew listenKey every 30 min
  - `executionReport` -> `OrderFill`
- [ ] Tests with mock server (aioresponses)

---

## Phase 5: RealtimeSpotBroker + Live Mode

Connect `BinanceSpotExecutor` to the runner for production live trading.

**Files**: `strategy/broker/realtime_spot.py` (stub -> full)

- [ ] Lifecycle management: `async start()` / `async stop()` / context manager
- [ ] Fill processing: WS `executionReport` -> `broker.add_pending_fill()`
- [ ] Order reconciliation on startup (compare open orders with state)
- [ ] Error handling: retry logic, circuit breaker, alerts
- [ ] Runner mode switch: `mode="paper"` (VirtualExecutor) / `mode="live"` (BinanceExecutor)

---

## Phase 6: Risk Management

**Files**: `strategy/risk/` (new module)

- [ ] **Pre-trade checks**: max position size, max exposure %, daily loss limit, Binance LOT_SIZE filter
- [ ] **Runtime guards**: cooldown between orders, max drawdown circuit breaker, equity monitoring
- [ ] **Position limits**: max concurrent positions, per-pair limits, concentration limits

---

## Phase 7: Testnet & E2E Testing

- [ ] Binance Testnet integration (`testnet.binance.vision`)
- [ ] E2E test: data -> features -> signals -> orders -> fills -> state
- [ ] Stress test: reconnect scenarios, partial fills, order rejections
- [ ] State recovery test: kill process -> restart -> verify consistency
- [ ] Paper trading minimum 48 hours before production

---

## Phase 8: CLI / Operational Interface

- [ ] CLI entry point (`signalflow run`, `signalflow status`)
- [ ] YAML/TOML strategy configuration
- [ ] Structured JSON logging for production
- [ ] Health check endpoint (optional HTTP/webhook)

---

## Priority Order

```
Phase 0 (housekeeping)     --- clean code, extensible RawDataType, autodiscovery
    |
Phase 1 (data stores)      --- SQLite + PostgreSQL backends, store factory
    |
Phase 2 (paper trading)    --- full pipeline with polling + VirtualExecutor
    |
Phase 3 (websocket)        --- real-time streaming replaces polling
    |
Phase 4 (executor)         --- real orders on Binance
    |
Phase 5 (live broker)      --- production live trading
    |
Phase 6 (risk)             --- loss protection
    |
Phase 7 (testing)          --- testnet verification
    |
Phase 8 (CLI/ops)          --- operational convenience
```

## Key Files

| File | Action | Phase |
|------|--------|-------|
| `core/registry.py` | Autodiscovery via importlib + entry_points | 0 |
| `.pre-commit-config.yaml` | New — ruff, mypy, pytest | 0 |
| `data/raw_store/sqlite_stores.py` | New — SQLite raw data store | 1 |
| `data/strategy_store/sqlite.py` | New — SQLite strategy store | 1 |
| `data/raw_store/pg_stores.py` | New — PostgreSQL raw data store | 1 |
| `data/strategy_store/pg.py` | New — PostgreSQL strategy store | 1 |
| `data/store_factory.py` | New — backend-agnostic store factory | 1 |
| `strategy/runner/realtime_runner.py` | Full paper trading runner | 2 |
| `data/source/binance_ws.py` | New — WebSocket manager | 3 |
| `strategy/broker/executor/binance_spot.py` | Full executor implementation | 4 |
| `data/source/binance.py` | Trading API endpoints | 4 |
| `strategy/broker/realtime_spot.py` | Full live broker implementation | 5 |
| `strategy/risk/` | New module | 6 |

## Code Reuse

- `BacktestRunner._process_bar()` -> template for `RealtimeRunner.process_bar()`
- `BacktestBroker` -> template for `RealtimeSpotBroker`
- `VirtualSpotExecutor` -> executor for paper trading (Phase 2), interface for BinanceSpotExecutor
- `BinanceClient` -> extend, don't rewrite
- `BinanceSpotLoader.sync()` -> data ingestion for Phase 2
- `DuckDbStrategyStore` -> template for SQLite/PostgreSQL stores (Phase 1), persistence layer (production-ready)
- `StrategyState.touch()` / `last_event_id` -> idempotency mechanism

## Verification

After each phase:
1. `pytest tests/ -v` — all existing tests must pass
2. New tests for all new code
3. Phase 2 — paper trading on real data (manual verification)
4. Phase 7 — full E2E on Binance Testnet
