# src/signalflow/data/strategy_store/schema.py
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategy_state (
  strategy_id TEXT PRIMARY KEY,
  last_ts TIMESTAMP,
  last_event_id TEXT,
  payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
  strategy_id TEXT NOT NULL,
  ts TIMESTAMP NOT NULL,
  position_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (strategy_id, ts, position_id)
);

CREATE TABLE IF NOT EXISTS trades (
  strategy_id TEXT NOT NULL,
  ts TIMESTAMP NOT NULL,
  trade_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (strategy_id, trade_id)
);

CREATE TABLE IF NOT EXISTS metrics (
  strategy_id TEXT NOT NULL,
  ts TIMESTAMP NOT NULL,
  name TEXT NOT NULL,
  value DOUBLE NOT NULL,
  PRIMARY KEY (strategy_id, ts, name)
);

CREATE INDEX IF NOT EXISTS idx_metrics_strategy_ts ON metrics(strategy_id, ts);
CREATE INDEX IF NOT EXISTS idx_positions_strategy_ts ON positions(strategy_id, ts);
"""

PG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategy_state (
  strategy_id TEXT PRIMARY KEY,
  last_ts TIMESTAMP,
  last_event_id TEXT,
  payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
  strategy_id TEXT NOT NULL,
  ts TIMESTAMP NOT NULL,
  position_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (strategy_id, ts, position_id)
);

CREATE TABLE IF NOT EXISTS trades (
  strategy_id TEXT NOT NULL,
  ts TIMESTAMP NOT NULL,
  trade_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (strategy_id, trade_id)
);

CREATE TABLE IF NOT EXISTS metrics (
  strategy_id TEXT NOT NULL,
  ts TIMESTAMP NOT NULL,
  name TEXT NOT NULL,
  value DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (strategy_id, ts, name)
);

CREATE INDEX IF NOT EXISTS idx_metrics_strategy_ts ON metrics(strategy_id, ts);
CREATE INDEX IF NOT EXISTS idx_positions_strategy_ts ON positions(strategy_id, ts);
"""
