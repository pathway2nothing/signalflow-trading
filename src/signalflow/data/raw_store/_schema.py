"""Shared schema helpers for raw data store backends."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from signalflow.core.registry import default_registry

# Columns that are always NOT NULL (core OHLCV).
CORE_COLUMNS = ("open", "high", "low", "close", "volume")


def normalize_ts(ts: datetime) -> datetime:
    """Strip timezone and round up to next minute if seconds/microseconds != 0."""
    if ts.second != 0 or ts.microsecond != 0:
        return ts.replace(tzinfo=None, second=0, microsecond=0) + timedelta(minutes=1)
    return ts.replace(tzinfo=None)


def resolve_columns(data_type: str) -> list[str]:
    """Return ordered value columns for *data_type* (excludes pair, timestamp).

    Order: open, high, low, close, volume, <extra sorted>, trades.
    """
    reg_cols = default_registry.get_raw_data_columns(data_type)
    fixed = {"pair", "timestamp"}
    ohlcv = [c for c in CORE_COLUMNS if c in reg_cols]
    extra = sorted(reg_cols - fixed - set(CORE_COLUMNS))
    return ohlcv + extra + ["trades"]


def polars_schema(columns: list[str]) -> dict[str, pl.DataType]:
    """Build a Polars schema dict for pair + timestamp + *columns*."""
    schema: dict[str, pl.DataType] = {"pair": pl.Utf8, "timestamp": pl.Datetime}
    for c in columns:
        schema[c] = pl.Int64 if c == "trades" else pl.Float64
    return schema
