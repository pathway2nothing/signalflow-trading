"""OHLCV timeframe resampling and alignment utilities.

Provides functions to resample market data between timeframes, detect
source timeframes, and select optimal download timeframes per exchange.

Example:
    ```python
    from signalflow.data.resample import align_to_timeframe, select_best_timeframe

    # Auto-detect source timeframe and resample to 1h
    df_1h = align_to_timeframe(raw_df, target_tf="1h")

    # Pick best download timeframe for exchange
    best_tf = select_best_timeframe("bybit", target_tf="8h")
    # Returns "4h" (largest Bybit tf that divides 8h)
    ```
"""

from __future__ import annotations

import warnings
from typing import Final

import polars as pl

from signalflow.data.source._helpers import TIMEFRAME_MS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMEFRAME_MINUTES: Final[dict[str, int]] = {tf: ms // 60_000 for tf, ms in TIMEFRAME_MS.items()}
"""Mapping from timeframe string to minutes (e.g. ``"4h"`` → ``240``)."""

DEFAULT_FILL_RULES: Final[dict[str, str]] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "trades": "sum",
}
"""Default OHLCV aggregation rules for resampling."""

EXCHANGE_TIMEFRAMES: Final[dict[str, set[str]]] = {
    "binance": {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"},
    "bybit": {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"},
    "okx": {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"},
    "kraken_spot": {"1m", "5m", "15m", "30m", "1h", "4h", "1d"},
    "kraken_futures": {"1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d"},
    "deribit": {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"},
    "hyperliquid": {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d"},
    "whitebit": {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"},
}
"""Supported timeframes per exchange, derived from each loader's interval map."""

# Polars-compatible duration strings for ``dt.truncate()``.
_TRUNCATE_EVERY: Final[dict[str, str]] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def timeframe_to_minutes(tf: str) -> int:
    """Convert a timeframe string to minutes.

    Args:
        tf: Timeframe string (e.g. ``"1h"``, ``"4h"``, ``"1d"``).

    Returns:
        Number of minutes.

    Raises:
        ValueError: If *tf* is not a recognised timeframe.

    Example:
        >>> timeframe_to_minutes("4h")
        240
    """
    if tf not in TIMEFRAME_MINUTES:
        valid = ", ".join(sorted(TIMEFRAME_MINUTES, key=lambda k: TIMEFRAME_MINUTES[k]))
        raise ValueError(f"Unknown timeframe {tf!r}. Valid: {valid}")
    return TIMEFRAME_MINUTES[tf]


def can_resample(source_tf: str, target_tf: str) -> bool:
    """Check whether *source_tf* can be resampled to *target_tf*.

    Resampling is possible when *target_tf* is an exact integer multiple
    of *source_tf* (and target >= source).

    Args:
        source_tf: Source timeframe (e.g. ``"1m"``).
        target_tf: Target timeframe (e.g. ``"1h"``).

    Returns:
        ``True`` if resampling is possible.

    Example:
        >>> can_resample("1h", "4h")
        True
        >>> can_resample("1h", "3h")
        False
    """
    if source_tf not in TIMEFRAME_MINUTES or target_tf not in TIMEFRAME_MINUTES:
        return False
    src = TIMEFRAME_MINUTES[source_tf]
    tgt = TIMEFRAME_MINUTES[target_tf]
    if tgt < src:
        return False
    return tgt % src == 0


def detect_timeframe(
    df: pl.DataFrame,
    ts_col: str = "timestamp",
    pair_col: str = "pair",
) -> str:
    """Auto-detect the timeframe of an OHLCV DataFrame.

    Computes the most common timestamp delta per pair and maps it to the
    closest known timeframe string.

    Args:
        df: OHLCV DataFrame with at least *ts_col* and *pair_col*.
        ts_col: Timestamp column name.
        pair_col: Pair/group column name.

    Returns:
        Detected timeframe string (e.g. ``"1h"``).

    Raises:
        ValueError: If the DataFrame is too small or the delta doesn't
            match any known timeframe.

    Example:
        >>> detect_timeframe(hourly_df)
        '1h'
    """
    if df.height < 2:
        raise ValueError("DataFrame must have at least 2 rows to detect timeframe")

    # Compute per-pair diffs and find the mode.
    deltas = (
        df.sort([pair_col, ts_col])
        .with_columns(pl.col(ts_col).diff().over(pair_col).alias("_delta"))
        .filter(pl.col("_delta").is_not_null())
        .select("_delta")
    )

    if deltas.height == 0:
        raise ValueError("Cannot detect timeframe: no timestamp deltas computed")

    mode_delta = deltas.group_by("_delta").len().sort("len", descending=True).row(0)[0]

    delta_minutes = int(mode_delta.total_seconds() // 60)

    # Match to known timeframe.
    minutes_to_tf = {v: k for k, v in TIMEFRAME_MINUTES.items()}
    if delta_minutes not in minutes_to_tf:
        raise ValueError(
            f"Detected delta of {delta_minutes} minutes does not match any known "
            f"timeframe. Known: {sorted(minutes_to_tf.keys())}"
        )

    return minutes_to_tf[delta_minutes]


# ---------------------------------------------------------------------------
# Core resample
# ---------------------------------------------------------------------------

_AGG_MAP = {
    "first": lambda col: pl.col(col).first(),
    "last": lambda col: pl.col(col).last(),
    "max": lambda col: pl.col(col).max(),
    "min": lambda col: pl.col(col).min(),
    "sum": lambda col: pl.col(col).sum(),
    "mean": lambda col: pl.col(col).mean(),
}


def resample_ohlcv(
    df: pl.DataFrame,
    source_tf: str,
    target_tf: str,
    *,
    pair_col: str = "pair",
    ts_col: str = "timestamp",
    fill_rules: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Resample an OHLCV DataFrame from *source_tf* to *target_tf*.

    Args:
        df: Source DataFrame with OHLCV columns.
        source_tf: Current timeframe of the data.
        target_tf: Desired target timeframe.
        pair_col: Pair/group column name.
        ts_col: Timestamp column name.
        fill_rules: Per-column aggregation rules. Defaults to
            :data:`DEFAULT_FILL_RULES`. Unknown columns default to
            ``"last"``.

    Returns:
        Resampled DataFrame.

    Raises:
        ValueError: If resampling is not possible.

    Example:
        >>> df_4h = resample_ohlcv(df_1h, "1h", "4h")
    """
    if source_tf == target_tf:
        return df

    if not can_resample(source_tf, target_tf):
        raise ValueError(f"Cannot resample from {source_tf} to {target_tf}: target must be an exact multiple of source")

    rules = {**DEFAULT_FILL_RULES, **(fill_rules or {})}
    every = _TRUNCATE_EVERY[target_tf]

    # Build aggregation expressions.
    agg_exprs: list[pl.Expr] = [pl.col(ts_col).max()]
    skip = {pair_col, ts_col}

    for col in df.columns:
        if col in skip:
            continue
        rule = rules.get(col, "last")
        if rule not in _AGG_MAP:
            raise ValueError(f"Unknown fill rule {rule!r} for column {col!r}")
        agg_exprs.append(_AGG_MAP[rule](col))

    # Truncate timestamps to target bins and aggregate.
    result = (
        df.with_columns(pl.col(ts_col).dt.truncate(every).alias("_bin"))
        .group_by([pair_col, "_bin"], maintain_order=True)
        .agg(agg_exprs)
        .drop("_bin")
        .sort([pair_col, ts_col])
    )

    return result


# ---------------------------------------------------------------------------
# High-level alignment
# ---------------------------------------------------------------------------


def align_to_timeframe(
    df: pl.DataFrame,
    target_tf: str,
    *,
    pair_col: str = "pair",
    ts_col: str = "timestamp",
    fill_rules: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Detect source timeframe and resample to *target_tf* if possible.

    If the source timeframe equals the target, returns the data unchanged.
    If resampling is not possible (e.g. ``"3m"`` → ``"2m"``), emits a
    warning and returns the original data.

    Args:
        df: OHLCV DataFrame.
        target_tf: Desired timeframe.
        pair_col: Pair/group column name.
        ts_col: Timestamp column name.
        fill_rules: Per-column aggregation rules.

    Returns:
        Resampled DataFrame, or original if alignment is not possible.

    Example:
        >>> df_1h = align_to_timeframe(raw_df, "1h")
    """
    try:
        source_tf = detect_timeframe(df, ts_col=ts_col, pair_col=pair_col)
    except ValueError:
        warnings.warn(
            f"Could not detect source timeframe; returning data as-is. Target was {target_tf!r}.",
            stacklevel=2,
        )
        return df

    if source_tf == target_tf:
        return df

    if not can_resample(source_tf, target_tf):
        warnings.warn(
            f"Cannot resample from {source_tf} to {target_tf} "
            f"(target must be an exact multiple of source). Returning data as-is.",
            stacklevel=2,
        )
        return df

    return resample_ohlcv(
        df,
        source_tf,
        target_tf,
        pair_col=pair_col,
        ts_col=ts_col,
        fill_rules=fill_rules,
    )


# ---------------------------------------------------------------------------
# Exchange helpers
# ---------------------------------------------------------------------------


def select_best_timeframe(exchange: str, target_tf: str) -> str:
    """Select the best download timeframe for an exchange.

    Strategy:
        1. If the exchange supports *target_tf*, return it.
        2. Otherwise pick the largest supported timeframe that evenly
           divides *target_tf*.
        3. If nothing divides evenly, return the smallest supported
           timeframe (``"1m"`` in most cases).

    Args:
        exchange: Exchange name (lowercase, e.g. ``"bybit"``).
        target_tf: Desired target timeframe.

    Returns:
        Best timeframe to download.

    Raises:
        ValueError: If *exchange* is unknown.

    Example:
        >>> select_best_timeframe("bybit", "8h")
        '4h'
        >>> select_best_timeframe("binance", "8h")
        '8h'
    """
    exchange = exchange.lower()
    if exchange not in EXCHANGE_TIMEFRAMES:
        valid = ", ".join(sorted(EXCHANGE_TIMEFRAMES))
        raise ValueError(f"Unknown exchange {exchange!r}. Known: {valid}")

    supported = EXCHANGE_TIMEFRAMES[exchange]

    if target_tf in supported:
        return target_tf

    target_min = timeframe_to_minutes(target_tf)

    # Find largest divisor.
    best: str | None = None
    best_min = 0
    for tf in supported:
        tf_min = TIMEFRAME_MINUTES[tf]
        if tf_min <= target_min and target_min % tf_min == 0 and tf_min > best_min:
            best = tf
            best_min = tf_min

    if best is not None:
        return best

    # Fallback: smallest supported timeframe.
    return min(supported, key=lambda t: TIMEFRAME_MINUTES[t])
