"""Utility functions for target labeling."""

from typing import Any

import numpy as np
import polars as pl
from loguru import logger


def _signals_frame(signals: Any) -> pl.DataFrame:
    """Extract a Polars frame from a signals container or raw frame."""
    s = getattr(signals, "value", signals)
    if not isinstance(s, pl.DataFrame):
        raise TypeError(f"Unsupported signals value type: {type(s)}")
    return s


def mask_targets_by_signals(
    df: pl.DataFrame,
    signals: Any,
    mask_signal_types: set[str],
    horizon_bars: int,
    cooldown_bars: int = 60,
    target_columns: list[str] | None = None,
    pair_col: str = "pair",
    ts_col: str = "ts",
) -> pl.DataFrame:
    """Mask target columns for timestamps overlapping with specified signals."""
    signals_df = _signals_frame(signals)

    if signals_df.height == 0:
        return df

    if "signal_type" not in signals_df.columns:
        logger.warning("Signals DataFrame has no 'signal_type' column, returning unchanged")
        return df

    events = signals_df.filter(pl.col("signal_type").is_in(list(mask_signal_types)))

    if events.height == 0:
        logger.debug(f"No signals matching types {mask_signal_types}, returning unchanged")
        return df

    if target_columns is None:
        target_columns = [c for c in df.columns if c.endswith("_label")]

    if not target_columns:
        logger.warning("No target columns found to mask")
        return df

    existing_cols = [c for c in target_columns if c in df.columns]
    if not existing_cols:
        logger.warning(f"Target columns {target_columns} not found in DataFrame")
        return df

    masked_rows: list[pl.DataFrame] = []

    for pair_name in df.get_column(pair_col).unique().to_list():
        pair_df = df.filter(pl.col(pair_col) == pair_name)
        pair_events = events.filter(pl.col(pair_col) == pair_name)

        if pair_events.height == 0:
            masked_rows.append(pair_df)
            continue

        ts_array = pair_df.get_column(ts_col).to_numpy()
        event_ts = pair_events.get_column(ts_col).to_numpy()

        mask = np.zeros(len(ts_array), dtype=bool)

        for evt in event_ts:
            idx = np.searchsorted(ts_array, evt)
            if idx >= len(ts_array):
                continue

            start = max(0, idx - horizon_bars)
            end = min(len(ts_array), idx + cooldown_bars + 1)
            mask[start:end] = True

        n_masked = int(mask.sum())
        if n_masked > 0:
            logger.debug(f"Masking {n_masked} timestamps for pair {pair_name}")

            mask_series = pl.Series("_mask", mask)
            pair_df = (
                pair_df.with_columns(mask_series)
                .with_columns(
                    [
                        pl.when(pl.col("_mask")).then(pl.lit(None)).otherwise(pl.col(col)).alias(col)
                        for col in existing_cols
                    ]
                )
                .drop("_mask")
            )

        masked_rows.append(pair_df)

    if not masked_rows:
        return df

    result = pl.concat(masked_rows, how="vertical_relaxed")

    total_events = events.height
    logger.info(
        f"mask_targets_by_signals: masked around {total_events} events "
        f"(types={mask_signal_types}, horizon={horizon_bars}, cooldown={cooldown_bars})"
    )

    return result


def mask_targets_by_timestamps(
    df: pl.DataFrame,
    event_timestamps: list,
    horizon_bars: int,
    cooldown_bars: int = 60,
    target_columns: list[str] | None = None,
    ts_col: str = "ts",
) -> pl.DataFrame:
    """Mask target columns for timestamps overlapping event timestamps."""
    if not event_timestamps:
        return df

    if target_columns is None:
        target_columns = [c for c in df.columns if c.endswith("_label")]

    if not target_columns:
        logger.warning("No target columns found to mask")
        return df

    existing_cols = [c for c in target_columns if c in df.columns]
    if not existing_cols:
        logger.warning(f"Target columns {target_columns} not found in DataFrame")
        return df

    ts_array = df.get_column(ts_col).to_numpy()
    mask = np.zeros(len(ts_array), dtype=bool)

    for evt in event_timestamps:
        idx = np.searchsorted(ts_array, evt)
        if idx >= len(ts_array):
            continue

        start = max(0, idx - horizon_bars)
        end = min(len(ts_array), idx + cooldown_bars + 1)
        mask[start:end] = True

    n_masked = int(mask.sum())
    if n_masked == 0:
        return df

    logger.info(f"mask_targets_by_timestamps: masked {n_masked} timestamps around {len(event_timestamps)} events")

    mask_series = pl.Series("_mask", mask)
    result = (
        df.with_columns(mask_series)
        .with_columns(
            [pl.when(pl.col("_mask")).then(pl.lit(None)).otherwise(pl.col(col)).alias(col) for col in existing_cols]
        )
        .drop("_mask")
    )

    return result
