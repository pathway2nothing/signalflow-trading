"""Utility functions for target labeling.

Provides functions for masking target labels during events
and other labeling-related utilities.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

from signalflow.core.containers.signals import Signals


def mask_targets_by_signals(
    df: pl.DataFrame,
    signals: Signals,
    mask_signal_types: set[str],
    horizon_bars: int,
    cooldown_bars: int = 60,
    target_columns: list[str] | None = None,
    pair_col: str = "pair",
    ts_col: str = "timestamp",
) -> pl.DataFrame:
    """Mask target columns for timestamps overlapping with specified signals.

    For each signal at time T with type in mask_signal_types:
    - Masks range [T - horizon_bars, T + cooldown_bars]

    This is useful for excluding labels that overlap with exogenous events
    (e.g., flash crashes, global market events) from training data, since
    no feature could predict such events.

    Args:
        df: DataFrame with target columns.
        signals: Signals object containing detected events.
        mask_signal_types: Signal types to mask (e.g. {"flash_crash", "global_event"}).
        horizon_bars: Forward horizon (bars before signal that "see" it).
        cooldown_bars: Bars after signal to mask (default: 60).
        target_columns: Columns to mask (default: all columns ending with "_label").
        pair_col: Pair column name (default: "pair").
        ts_col: Timestamp column name (default: "timestamp").

    Returns:
        DataFrame with affected target columns set to null.

    Example:
        ```python
        from signalflow.detector import ZScoreAnomalyDetector
        from signalflow.target.utils import mask_targets_by_signals

        # Detect anomalies
        detector = ZScoreAnomalyDetector(threshold=4.0)
        signals = detector.run(raw_data_view)

        # Mask labels overlapping with flash crashes
        labeled_df = mask_targets_by_signals(
            df=labeled_df,
            signals=signals,
            mask_signal_types={"anomaly_low"},  # flash crashes
            horizon_bars=60,
            cooldown_bars=60,
        )
        ```

    Note:
        - Masking is done per-pair: signal at time T for pair A only masks
          labels for pair A at affected timestamps.
        - If signals DataFrame is empty or has no matching signal_types,
          the input DataFrame is returned unchanged.
    """
    # Get events matching the specified signal types
    signals_df = signals.value

    if signals_df.height == 0:
        return df

    if "signal_type" not in signals_df.columns:
        logger.warning("Signals DataFrame has no 'signal_type' column, returning unchanged")
        return df

    # Filter to matching signal types
    events = signals_df.filter(pl.col("signal_type").is_in(list(mask_signal_types)))

    if events.height == 0:
        logger.debug(f"No signals matching types {mask_signal_types}, returning unchanged")
        return df

    # Determine target columns to mask
    if target_columns is None:
        target_columns = [c for c in df.columns if c.endswith("_label")]

    if not target_columns:
        logger.warning("No target columns found to mask")
        return df

    existing_cols = [c for c in target_columns if c in df.columns]
    if not existing_cols:
        logger.warning(f"Target columns {target_columns} not found in DataFrame")
        return df

    # Get all unique timestamps for efficient index lookup
    all_timestamps = df.select([pair_col, ts_col]).unique().sort([pair_col, ts_col])

    # Process per pair for correct masking
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

            # Create masked DataFrame
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
    ts_col: str = "timestamp",
) -> pl.DataFrame:
    """Mask target columns for timestamps overlapping with event timestamps.

    Simpler version of mask_targets_by_signals that works with raw timestamps
    instead of Signals objects. Applies masking globally (not per-pair).

    Args:
        df: DataFrame with target columns.
        event_timestamps: List of event timestamps to mask around.
        horizon_bars: Forward horizon (bars before event that "see" it).
        cooldown_bars: Bars after event to mask (default: 60).
        target_columns: Columns to mask (default: all columns ending with "_label").
        ts_col: Timestamp column name (default: "timestamp").

    Returns:
        DataFrame with affected target columns set to null.

    Example:
        ```python
        from signalflow.target.utils import mask_targets_by_timestamps
        from datetime import datetime

        # Mask around known events
        labeled_df = mask_targets_by_timestamps(
            df=labeled_df,
            event_timestamps=[
                datetime(2024, 3, 1, 10, 30),  # Known flash crash
                datetime(2024, 5, 15, 14, 0),  # Fed announcement
            ],
            horizon_bars=60,
            cooldown_bars=120,
        )
        ```
    """
    if not event_timestamps:
        return df

    # Determine target columns to mask
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
