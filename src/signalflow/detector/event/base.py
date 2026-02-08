"""Base class for global event detectors.

All event detectors share the same masking logic -- only the detection
algorithm differs. Subclass ``EventDetectorBase`` and implement
``detect()`` to create a new detector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl
from loguru import logger

from signalflow.core.enums import SfComponentType


@dataclass
class EventDetectorBase(ABC):
    """Abstract base for global event detectors.

    Detects timestamps where exogenous market-wide events occur.
    Events are used to mask (nullify) target labels whose forward windows
    overlap with the event, since no feature could predict an exogenous shock.

    Subclasses must implement ``detect(df)`` which returns a DataFrame
    with columns ``(ts_col, _is_global_event)``.

    Shared ``mask_targets`` logic handles the masking window:
        For event at time T and horizon H:
        - Labels at ``t in [T - H, T]``: forward window crosses event -> NaN
        - Labels at ``t in [T, T + cooldown]``: post-event chaos -> NaN
        - Total masked range: ``[T - H, T + cooldown]``

    Attributes:
        cooldown_bars: Bars to mask after the event.
        pair_col: Pair column name.
        ts_col: Timestamp column name.
        price_col: Price column for return computation.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.EVENT_DETECTOR

    cooldown_bars: int = 60
    pair_col: str = "pair"
    ts_col: str = "timestamp"
    price_col: str = "close"

    @abstractmethod
    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detect global event timestamps.

        Args:
            df: Multi-pair OHLCV DataFrame sorted by (pair, timestamp).

        Returns:
            DataFrame with columns: (ts_col, _is_global_event).
            One row per unique timestamp.
        """
        ...

    def mask_targets(
        self,
        df: pl.DataFrame,
        event_timestamps: pl.DataFrame,
        horizon_configs: list,
        target_columns_by_horizon: dict[str, list[str]],
    ) -> pl.DataFrame:
        """Mask target columns at timestamps affected by global events.

        For each horizon H and event at time T, nullifies target columns
        in the range ``[T - H, T + cooldown]``.

        Args:
            df: DataFrame with target columns.
            event_timestamps: Output of ``detect()`` with _is_global_event column.
            horizon_configs: List of HorizonConfig with name and horizon attributes.
            target_columns_by_horizon: Mapping ``{horizon_name: [col1, col2, ...]}``

        Returns:
            DataFrame with affected target columns set to null.
        """
        events = event_timestamps.filter(pl.col("_is_global_event"))
        if events.height == 0:
            return df

        event_ts = events.get_column(self.ts_col)

        all_timestamps = df.select(self.ts_col).unique().sort(self.ts_col)
        ts_array = all_timestamps.get_column(self.ts_col).to_numpy()

        for h_config in horizon_configs:
            cols = target_columns_by_horizon.get(h_config.name, [])
            if not cols:
                continue

            existing_cols = [c for c in cols if c in df.columns]
            if not existing_cols:
                continue

            mask = np.zeros(len(ts_array), dtype=bool)

            for evt in event_ts.to_numpy():
                idx = np.searchsorted(ts_array, evt)
                if idx >= len(ts_array):
                    continue

                start = max(0, idx - h_config.horizon)
                end = min(len(ts_array), idx + self.cooldown_bars + 1)
                mask[start:end] = True

            masked_ts_df = pl.DataFrame({self.ts_col: ts_array[mask]}).with_columns(pl.lit(True).alias("_masked"))

            df = (
                df.join(masked_ts_df, on=self.ts_col, how="left")
                .with_columns(
                    [
                        pl.when(pl.col("_masked").fill_null(False)).then(pl.lit(None)).otherwise(pl.col(col)).alias(col)
                        for col in existing_cols
                    ]
                )
                .drop("_masked")
            )

            n_masked = int(mask.sum())
            logger.debug(f"Masked {n_masked} timestamps for horizon '{h_config.name}' ({len(existing_cols)} columns)")

        return df

    def _validate(self, df: pl.DataFrame) -> None:
        """Validate required columns are present."""
        required = {self.pair_col, self.ts_col, self.price_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
