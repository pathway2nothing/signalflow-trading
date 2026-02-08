"""CUSUM (Cumulative Sum) global event detector.

Detects sustained regime shifts by tracking cumulative deviations of the
cross-pair aggregate return from its expected value. Unlike point-in-time
z-score detection, CUSUM accumulates evidence over multiple bars, making
it better at detecting gradual structural changes.

Reference:
    Page, E. S. (1954) - "Continuous Inspection Schemes"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from loguru import logger

from signalflow.core.decorators import sf_component
from signalflow.detector.event.base import EventDetectorBase


@dataclass
@sf_component(name="event_detector/cusum")
class CusumEventDetector(EventDetectorBase):
    """Detects global events via CUSUM of cross-pair aggregate return.

    Algorithm:
        1. Compute cross-pair mean return at each timestamp.
        2. Compute rolling mean ``mu`` (expected return) over ``rolling_window``.
        3. S_pos = max(0, S_pos + (x - mu - drift))
        4. S_neg = max(0, S_neg + (-x + mu - drift))
        5. Event if S_pos > cusum_threshold or S_neg > cusum_threshold.
        6. Reset S_pos, S_neg to 0 after event detection.

    Attributes:
        drift: Slack parameter (allowance for normal variation).
        cusum_threshold: Decision interval for CUSUM alarm.
        rolling_window: Window for estimating expected return (mu).
        min_pairs: Minimum number of active pairs at a timestamp.
        return_window: Bars for return computation.
    """

    drift: float = 0.005
    cusum_threshold: float = 0.05
    rolling_window: int = 100
    min_pairs: int = 5
    return_window: int = 1

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detect global event timestamps via CUSUM.

        Args:
            df: Multi-pair OHLCV DataFrame sorted by (pair, timestamp).

        Returns:
            DataFrame with columns: (timestamp, _is_global_event).
        """
        self._validate(df)

        # Step 1: log-returns per pair
        returns_df = df.sort([self.pair_col, self.ts_col]).with_columns(
            (pl.col(self.price_col) / pl.col(self.price_col).shift(self.return_window))
            .log()
            .over(self.pair_col)
            .alias("_ret")
        )

        # Step 2: cross-pair mean return per timestamp
        min_periods = self.rolling_window
        agg_return = (
            returns_df.filter(pl.col("_ret").is_not_null() & pl.col("_ret").is_finite())
            .group_by(self.ts_col)
            .agg(
                pl.col("_ret").mean().alias("_agg_return"),
                pl.col("_ret").count().alias("_n_pairs"),
            )
            .filter(pl.col("_n_pairs") >= self.min_pairs)
            .sort(self.ts_col)
        )

        # Compute rolling mean (expected return mu)
        agg_return = agg_return.with_columns(
            pl.col("_agg_return").rolling_mean(window_size=self.rolling_window, min_periods=min_periods).alias("_mu")
        )

        # Steps 3-6: CUSUM with reset (sequential — inherently stateful)
        x_arr = agg_return.get_column("_agg_return").to_numpy()
        mu_arr = agg_return.get_column("_mu").to_numpy()

        n = len(x_arr)
        is_event = np.zeros(n, dtype=bool)
        s_pos = 0.0
        s_neg = 0.0

        for i in range(n):
            if np.isnan(mu_arr[i]) or np.isnan(x_arr[i]):
                continue

            deviation = x_arr[i] - mu_arr[i]
            s_pos = max(0.0, s_pos + deviation - self.drift)
            s_neg = max(0.0, s_neg - deviation - self.drift)

            if s_pos > self.cusum_threshold or s_neg > self.cusum_threshold:
                is_event[i] = True
                s_pos = 0.0
                s_neg = 0.0

        result = pl.DataFrame(
            {
                self.ts_col: agg_return.get_column(self.ts_col),
                "_is_global_event": is_event,
            }
        )

        n_events = int(is_event.sum())
        logger.info(f"CusumEventDetector: detected {n_events} event timestamps")

        return result
