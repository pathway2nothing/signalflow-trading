"""CUSUM (Cumulative Sum) market-wide detector.

Detects sustained regime shifts by tracking cumulative deviations of the
cross-pair aggregate return from its expected value. Unlike point-in-time
z-score detection, CUSUM accumulates evidence over multiple bars, making
it better at detecting gradual structural changes.

Reference:
    Page, E. S. (1954) - "Continuous Inspection Schemes"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl
from loguru import logger

from signalflow.core import RawDataView, Signals
from signalflow.core.decorators import sf_component
from signalflow.core.enums import SfComponentType, SignalCategory
from signalflow.detector.base import SignalDetector


@dataclass
@sf_component(name="market_wide/cusum")
class MarketCusumDetector(SignalDetector):
    """Detects market-wide signals via CUSUM of cross-pair aggregate return.

    Unlike point-in-time z-score detection, CUSUM accumulates evidence over
    multiple bars, making it better at detecting gradual structural changes.

    Algorithm:
        1. Compute cross-pair mean return at each timestamp.
        2. Compute rolling mean ``mu`` (expected return) over ``rolling_window``.
        3. S_pos = max(0, S_pos + (x - mu - drift))
        4. S_neg = max(0, S_neg + (-x + mu - drift))
        5. Signal if S_pos > cusum_threshold or S_neg > cusum_threshold.
        6. Reset S_pos, S_neg to 0 after signal detection.

    Attributes:
        drift: Slack parameter (allowance for normal variation).
        cusum_threshold: Decision interval for CUSUM alarm.
        rolling_window: Window for estimating expected return (mu).
        min_pairs: Minimum number of active pairs at a timestamp.
        return_window: Bars for return computation.
        signal_type_name: Signal type name for detected signals.

    Example:
        ```python
        from signalflow.detector import MarketCusumDetector
        from signalflow.target.utils import mask_targets_by_signals

        # Detect market-wide regime shifts
        detector = MarketCusumDetector(cusum_threshold=0.05)
        signals = detector.run(raw_data_view)

        # Mask labels overlapping with detected signals
        labeled_df = mask_targets_by_signals(
            df=labeled_df,
            signals=signals,
            mask_signal_types={"structural_break"},
            horizon_bars=60,
        )
        ```

    Note:
        Returns one signal per timestamp (not per pair) since market-wide
        signals affect all pairs simultaneously. The signal has a synthetic
        "ALL" pair.

    Reference:
        Page, E. S. (1954) - "Continuous Inspection Schemes"
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.DETECTOR

    signal_category: SignalCategory = SignalCategory.MARKET_WIDE
    allowed_signal_types: set[str] | None = field(default_factory=lambda: {"structural_break"})

    drift: float = 0.005
    cusum_threshold: float = 0.05
    rolling_window: int = 100
    min_pairs: int = 5
    return_window: int = 1
    price_col: str = "close"
    signal_type_name: str = "structural_break"

    def __post_init__(self) -> None:
        self.allowed_signal_types = {self.signal_type_name}

    def preprocess(
        self,
        raw_data_view: RawDataView,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Preprocess raw data: compute log returns.

        Returns raw OHLCV with _ret column added.
        """
        df = super().preprocess(raw_data_view, context)

        df = df.sort([self.pair_col, self.ts_col]).with_columns(
            (pl.col(self.price_col) / pl.col(self.price_col).shift(self.return_window))
            .log()
            .over(self.pair_col)
            .alias("_ret")
        )

        return df

    def detect(
        self,
        features: pl.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> Signals:
        """Detect market-wide signals via CUSUM.

        Args:
            features: Multi-pair OHLCV DataFrame with _ret column.
            context: Additional context (unused).

        Returns:
            Signals with structural_break signal type for detected timestamps.
        """
        min_samples = self.rolling_window

        # Cross-pair mean return per timestamp
        agg_return = (
            features.filter(pl.col("_ret").is_not_null() & pl.col("_ret").is_finite())
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
            pl.col("_agg_return").rolling_mean(window_size=self.rolling_window, min_samples=min_samples).alias("_mu")
        )

        # CUSUM with reset (sequential — inherently stateful)
        x_arr = agg_return.get_column("_agg_return").to_numpy()
        mu_arr = agg_return.get_column("_mu").to_numpy()

        n = len(x_arr)
        is_signal = np.zeros(n, dtype=bool)
        cusum_values = np.zeros(n, dtype=np.float64)
        s_pos = 0.0
        s_neg = 0.0

        for i in range(n):
            if np.isnan(mu_arr[i]) or np.isnan(x_arr[i]):
                continue

            deviation = x_arr[i] - mu_arr[i]
            s_pos = max(0.0, s_pos + deviation - self.drift)
            s_neg = max(0.0, s_neg - deviation - self.drift)

            cusum_values[i] = max(s_pos, s_neg)

            if s_pos > self.cusum_threshold or s_neg > self.cusum_threshold:
                is_signal[i] = True
                s_pos = 0.0
                s_neg = 0.0

        n_signals = int(is_signal.sum())
        logger.info(f"MarketCusumDetector: detected {n_signals} timestamps")

        if n_signals == 0:
            return Signals(
                pl.DataFrame(
                    schema={
                        self.pair_col: pl.Utf8,
                        self.ts_col: pl.Datetime,
                        "signal_type": pl.Utf8,
                        "signal": pl.Int64,
                        "probability": pl.Float64,
                    }
                )
            )

        # Filter to signal timestamps only
        signal_df = agg_return.with_columns(
            [
                pl.Series("_is_signal", is_signal),
                pl.Series("_cusum", cusum_values),
            ]
        ).filter(pl.col("_is_signal"))

        # Create signals with synthetic "ALL" pair for market-wide signals
        # Probability based on normalized cusum value
        signals_df = signal_df.select(
            [
                pl.lit("ALL").alias(self.pair_col),
                self.ts_col,
                pl.lit(self.signal_type_name).alias("signal_type"),
                pl.lit(1).alias("signal"),
                (pl.col("_cusum") / self.cusum_threshold).clip(0.0, 1.0).alias("probability"),
            ]
        )

        return Signals(signals_df)


# Backward compatibility alias with legacy default signal_type_name
@dataclass
class CusumEventDetector(MarketCusumDetector):
    """Backward compatibility alias for MarketCusumDetector.

    .. deprecated::
        Use MarketCusumDetector instead. This alias preserves the legacy
        "global_event" signal_type_name for backward compatibility.
    """

    signal_type_name: str = "global_event"
    allowed_signal_types: set[str] | None = field(default_factory=lambda: {"global_event"})
