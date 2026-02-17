"""Real-time volatility regime detector.

Classifies the current volatility regime using backward-looking realized
volatility percentile within a rolling lookback window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from signalflow.core import RawDataView, Signals, sf_component
from signalflow.core.enums import SignalCategory
from signalflow.detector.base import SignalDetector


@dataclass
@sf_component(name="volatility_detector")
class VolatilityDetector(SignalDetector):
    """Detects volatility regime shifts in real-time (backward-looking only).

    Unlike ``VolatilityRegimeLabeler``, this detector uses only past and
    current data and is safe for live trading.

    Algorithm:
        1. Compute log returns: log(close[t] / close[t-1])
        2. Backward realized volatility: std of last ``vol_window`` returns
        3. Rolling percentile of realized vol over ``lookback_window``
        4. If percentile > upper_quantile -> "high_volatility"
        5. If percentile < lower_quantile -> "low_volatility"
        6. Otherwise: no signal emitted

    Attributes:
        price_col (str): Price column name. Default: "close".
        vol_window (int): Window for realized vol calculation. Default: 60.
        lookback_window (int): Window for percentile computation. Default: 1440.
        upper_quantile (float): Upper percentile threshold. Default: 0.67.
        lower_quantile (float): Lower percentile threshold. Default: 0.33.

    Example:
        ```python
        from signalflow.detector.volatility_detector import VolatilityDetector

        detector = VolatilityDetector(
            vol_window=60,
            upper_quantile=0.67,
            lower_quantile=0.33,
        )
        signals = detector.run(raw_data_view)
        ```

    Note:
        This detector overrides ``preprocess()`` to work directly with raw
        OHLCV data and does not require a FeaturePipeline.
    """

    signal_category: SignalCategory = SignalCategory.VOLATILITY
    allowed_signal_types: set[str] | None = field(default_factory=lambda: {"high_volatility", "low_volatility"})

    price_col: str = "close"
    vol_window: int = 60
    lookback_window: int = 1440
    upper_quantile: float = 0.67
    lower_quantile: float = 0.33

    def preprocess(
        self,
        raw_data_view: RawDataView,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Extract raw OHLCV data without feature pipeline.

        Args:
            raw_data_view: View to raw market data.
            context: Additional context (unused).

        Returns:
            Raw OHLCV data sorted by (pair, timestamp).
        """
        key = self.raw_data_type.value if hasattr(self.raw_data_type, "value") else str(self.raw_data_type)
        return raw_data_view.to_polars(key).sort([self.pair_col, self.ts_col])

    def detect(
        self,
        features: pl.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> Signals:
        """Detect volatility regime from backward-looking realized vol.

        Args:
            features: OHLCV data with pair and timestamp columns.
            context: Additional context (unused).

        Returns:
            Signals with high_volatility/low_volatility signal types.
        """
        price = pl.col(self.price_col)

        # Step 1: log returns (per pair)
        df = features.with_columns(
            (price / price.shift(1).over(self.pair_col)).log().alias("_log_ret"),
        )

        # Step 2: backward realized volatility (rolling std of returns)
        df = df.with_columns(
            pl.col("_log_ret")
            .rolling_std(window_size=self.vol_window, min_samples=max(2, self.vol_window // 4))
            .over(self.pair_col)
            .alias("_realized_vol"),
        )

        # Step 3-5: compute rolling percentile and classify per group
        # Polars doesn't have rolling_quantile with a rank, so we compute via numpy per group
        results = []
        for _pair_name, group in df.group_by(self.pair_col, maintain_order=True):
            vol_arr = group["_realized_vol"].to_numpy().astype(np.float64)
            n = len(vol_arr)

            signal_types = [None] * n
            probabilities = [None] * n

            for t in range(n):
                if np.isnan(vol_arr[t]):
                    continue

                lb_start = max(0, t - self.lookback_window + 1)
                window = vol_arr[lb_start : t + 1]
                valid = window[~np.isnan(window)]
                if len(valid) < 2:
                    continue

                percentile = float(np.mean(valid <= vol_arr[t]))

                if percentile > self.upper_quantile:
                    signal_types[t] = "high_volatility"
                    probabilities[t] = percentile
                elif percentile < self.lower_quantile:
                    signal_types[t] = "low_volatility"
                    probabilities[t] = 1.0 - percentile

            group = group.with_columns(
                [
                    pl.Series(name="signal_type", values=signal_types, dtype=pl.Utf8),
                    pl.Series(name="probability", values=probabilities, dtype=pl.Float64),
                ]
            )
            results.append(group)

        if not results:
            return Signals(pl.DataFrame())

        combined = pl.concat(results, how="vertical_relaxed")

        # Filter to only bars with signals
        signals_df = combined.filter(pl.col("signal_type").is_not_null()).select(
            [
                self.pair_col,
                self.ts_col,
                "signal_type",
                pl.lit(1).alias("signal"),
                "probability",
            ]
        )

        return Signals(signals_df)
