"""Real-time price structure detector (local extrema).

Detects local tops and bottoms using backward-looking zigzag with
confirmation delay.
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
@sf_component(name="structure_detector")
class StructureDetector(SignalDetector):
    """Detects local price structure (tops/bottoms) in real-time.

    Unlike ``StructureLabeler``, this detector uses only past data and
    requires a confirmation delay -- a local extremum is only confirmed
    after ``confirmation_bars`` bars have passed showing the reversal.

    Algorithm:
        1. For each bar t, look back ``lookback`` bars
        2. Find the max and min in the lookback window
        3. A LOCAL_TOP is confirmed when:
           - The max occurred at bar (t - confirmation_bars) or earlier
           - Price has dropped >= min_swing_pct from the max
           - Current price < max price
        4. A LOCAL_BOTTOM is confirmed when:
           - The min occurred at bar (t - confirmation_bars) or earlier
           - Price has risen >= min_swing_pct from the min
           - Current price > min price
        5. Only emit the signal once at the confirmation bar

    Attributes:
        price_col (str): Price column name. Default: "close".
        lookback (int): Backward window for extrema search. Default: 60.
        confirmation_bars (int): Bars of reversal needed for confirmation.
            Default: 10.
        min_swing_pct (float): Minimum swing percentage. Default: 0.02.

    Example:
        ```python
        from signalflow.detector.structure_detector import StructureDetector

        detector = StructureDetector(
            lookback=60,
            confirmation_bars=10,
            min_swing_pct=0.02,
        )
        signals = detector.run(raw_data_view)
        ```

    Note:
        This detector overrides ``preprocess()`` to work directly with raw
        OHLCV data and does not require a FeaturePipeline.
    """

    signal_category: SignalCategory = SignalCategory.PRICE_STRUCTURE
    allowed_signal_types: set[str] | None = field(default_factory=lambda: {"local_top", "local_bottom"})

    price_col: str = "close"
    lookback: int = 60
    confirmation_bars: int = 10
    min_swing_pct: float = 0.02

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
        """Detect local tops/bottoms with confirmation delay.

        Args:
            features: OHLCV data with pair and timestamp columns.
            context: Additional context (unused).

        Returns:
            Signals with local_top/local_bottom signal types.
        """
        results = []

        for pair_name, group in features.group_by(self.pair_col, maintain_order=True):
            prices = group[self.price_col].to_numpy().astype(np.float64)
            n = len(prices)

            signal_types = [None] * n
            probabilities = [None] * n

            # Track last emitted extremum to avoid duplicates
            last_emitted_type = None
            last_emitted_idx = -self.lookback

            for t in range(self.lookback + self.confirmation_bars, n):
                p_current = prices[t]
                if np.isnan(p_current):
                    continue

                # Search window: [t - lookback, t - confirmation_bars]
                search_start = t - self.lookback
                search_end = t - self.confirmation_bars + 1

                if search_end <= search_start:
                    continue

                search_window = prices[search_start:search_end]
                valid_mask = ~np.isnan(search_window)
                if not np.any(valid_mask):
                    continue

                valid_prices = search_window[valid_mask]

                max_val = np.max(valid_prices)
                min_val = np.min(valid_prices)

                # Check LOCAL_TOP: max in search window, price dropped since
                if max_val > 0 and p_current < max_val:
                    swing = (max_val - p_current) / max_val
                    if swing >= self.min_swing_pct:
                        if last_emitted_type != "local_top" or (t - last_emitted_idx) > self.lookback:
                            signal_types[t] = "local_top"
                            probabilities[t] = min(1.0, swing / (self.min_swing_pct * 3))
                            last_emitted_type = "local_top"
                            last_emitted_idx = t
                            continue

                # Check LOCAL_BOTTOM: min in search window, price risen since
                if min_val > 0 and p_current > min_val:
                    swing = (p_current - min_val) / min_val
                    if swing >= self.min_swing_pct:
                        if last_emitted_type != "local_bottom" or (t - last_emitted_idx) > self.lookback:
                            signal_types[t] = "local_bottom"
                            probabilities[t] = min(1.0, swing / (self.min_swing_pct * 3))
                            last_emitted_type = "local_bottom"
                            last_emitted_idx = t

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
