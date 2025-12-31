from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl

from signalflow.core import Signals
from signalflow.detector.base_detector import SignalDetector


@dataclass
class PandasSignalDetector(SignalDetector):
    """
    Adapter: pandas-based detector logic, polars-first public interface.

    Rule:
      - preprocess() still returns pl.DataFrame (from FeatureSet or overridden)
      - detect(pl.DataFrame) converts to pandas -> detect_pd() -> back to pl -> Signals
    """

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        if not isinstance(features, pl.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.detect expects pl.DataFrame, got {type(features)}")

        pdf = features.to_pandas()
        out_pd = self.detect_pd(pdf, context=context)

        if not isinstance(out_pd, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.detect_pd must return pd.DataFrame, got {type(out_pd)}")

        out_pl = pl.from_pandas(out_pd, include_index=False)
        out_pl = self._normalize_index(out_pl)
        return Signals(out_pl)

    @abstractmethod
    def detect_pd(self, features: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Pandas detection implementation.

        Must return a DataFrame with at least:
          - pair, timestamp, signal_type
        """
        raise NotImplementedError
