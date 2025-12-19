from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.core import RawDataView, Signals, SfComponentType, SignalType, RawDataType
from signalflow.feature import FeatureSet


@dataclass
class SignalDetector(ABC):
    """
    Base Signal Detector.

    Contract:
      1) preprocess(raw_data_view) -> pl.DataFrame
         - computes features used for decision making
         - must return at least [pair_col, ts_col] + feature columns
      2) detect(features) -> Signals
         - makes decisions for all pairs / timestamps contained in features
      3) run(raw_data_view) -> Signals
         - orchestrator: preprocess -> validate -> detect -> validate output

    Notes:
      - Detectors are expected to work in real-time. (Your framework description) :contentReference[oaicite:1]{index=1}
      - Feature computation can be implemented via FeatureSet (recommended).
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.DETECTOR

    pair_col: str = "pair"
    ts_col: str = "timestamp"

    raw_data_type: RawDataType = RawDataType.SPOT

    feature_set: FeatureSet | None = None

    require_probability: bool = False
    keep_only_latest_per_pair: bool = False

    def run(self, raw_data_view: RawDataView, context: dict[str, Any] | None = None) -> Signals:
        """
        Orchestrator:
          preprocess -> validate features -> detect -> validate output -> (optional) postprocess
        """
        feats = self.preprocess(raw_data_view, context=context)
        self._validate_features(feats)

        signals = self.detect(feats, context=context)
        self._validate_signals(signals)

        if self.keep_only_latest_per_pair:
            signals = self._keep_only_latest(signals)

        return signals

    __call__ = run

    def preprocess(self, raw_data_view: RawDataView, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """
        Default implementation: if feature_set is provided -> run it.
        Otherwise fail fast.

        Override in child detectors if:
          - you need custom feature logic
          - you want to join multiple raw tables yourself
        """
        if self.feature_set is None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.preprocess is not implemented and feature_set is None"
            )

        features = self.feature_set.extract(raw_data_view, context=context)
        return features

    @abstractmethod
    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        """
        Must return Signals with at least:
          - pair, timestamp, signal_type
        Recommended:
          - signal (0/1 or signed score)
          - probability (0..1) if you do classification
        """
        raise NotImplementedError

    def _validate_features(self, df: pl.DataFrame) -> None:
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"preprocess must return polars.DataFrame, got {type(df)}")

        missing = [c for c in (self.pair_col, self.ts_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Features missing required columns: {missing}")

        ts_dtype = df.schema.get(self.ts_col)
        if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
            raise ValueError(
                f"Features column '{self.ts_col}' must be timezone-naive, got tz={ts_dtype.time_zone}. "
                f"Use .dt.replace_time_zone(None)."
            )

        dup = (
            df.group_by([self.pair_col, self.ts_col])
            .len()
            .filter(pl.col("len") > 1)
        )
        if dup.height > 0:
            raise ValueError(
                f"Features contain duplicate keys (pair,timestamp). "
                f"Examples:\n{dup.select([self.pair_col, self.ts_col]).head(10)}"
            )

    def _validate_signals(self, signals: Signals) -> None:
        if not isinstance(signals, Signals):
            raise TypeError(f"detect must return Signals, got {type(signals)}")

        s = signals.value
        if not isinstance(s, pl.DataFrame):
            raise TypeError(f"Signals.value must be polars.DataFrame, got {type(s)}")

        required = {self.pair_col, self.ts_col, "signal_type"}
        missing = sorted(required - set(s.columns))
        if missing:
            raise ValueError(f"Signals missing required columns: {missing}")

        allowed = {t.value for t in SignalType}
        bad = (
            s.select(pl.col("signal_type"))
            .unique()
            .filter(~pl.col("signal_type").is_in(list(allowed)))
        )
        if bad.height > 0:
            raise ValueError(f"Signals contain unknown signal_type values: {bad.get_column('signal_type').to_list()}")

        if self.require_probability and "probability" not in s.columns:
            raise ValueError("Signals must contain 'probability' column (require_probability=True)")

        ts_dtype = s.schema.get(self.ts_col)
        if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
            raise ValueError(
                f"Signals column '{self.ts_col}' must be timezone-naive, got tz={ts_dtype.time_zone}."
            )

    def _keep_only_latest(self, signals: Signals) -> Signals:
        s = signals.value
        out = (
            s.sort([self.pair_col, self.ts_col])
            .group_by(self.pair_col, maintain_order=True)
            .tail(1)
            .sort([self.pair_col, self.ts_col])
        )
        return Signals(out)
