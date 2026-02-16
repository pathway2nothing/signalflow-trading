from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar

import plotly.graph_objects as go
import polars as pl
from loguru import logger

from signalflow.core import RawData, SfComponentType, Signals, StrategyState


@dataclass
class SignalMetric:
    """Base class for signal metrics computation and visualization."""

    component_type = SfComponentType.SIGNAL_METRIC

    def compute(
        self,
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute metrics from signals.

        Returns:
            Dictionary with computed metrics
        """
        logger.warning("Computing is not implemented for this component")
        return {}, {}

    def plot(
        self,
        computed_metrics: dict[str, Any],
        plots_context: dict[str, Any],
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ) -> list[go.Figure] | go.Figure | None:
        """Generate visualization from computed metrics.

        Returns:
            Single figure or list of figures
        """
        logger.warning("Plotting is not implemented for this component")
        return None

    def __call__(
        self,
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ):
        computed_metrics, plots_context = self.compute(
            raw_data=raw_data,
            signals=signals,
            labels=labels,
        )
        metric_plots = self.plot(
            computed_metrics=computed_metrics,
            plots_context=plots_context,
            raw_data=raw_data,
            signals=signals,
            labels=labels,
        )

        return computed_metrics, metric_plots


@dataclass
class StrategyMetric(ABC):
    """Base class for strategy metrics."""

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_METRIC

    def compute(self, state: StrategyState, prices: dict[str, float], **kwargs) -> dict[str, float]:
        """Compute metric values."""
        logger.warning("Computing is not implemented for this component")
        return {}

    def plot(
        self, results: dict, state: StrategyState | None = None, raw_data: RawData | None = None, **kwargs
    ) -> list[go.Figure] | go.Figure | None:
        """Plot metric values."""
        logger.warning("Plotting is not implemented for this component")
        return None
