from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar
from signalflow.core import SfComponentType, RawData, Signals, StrategyState
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple
import polars as pl
from loguru import logger


@dataclass
class SignalMetric:
    """Base class for signal metrics computation and visualization."""
    
    component_type = SfComponentType.SIGNAL_METRIC

    def compute(
        self,
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]: 
        """Compute metrics from signals.
        
        Returns:
            Dictionary with computed metrics
        """
        logger.warning("Computing is not implemented for this component")
        return {}, {}

    def plot(
        self,
        computed_metrics: Dict[str, Any],
        plots_context: Dict[str, Any],
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ) -> List[go.Figure] | go.Figure | None:
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
    
    def compute(
        self,
        state: StrategyState,
        prices: dict[str, float], 
        **kwargs
    ) -> Dict[str, float]:
        """Compute metric values."""
        logger.warning("Computing is not implemented for this component")
        return {}

    def plot(
        self,
        results: dict, 
        state: StrategyState|None=None,
        raw_data: RawData|None=None,
        **kwargs
    ) -> list[go.Figure] | go.Figure | None:
        """Plot metric values."""
        logger.warning("Plotting is not implemented for this component")
        return None
