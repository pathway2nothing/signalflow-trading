from signalflow.core import sf_component
from signalflow.analytic.base import SignalMetricsProcessor
from typing import List, Dict, Any
import plotly.graph_objects as go
import polars as pl
from loguru import logger


@dataclass
@sf_component(name="pair")
class SignalPairPrice(SignalMetricsProcessor):
    """Visualize signals overlaid on price chart for specified pairs."""
    
    pairs: List[str] = None  
    buy_marker_color: str = '#00CC96'
    sell_marker_color: str = '#EF553B'
    price_line_color: str = '#2E86C1'
    marker_size: int = 13
    chart_height: int = 600
    
    def __post_init__(self):
        """Convert single pair string to list if needed."""
        if isinstance(self.pairs, str):
            self.pairs = [self.pairs]
    
    def compute(
        self,
        raw_data: sf.RawData,
        signals: sf.Signals,
        labels: pl.DataFrame | None = None,
    ) -> Dict[str, Any]:
        """Compute basic signal statistics per pair.
        
        Returns:
            Dictionary with signal counts per pair
        """
        signals_df = signals.value
        
        # Determine pairs to analyze
        pairs_to_analyze = self.pairs
        if pairs_to_analyze is None:
            pairs_to_analyze = signals_df["pair"].unique().to_list()
        
        metrics = {}
        for pair in pairs_to_analyze:
            pair_signals = signals_df.filter(pl.col("pair") == pair)
            
            buy_count = pair_signals.filter(pl.col("signal") == 1).height
            sell_count = pair_signals.filter(pl.col("signal") == -1).height
            neutral_count = pair_signals.filter(pl.col("signal") == 0).height
            
            metrics[pair] = {
                "buy_signals": buy_count,
                "sell_signals": sell_count,
                "neutral_signals": neutral_count,
                "total_signals": pair_signals.height,
            }
        
        logger.info(f"Computed signal metrics for {len(metrics)} pairs")
        return metrics, {}
    
    def plot(
        self,
        computed_metrics: Dict[str, Any],
        plots_context: Dict[str, Any],
        raw_data: sf.RawData,
        signals: sf.Signals,
        labels: pl.DataFrame | None = None,
    ) -> List[go.Figure]:
        """Generate price charts with signal overlays for each pair.
        
        Returns:
            List of figures, one per pair
        """
        figures = []
        
        if "spot" in raw_data:
            price_df = raw_data["spot"]
        elif "futures" in raw_data:
            price_df = raw_data["futures"]
        else:
            raise ValueError("No price data found in raw_data")
        
        signals_df = signals.value
        
        price_df = price_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        signals_df = signals_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        
        for pair in computed_metrics.keys():
            fig = self._plot_single_pair(
                price_df=price_df,
                signals_df=signals_df,
                pair=pair,
                metrics=computed_metrics[pair]
            )
            figures.append(fig)
        
        logger.info(f"Generated {len(figures)} signal overlay charts")
        return figures
    
    def _plot_single_pair(
        self,
        price_df: pl.DataFrame,
        signals_df: pl.DataFrame,
        pair: str,
        metrics: Dict[str, int]
    ) -> go.Figure:
        """Create signal overlay chart for a single pair."""
        
        price_data = price_df.filter(pl.col("pair") == pair).sort("timestamp")
        pair_signals = signals_df.filter(pl.col("pair") == pair)
        
        signals_with_price = pair_signals.join(
            price_data.select(["timestamp", "pair", "close"]),
            on=["timestamp", "pair"],
            how="inner"
        )
        
        price_pd = price_data.to_pandas()
        signals_pd = signals_with_price.to_pandas()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=price_pd['timestamp'],
            y=price_pd['close'],
            mode='lines',
            name=f'{pair} Price',
            line=dict(color=self.price_line_color, width=1.5),
            hovertemplate='%{x}<br>Price: %{y:.2f}<extra></extra>'
        ))
        
        buys = signals_pd[signals_pd['signal'] == 1]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys['timestamp'],
                y=buys['close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(
                    symbol='circle',
                    size=self.marker_size,
                    color=self.buy_marker_color,
                    line=dict(width=1, color='black')
                ),
                hovertemplate='Buy<br>%{x}<br>Price: %{y:.2f}<extra></extra>'
            ))
        
        # Add sell signals
        sells = signals_pd[signals_pd['signal'] == -1]
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells['timestamp'],
                y=sells['close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(
                    symbol='circle',
                    size=self.marker_size,
                    color=self.sell_marker_color,
                    line=dict(width=1, color='black')
                ),
                hovertemplate='Sell<br>%{x}<br>Price: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text=f'SignalFlow: {pair} Analysis<br>'
                     f'<sub>Buys: {metrics["buy_signals"]} | '
                     f'Sells: {metrics["sell_signals"]} | '
                     f'Total: {metrics["total_signals"]}</sub>',
                font=dict(color='black')
            ),
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            hovermode='x unified',
            height=self.chart_height,
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            showlegend=True
        )
        
        return fig