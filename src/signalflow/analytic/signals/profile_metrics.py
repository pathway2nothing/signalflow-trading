from typing import Dict, Any, Tuple
from dataclasses import dataclass

import polars as pl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from signalflow.core import sf_component, RawData, Signals
from signalflow.analytic.base import SignalMetric


@dataclass
@sf_component(name="profile")
class SignalProfileMetric(SignalMetric):
    """Analyze post-signal price behavior profiles with statistical aggregations.

    Computes mean, median, percentile profiles of price changes after signals,
    including cumulative max/min statistics for understanding typical signal outcomes.
    """

    look_ahead: int = 1440
    quantiles: Tuple[float, float] = (0.25, 0.75)

    chart_height: int = 900
    chart_width: int = 1400

    def compute(
        self,
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Calculate performance metrics for signals across all pairs."""

        if "spot" in raw_data:
            price_df = raw_data["spot"]
        elif "futures" in raw_data:
            price_df = raw_data["futures"]
        else:
            raise ValueError("No price data found in raw_data")

        signals_df = signals.value

        buy_signals = signals_df.filter(pl.col("signal") == 1)

        if buy_signals.height == 0:
            logger.warning("No buy signals found for profile analysis")
            return None, {}

        post_signal_changes = []
        daily_max_uplifts = []

        pairs = buy_signals["pair"].unique().to_list()
        logger.info(f"Analyzing {buy_signals.height} signals across {len(pairs)} pairs")

        for pair in pairs:
            pair_price = price_df.filter(pl.col("pair") == pair).sort("timestamp")
            pair_signals = buy_signals.filter(pl.col("pair") == pair)

            price_pd = pair_price.to_pandas().set_index("timestamp")

            for signal_row in pair_signals.iter_rows(named=True):
                signal_ts = signal_row["timestamp"]

                try:
                    signal_idx = price_pd.index.get_loc(signal_ts)
                except KeyError:
                    continue

                if signal_idx + self.look_ahead < len(price_pd):
                    signal_price = price_pd.iloc[signal_idx]["close"]
                    future_prices = price_pd["close"].iloc[signal_idx : signal_idx + self.look_ahead + 1].values

                    relative_changes = (future_prices / signal_price) - 1.0
                    post_signal_changes.append(relative_changes)

                    max_uplift = relative_changes.max()
                    daily_max_uplifts.append(max_uplift)

        if not post_signal_changes:
            logger.warning("No valid signal sequences found with sufficient future data")
            return None, {}

        post_signal_df = pd.DataFrame(post_signal_changes)

        mean_profile = post_signal_df.mean()
        std_profile = post_signal_df.std()
        median_profile = post_signal_df.median()
        lower_quant = post_signal_df.quantile(self.quantiles[0])
        upper_quant = post_signal_df.quantile(self.quantiles[1])

        # Compute cumulative max/min profiles
        cummax_df = post_signal_df.cummax(axis=1)
        cummax_mean = cummax_df.mean()
        cummax_median = cummax_df.median()
        cummax_lower = cummax_df.quantile(self.quantiles[0])
        cummax_upper = cummax_df.quantile(self.quantiles[1])

        cummin_df = post_signal_df.cummin(axis=1)
        cummin_mean = cummin_df.mean()
        cummin_median = cummin_df.median()
        cummin_lower = cummin_df.quantile(self.quantiles[0])
        cummin_upper = cummin_df.quantile(self.quantiles[1])

        signal_counts = post_signal_df.count()

        avg_max_uplift = np.mean(daily_max_uplifts) * 100
        median_max_uplift = np.median(daily_max_uplifts) * 100
        max_mean_val = mean_profile.max()
        max_mean_idx = mean_profile.idxmax()
        max_mean_pct = max_mean_val * 100
        final_mean = mean_profile.iloc[-1] * 100
        final_median = median_profile.iloc[-1] * 100
        n_signals = len(post_signal_changes)

        computed_metrics = {
            "quant": {
                "n_signals": n_signals,
                "final_mean": final_mean,
                "final_median": final_median,
                "avg_max_uplift": avg_max_uplift,
                "median_max_uplift": median_max_uplift,
                "max_mean_val": max_mean_val,
                "max_mean_idx": max_mean_idx,
                "max_mean_pct": max_mean_pct,
            },
            "series": {
                "mean_profile": mean_profile,
                "std_profile": std_profile,
                "median_profile": median_profile,
                "lower_quant": lower_quant,
                "upper_quant": upper_quant,
                "cummax_mean": cummax_mean,
                "cummax_median": cummax_median,
                "cummax_lower": cummax_lower,
                "cummax_upper": cummax_upper,
                "cummin_mean": cummin_mean,
                "cummin_median": cummin_median,
                "cummin_lower": cummin_lower,
                "cummin_upper": cummin_upper,
                "signal_counts": signal_counts,
            },
        }

        plots_context = {
            "pairs_analyzed": len(pairs),
            "total_signals": n_signals,
        }

        logger.info(f"Profile computed: {n_signals} signals, final mean: {final_mean:.2f}%, max: {max_mean_pct:.2f}%")

        return computed_metrics, plots_context

    def plot(
        self,
        computed_metrics: Dict[str, Any],
        plots_context: Dict[str, Any],
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ) -> go.Figure:
        """Generate visualization from computed metrics."""

        if computed_metrics is None:
            logger.error("No metrics available for plotting")
            return None

        fig = self._create_figure()

        self._add_mean_profile(fig, computed_metrics)
        self._add_std_bands(fig, computed_metrics)
        self._add_median_profile(fig, computed_metrics)
        self._add_percentile_bands(fig, computed_metrics)
        self._add_key_timepoints(fig, computed_metrics)
        self._add_max_mean_marker(fig, computed_metrics)

        self._add_cummax_profiles(fig, computed_metrics)
        self._add_cummin_profiles(fig, computed_metrics)
        self._add_cummax_percentiles(fig, computed_metrics)

        self._add_summary_annotation(fig, computed_metrics)
        self._add_profit_target_line(fig)
        self._update_layout(fig, computed_metrics, plots_context)

        return fig

    @staticmethod
    def _create_figure():
        """Create subplot structure."""
        return make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                "Average Post-Signal Price Change Profile",
                "Cumulative Maximum Profile",
            ),
        )

    @staticmethod
    def _add_mean_profile(fig, metrics):
        """Add mean profile line."""
        mean = metrics["series"]["mean_profile"]
        fig.add_trace(
            go.Scatter(
                x=mean.index,
                y=mean.values,
                mode="lines",
                name="Mean",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

    @staticmethod
    def _add_std_bands(fig, metrics):
        """Add ±1 STD bands around mean."""
        mean = metrics["series"]["mean_profile"]
        std = metrics["series"]["std_profile"]

        fig.add_trace(
            go.Scatter(
                x=mean.index,
                y=(mean + std).values,
                mode="lines",
                name="+1 STD",
                line=dict(color="lightblue", dash="dash"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=mean.index,
                y=(mean - std).values,
                mode="lines",
                name="-1 STD",
                line=dict(color="lightblue", dash="dash"),
                fill="tonexty",
                fillcolor="rgba(173,216,230,0.1)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    @staticmethod
    def _add_median_profile(fig, metrics):
        """Add median profile line."""
        median = metrics["series"]["median_profile"]
        fig.add_trace(
            go.Scatter(
                x=median.index,
                y=median.values,
                mode="lines",
                name="Median",
                line=dict(color="red", dash="dot"),
            ),
            row=1,
            col=1,
        )

    @staticmethod
    def _add_percentile_bands(fig, metrics):
        """Add 25th-75th percentile bands."""
        lower = metrics["series"]["lower_quant"]
        upper = metrics["series"]["upper_quant"]

        fig.add_trace(
            go.Scatter(
                x=upper.index,
                y=upper.values,
                mode="lines",
                name="75th %ile",
                line=dict(color="green", dash="dash"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=lower.index,
                y=lower.values,
                mode="lines",
                name="25th %ile",
                line=dict(color="green", dash="dash"),
                fill="tonexty",
                fillcolor="rgba(0,128,0,0.1)",
            ),
            row=1,
            col=1,
        )

    @staticmethod
    def _add_key_timepoints(fig, metrics):
        """Add vertical lines at key time intervals."""
        mean = metrics["series"]["mean_profile"]

        key_minutes = [60, 120, 360, 720, 1440]

        for km in key_minutes:
            if km <= len(mean):
                fig.add_vline(
                    x=km,
                    line=dict(color="gray", dash="dot", width=1),
                    row=1,
                    col=1,
                )
                fig.add_annotation(
                    x=km,
                    y=mean.max(),
                    text=f"{km} min",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-30,
                    row=1,
                    col=1,
                )

    @staticmethod
    def _add_max_mean_marker(fig, metrics):
        """Mark the point of maximum mean return."""
        max_val = metrics["quant"]["max_mean_val"]
        max_idx = metrics["quant"]["max_mean_idx"]
        max_pct = metrics["quant"]["max_mean_pct"]

        fig.add_trace(
            go.Scatter(
                x=[max_idx],
                y=[max_val],
                mode="markers+text",
                name="Max Mean",
                text=[f"{max_pct:.2f}%"],
                textposition="top center",
                marker=dict(color="purple", size=10, symbol="star"),
            ),
            row=1,
            col=1,
        )

    @staticmethod
    def _add_cummax_profiles(fig, metrics):
        """Add cumulative maximum mean and median."""
        cummax_mean = metrics["series"]["cummax_mean"]
        cummax_median = metrics["series"]["cummax_median"]

        fig.add_trace(
            go.Scatter(
                x=cummax_mean.index,
                y=cummax_mean.values,
                mode="lines",
                name="CumMax Mean",
                line=dict(color="darkblue", width=2),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=cummax_median.index,
                y=cummax_median.values,
                mode="lines",
                name="CumMax Median",
                line=dict(color="darkred", dash="dot"),
            ),
            row=2,
            col=1,
        )

    @staticmethod
    def _add_cummin_profiles(fig, metrics):
        """Add cumulative minimum mean and median."""
        cummin_mean = metrics["series"]["cummin_mean"]
        cummin_median = metrics["series"]["cummin_median"]

        fig.add_trace(
            go.Scatter(
                x=cummin_mean.index,
                y=cummin_mean.values,
                mode="lines",
                name="CumMin Mean",
                line=dict(color="darkgreen"),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=cummin_median.index,
                y=cummin_median.values,
                mode="lines",
                name="CumMin Median",
                line=dict(color="orange", dash="dot"),
            ),
            row=2,
            col=1,
        )

    @staticmethod
    def _add_cummax_percentiles(fig, metrics):
        """Add cumulative max/min percentile bands."""
        cummax_lower = metrics["series"]["cummax_lower"]
        cummax_upper = metrics["series"]["cummax_upper"]
        cummin_lower = metrics["series"]["cummin_lower"]
        cummin_upper = metrics["series"]["cummin_upper"]

        fig.add_trace(
            go.Scatter(
                x=cummax_upper.index,
                y=cummax_upper.values,
                mode="lines",
                name="CumMax 75th",
                line=dict(color="lightblue", dash="dash"),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=cummax_lower.index,
                y=cummax_lower.values,
                mode="lines",
                name="CumMax 25th",
                line=dict(color="lightblue", dash="dash"),
                fill="tonexty",
                fillcolor="rgba(173,216,230,0.1)",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=cummin_upper.index,
                y=cummin_upper.values,
                mode="lines",
                name="CumMin 75th",
                line=dict(color="lightgreen", dash="dash"),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=cummin_lower.index,
                y=cummin_lower.values,
                mode="lines",
                name="CumMin 25th",
                line=dict(color="lightgreen", dash="dash"),
                fill="tonexty",
                fillcolor="rgba(144,238,144,0.1)",
            ),
            row=2,
            col=1,
        )

    @staticmethod
    def _add_summary_annotation(fig, metrics):
        """Add text box with key statistics."""
        q = metrics["quant"]

        summary_text = (
            f"<b>Profile Statistics</b><br>"
            f"Signals: {q['n_signals']}<br>"
            f"Final Mean: {q['final_mean']:.2f}%<br>"
            f"Final Median: {q['final_median']:.2f}%<br>"
            f"Avg Max Uplift: {q['avg_max_uplift']:.2f}%<br>"
            f"Median Max Uplift: {q['median_max_uplift']:.2f}%<br>"
            f"Peak Mean: {q['max_mean_pct']:.2f}% @ {q['max_mean_idx']} min"
        )

        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=summary_text,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=8,
            bgcolor="white",
            opacity=0.9,
            align="left",
            font=dict(size=11),
            row=1,
            col=1,
        )

    def _add_profit_target_line(self, fig):
        """Add 5% profit target reference line."""
        fig.add_hline(
            y=0.05,
            line=dict(color="red", dash="dot", width=1),
            row=2,
            col=1,
        )

        fig.add_annotation(
            x=self.look_ahead / 2,
            y=0.05,
            text="5% Target",
            showarrow=False,
            yshift=10,
            font=dict(size=10),
            row=2,
            col=1,
        )

    def _update_layout(self, fig, metrics, plots_context):
        """Update figure layout and axes."""
        mean = metrics["series"]["mean_profile"]
        std = metrics["series"]["std_profile"]
        signal_counts = metrics["series"]["signal_counts"]

        y_min = (mean - std).min()
        y_max = (mean + std).max()
        margin = (y_max - y_min) * 0.15

        fig.update_xaxes(title_text="Minutes After Signal", row=2, col=1)
        fig.update_yaxes(
            title_text="Relative Change",
            row=1,
            col=1,
            range=[y_min - margin, y_max + margin],
        )
        fig.update_yaxes(
            title_text="Cumulative Max/Min Change",
            row=2,
            col=1,
            rangemode="tozero",
        )

        pairs_count = plots_context.get("pairs_analyzed", "?")

        fig.update_layout(
            title=dict(
                text=f"SignalFlow: Post-Signal Price Profile Analysis<br>"
                f"<sub>{metrics['quant']['n_signals']} signals across "
                f"{pairs_count} pairs | Look-ahead: {self.look_ahead} min</sub>",
                font=dict(color="black"),
            ),
            template="plotly_white",
            hovermode="x unified",
            height=self.chart_height,
            width=self.chart_width,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
            ),
        )

        customdata = np.array(signal_counts.fillna(0)).reshape(-1, 1)

        row1_trace_names = [
            "Mean",
            "+1 STD",
            "-1 STD",
            "Median",
            "75th %ile",
            "25th %ile",
            "Max Mean",
        ]

        row2_trace_names = [
            "CumMax Mean",
            "CumMax Median",
            "CumMin Mean",
            "CumMin Median",
            "CumMax 75th",
            "CumMax 25th",
            "CumMin 75th",
            "CumMin 25th",
        ]

        for trace in fig.data:
            trace.customdata = customdata

            if trace.name in row1_trace_names:
                trace.hovertemplate = "Minute: %{x}<br>Change: %{y:.4f}<br>Signals: %{customdata[0]:.0f}<extra></extra>"
            elif trace.name in row2_trace_names:
                trace.hovertemplate = (
                    "Minute: %{x}<br>Cum Change: %{y:.4f}<br>Signals: %{customdata[0]:.0f}<extra></extra>"
                )
