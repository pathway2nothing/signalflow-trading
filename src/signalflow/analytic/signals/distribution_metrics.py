from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
from loguru import logger
from plotly.subplots import make_subplots

from signalflow.analytic.base import SignalMetric
from signalflow.core import RawData, Signals, sf_component


@dataclass
@sf_component(name="distribution")
class SignalDistributionMetric(SignalMetric):
    """Analyze signal distribution across pairs and time."""

    n_bars: int = 10
    rolling_window_minutes: int = 60
    ma_window_hours: int = 12
    chart_height: int = 1200
    chart_width: int = 1400

    def compute(
        self,
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute signal distribution metrics."""

        signals_df = signals.value

        signals_per_pair = (
            signals_df.filter(pl.col("signal") != 0)
            .group_by("pair")
            .agg(pl.count().alias("signal_count"))
            .sort("signal_count", descending=True)
        )

        if signals_per_pair.height == 0:
            logger.warning("No non-zero signals found")
            return None, {}

        signal_counts = signals_per_pair["signal_count"].to_numpy()
        min_count = int(signal_counts.min())
        max_count = int(signal_counts.max())
        mean_count = signal_counts.mean()
        median_count = np.median(signal_counts)
        n_pairs = len(signal_counts)

        if n_pairs <= 15:
            grouped_data = []
            for row in signals_per_pair.iter_rows(named=True):
                grouped_data.append(
                    {
                        "category": row["pair"],
                        "num_columns": row["signal_count"],
                        "columns_in_group": row["pair"],
                    }
                )
            bin_labels = [g["category"] for g in grouped_data]
            use_histogram = False
        else:
            actual_n_bars = min(self.n_bars, max(3, n_pairs // 5))

            if min_count == max_count:
                bin_edges = [min_count - 0.5, max_count + 0.5]
                bin_labels = [f"{min_count}"]
            else:
                bin_edges = np.linspace(min_count, max_count, actual_n_bars + 1)
                bin_labels = []
                for i in range(actual_n_bars):
                    lower = int(np.floor(bin_edges[i]))
                    upper = int(np.ceil(bin_edges[i + 1]))
                    label = f"{lower}" if lower == upper else f"{lower}-{upper}"
                    bin_labels.append(label)

            binned = np.digitize(signal_counts, bin_edges[:-1]) - 1
            binned = np.clip(binned, 0, len(bin_labels) - 1)

            grouped_data = []
            for i, label in enumerate(bin_labels):
                mask = binned == i
                pairs_in_bin = signals_per_pair.filter(pl.Series(mask))["pair"].to_list()

                if pairs_in_bin:
                    grouped_data.append(
                        {
                            "category": label,
                            "num_columns": len(pairs_in_bin),
                            "columns_in_group": "<br>".join(pairs_in_bin),
                        }
                    )
            use_histogram = True

        signals_by_time = (
            signals_df.filter(pl.col("signal") != 0)
            .sort("timestamp")
            .group_by_dynamic("timestamp", every="1m")
            .agg(pl.count().alias("signal_count"))
            .sort("timestamp")
        )

        signals_rolling = signals_by_time.with_columns(
            pl.col("signal_count")
            .rolling_sum(
                window_size=self.rolling_window_minutes,
                min_periods=1,
                center=False,
            )
            .alias("rolling_sum")
        )

        ma_window_minutes = self.ma_window_hours * 60
        if signals_rolling.height > ma_window_minutes:
            signals_rolling = signals_rolling.with_columns(
                pl.col("rolling_sum")
                .rolling_mean(
                    window_size=ma_window_minutes,
                    min_periods=1,
                    center=True,
                )
                .alias("ma")
            )
        else:
            signals_rolling = signals_rolling.with_columns(pl.lit(None).alias("ma"))

        mean_rolling = signals_rolling["rolling_sum"].mean()
        max_rolling = signals_rolling["rolling_sum"].max()

        computed_metrics = {
            "quant": {
                "mean_signals_per_pair": float(mean_count),
                "median_signals_per_pair": float(median_count),
                "min_signals_per_pair": min_count,
                "max_signals_per_pair": max_count,
                "total_pairs": n_pairs,
                "mean_rolling_signals": float(mean_rolling) if mean_rolling else 0.0,
                "max_rolling_signals": int(max_rolling) if max_rolling else 0,
            },
            "series": {
                "grouped": grouped_data,
                "signals_per_pair": signals_per_pair,
                "signals_rolling": signals_rolling,
            },
        }

        plots_context = {
            "bin_labels": bin_labels,
            "rolling_window": self.rolling_window_minutes,
            "ma_window": self.ma_window_hours,
            "use_histogram": use_histogram,
        }

        logger.info(
            f"Distribution computed: {n_pairs} pairs, "
            f"mean {mean_count:.1f} signals/pair, "
            f"max rolling {max_rolling} signals/{self.rolling_window_minutes}min"
        )

        return computed_metrics, plots_context

    def plot(
        self,
        computed_metrics: dict[str, Any],
        plots_context: dict[str, Any],
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ) -> go.Figure:
        """Generate distribution visualization."""

        if computed_metrics is None:
            logger.error("No metrics available for plotting")
            return None

        fig = self._create_figure(plots_context)

        self._add_histogram(fig, computed_metrics, plots_context)
        self._add_sorted_signals(fig, computed_metrics)
        self._add_rolling_signals(fig, computed_metrics, plots_context)
        self._update_layout(fig, plots_context)

        return fig

    @staticmethod
    def _create_figure(plots_context):
        """Create subplot structure."""
        use_histogram = plots_context.get("use_histogram", True)

        title1 = "Pairs Distribution by Signal Count" if use_histogram else "Signal Count per Pair"

        return make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=False,
            row_heights=[0.3, 0.35, 0.35],
            vertical_spacing=0.1,
            subplot_titles=[
                title1,
                "Signal Count per Pair (Ranked)",
                "Temporal Signal Density",
            ],
        )

    @staticmethod
    def _add_histogram(fig, metrics, plots_context):
        """Add histogram/bar chart of signal distribution."""
        grouped = metrics["series"]["grouped"]
        use_histogram = plots_context.get("use_histogram", True)

        if not grouped:
            return

        categories = [g["category"] for g in grouped]

        if use_histogram:
            counts = [g["num_columns"] for g in grouped]
            hovertexts = [g["columns_in_group"] for g in grouped]
            y_title = "Number of Pairs"
            hovertemplate = (
                "<b>Signal Range:</b> %{x}<br>"
                "<b>Number of Pairs:</b> %{y}<br>"
                "<b>Pairs:</b><br>%{customdata[0]}"
                "<extra></extra>"
            )
            customdata = [[ht] for ht in hovertexts]
        else:
            counts = [g["num_columns"] for g in grouped]
            y_title = "Signal Count"
            hovertemplate = "<b>Pair:</b> %{x}<br><b>Signals:</b> %{y}<extra></extra>"
            customdata = None

        max_val = max(counts) if counts else 1
        colors = [f"rgba(33, 113, 181, {0.4 + 0.6 * (c / max_val)})" for c in counts]

        fig.add_trace(
            go.Bar(
                x=categories,
                y=counts,
                customdata=customdata,
                marker=dict(
                    color=colors,
                    line=dict(color="#084594", width=1),
                ),
                hovertemplate=hovertemplate,
                name="Distribution",
            ),
            row=1,
            col=1,
        )

        fig.update_yaxes(
            title_text=y_title,
            dtick=1 if max(counts) <= 10 else None,
            row=1,
            col=1,
        )

    @staticmethod
    def _add_sorted_signals(fig, metrics):
        """Add sorted signal counts per pair."""
        signals_per_pair = metrics["series"]["signals_per_pair"]

        pairs = signals_per_pair["pair"].to_list()
        counts = signals_per_pair["signal_count"].to_list()
        n_pairs = len(pairs)

        ranks = list(range(1, n_pairs + 1))

        fig.add_trace(
            go.Scatter(
                x=ranks,
                y=counts,
                mode="lines+markers",
                line=dict(color="#e6550d", width=2),
                marker=dict(size=8 if n_pairs <= 20 else 5, color="#a63603"),
                text=pairs,
                hovertemplate=("<b>Rank:</b> %{x}<br><b>Pair:</b> %{text}<br><b>Signals:</b> %{y}<extra></extra>"),
                name="Signal Count",
            ),
            row=2,
            col=1,
        )

        mean_count = metrics["quant"]["mean_signals_per_pair"]
        fig.add_hline(
            y=mean_count,
            line=dict(color="#31a354", dash="dash", width=2),
            annotation_text=f"Mean: {mean_count:.1f}",
            annotation_position="right",
            annotation_font_color="#31a354",
            row=2,
            col=1,
        )

        fig.update_xaxes(
            title_text="Pair Rank",
            dtick=1 if n_pairs <= 20 else None,
            range=[0.5, n_pairs + 0.5],
            row=2,
            col=1,
        )

    @staticmethod
    def _add_rolling_signals(fig, metrics, plots_context):
        """Add rolling signal count over time."""
        signals_rolling = metrics["series"]["signals_rolling"]

        if signals_rolling.height == 0:
            return

        timestamps = signals_rolling["timestamp"].to_list()
        rolling_sum = signals_rolling["rolling_sum"].to_list()

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=rolling_sum,
                mode="lines",
                line=dict(color="#6baed6", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(107, 174, 214, 0.2)",
                hovertemplate=(
                    "<b>Time:</b> %{x}<br>"
                    f"<b>{plots_context['rolling_window']}min Signals:</b> %{{y:.0f}}"
                    "<extra></extra>"
                ),
                name=f"{plots_context['rolling_window']}min Rolling",
            ),
            row=3,
            col=1,
        )

        # Moving average
        if "ma" in signals_rolling.columns:
            ma_values = signals_rolling["ma"].to_list()
            if any(v is not None for v in ma_values):
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=ma_values,
                        mode="lines",
                        line=dict(color="#08519c", width=2.5),
                        hovertemplate=(
                            f"<b>Time:</b> %{{x}}<br><b>{plots_context['ma_window']}h MA:</b> %{{y:.1f}}<extra></extra>"
                        ),
                        name=f"{plots_context['ma_window']}h MA",
                    ),
                    row=3,
                    col=1,
                )

    def _update_layout(self, fig, plots_context):
        """Update figure layout and axes."""
        fig.update_yaxes(
            title_text="Signal Count",
            row=2,
            col=1,
        )

        fig.update_xaxes(
            title_text="Time",
            row=3,
            col=1,
        )
        fig.update_yaxes(
            title_text=f"Signals ({plots_context['rolling_window']}min)",
            row=3,
            col=1,
        )

        fig.update_layout(
            title=dict(
                text="<b>Signal Distribution Analysis</b>",
                font=dict(color="#333333", size=18),
                x=0.5,
                xanchor="center",
            ),
            height=self.chart_height,
            width=self.chart_width,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.04,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.8)",
            ),
            paper_bgcolor="#fafafa",
            plot_bgcolor="#ffffff",
        )

        fig.update_xaxes(gridcolor="#e8e8e8", zeroline=False)
        fig.update_yaxes(gridcolor="#e8e8e8", zeroline=False)
