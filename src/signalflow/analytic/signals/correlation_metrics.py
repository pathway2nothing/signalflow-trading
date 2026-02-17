"""Signal correlation and timing analysis metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
from loguru import logger
from plotly.subplots import make_subplots
from scipy import stats

from signalflow.analytic.base import SignalMetric
from signalflow.core import RawData, Signals, sf_component


@dataclass
@sf_component(name="correlation")
class SignalCorrelationMetric(SignalMetric):
    """Analyze correlation between signal strength and actual returns.

    Computes Pearson and Spearman correlations for different look-ahead periods,
    and analyzes returns by signal strength quintiles.
    """

    look_ahead_periods: list[int] = field(default_factory=lambda: [15, 60, 240, 1440])
    strength_col: str = "strength"
    chart_height: int = 900
    chart_width: int = 1400

    def compute(
        self,
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute signal-return correlations."""
        if "spot" in raw_data:
            price_df = raw_data["spot"]
        elif "futures" in raw_data:
            price_df = raw_data["futures"]
        else:
            logger.error("No price data found in raw_data")
            return None, {}

        signals_df = signals.value
        active_signals = signals_df.filter(pl.col("signal") != 0)

        if active_signals.height == 0:
            logger.warning("No non-zero signals found for correlation analysis")
            return None, {}

        correlations = {}
        scatter_data = {}

        for period in self.look_ahead_periods:
            strengths, returns = self._calculate_signal_returns(
                signals_df=active_signals,
                price_df=price_df,
                look_ahead=period,
            )

            if len(strengths) > 2:
                corr, p_value = stats.pearsonr(strengths, returns)
                spearman_corr, spearman_p = stats.spearmanr(strengths, returns)

                correlations[f"period_{period}"] = {
                    "pearson_corr": float(corr),
                    "pearson_p_value": float(p_value),
                    "spearman_corr": float(spearman_corr),
                    "spearman_p_value": float(spearman_p),
                    "n_samples": len(strengths),
                }

                scatter_data[f"period_{period}"] = {
                    "strengths": strengths.tolist(),
                    "returns": returns.tolist(),
                }

        quintile_returns = self._analyze_quintiles(active_signals, price_df)

        computed_metrics = {
            "quant": {
                "correlations": correlations,
                "quintile_analysis": quintile_returns,
                "total_signals": active_signals.height,
            },
            "series": {
                "scatter_data": scatter_data,
            },
        }

        logger.info(
            f"Correlation computed for {active_signals.height} signals across {len(self.look_ahead_periods)} periods"
        )

        return computed_metrics, {"look_ahead_periods": self.look_ahead_periods}

    def _calculate_signal_returns(
        self,
        signals_df: pl.DataFrame,
        price_df: pl.DataFrame,
        look_ahead: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate returns for signals at specified look-ahead."""
        strengths = []
        returns = []

        pairs = signals_df["pair"].unique().to_list()

        for pair in pairs:
            pair_price = price_df.filter(pl.col("pair") == pair).sort("timestamp")
            pair_signals = signals_df.filter(pl.col("pair") == pair)

            if pair_price.height == 0:
                continue

            price_pd = pair_price.to_pandas().set_index("timestamp")

            for signal_row in pair_signals.iter_rows(named=True):
                signal_ts = signal_row["timestamp"]

                if self.strength_col in signal_row and signal_row[self.strength_col] is not None:
                    strength = abs(signal_row[self.strength_col])
                else:
                    strength = abs(signal_row["signal"])

                try:
                    signal_idx = price_pd.index.get_loc(signal_ts)
                except KeyError:
                    continue

                if signal_idx + look_ahead < len(price_pd):
                    signal_price = price_pd.iloc[signal_idx]["close"]
                    future_price = price_pd.iloc[signal_idx + look_ahead]["close"]

                    direction = 1 if signal_row["signal"] > 0 else -1
                    ret = direction * (future_price / signal_price - 1.0)

                    strengths.append(strength)
                    returns.append(ret)

        return np.array(strengths), np.array(returns)

    def _analyze_quintiles(
        self,
        signals_df: pl.DataFrame,
        price_df: pl.DataFrame,
    ) -> dict[str, Any]:
        """Analyze returns by signal strength quintiles."""
        default_period = self.look_ahead_periods[0] if self.look_ahead_periods else 60
        strengths, returns = self._calculate_signal_returns(signals_df, price_df, default_period)

        if len(strengths) < 5:
            return {}

        quintiles = np.percentile(strengths, [20, 40, 60, 80])
        quintile_labels = ["Q1 (Weakest)", "Q2", "Q3", "Q4", "Q5 (Strongest)"]

        quintile_returns = {}
        for i, label in enumerate(quintile_labels):
            if i == 0:
                mask = strengths <= quintiles[0]
            elif i == 4:
                mask = strengths > quintiles[3]
            else:
                mask = (strengths > quintiles[i - 1]) & (strengths <= quintiles[i])

            if mask.sum() > 0:
                q_returns = returns[mask]
                quintile_returns[label] = {
                    "mean_return": float(np.mean(q_returns)),
                    "median_return": float(np.median(q_returns)),
                    "std_return": float(np.std(q_returns)),
                    "win_rate": float(np.mean(q_returns > 0)),
                    "count": int(mask.sum()),
                }

        return quintile_returns

    def plot(
        self,
        computed_metrics: dict[str, Any],
        plots_context: dict[str, Any],
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ) -> go.Figure:
        """Generate correlation visualization."""
        if computed_metrics is None:
            logger.error("No metrics available for plotting")
            return None

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Correlation by Look-ahead Period",
                "Strength vs Return Scatter",
                "Returns by Strength Quintile",
                "Win Rate by Quintile",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        self._add_correlation_bars(fig, computed_metrics)
        self._add_scatter_plot(fig, computed_metrics, plots_context)
        self._add_quintile_returns(fig, computed_metrics)
        self._add_quintile_winrate(fig, computed_metrics)
        self._update_layout(fig)

        return fig

    def _add_correlation_bars(self, fig: go.Figure, metrics: dict[str, Any]):
        """Add correlation bar chart."""
        correlations = metrics["quant"]["correlations"]

        periods = []
        pearson_vals = []
        spearman_vals = []

        for key, vals in correlations.items():
            period = int(key.split("_")[1])
            periods.append(f"{period}min")
            pearson_vals.append(vals["pearson_corr"])
            spearman_vals.append(vals["spearman_corr"])

        fig.add_trace(
            go.Bar(
                x=periods,
                y=pearson_vals,
                name="Pearson",
                marker_color="#2171b5",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=periods,
                y=spearman_vals,
                name="Spearman",
                marker_color="#6baed6",
            ),
            row=1,
            col=1,
        )

    def _add_scatter_plot(self, fig: go.Figure, metrics: dict[str, Any], ctx: dict[str, Any]):
        """Add scatter plot for first period."""
        scatter_data = metrics["series"]["scatter_data"]
        if not scatter_data:
            return

        first_key = next(iter(scatter_data.keys()))
        data = scatter_data[first_key]

        fig.add_trace(
            go.Scatter(
                x=data["strengths"],
                y=[r * 100 for r in data["returns"]],
                mode="markers",
                marker=dict(
                    size=5,
                    color=[r * 100 for r in data["returns"]],
                    colorscale="RdYlGn",
                    cmin=-5,
                    cmax=5,
                    opacity=0.6,
                ),
                name="Signals",
                hovertemplate="Strength: %{x:.3f}<br>Return: %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

    def _add_quintile_returns(self, fig: go.Figure, metrics: dict[str, Any]):
        """Add quintile returns bar chart."""
        quintile_data = metrics["quant"]["quintile_analysis"]
        if not quintile_data:
            return

        labels = list(quintile_data.keys())
        mean_returns = [quintile_data[label]["mean_return"] * 100 for label in labels]
        colors = ["#d73027" if r < 0 else "#1a9850" for r in mean_returns]

        fig.add_trace(
            go.Bar(
                x=labels,
                y=mean_returns,
                marker_color=colors,
                name="Mean Return",
                hovertemplate="Quintile: %{x}<br>Mean Return: %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    def _add_quintile_winrate(self, fig: go.Figure, metrics: dict[str, Any]):
        """Add quintile win rate chart."""
        quintile_data = metrics["quant"]["quintile_analysis"]
        if not quintile_data:
            return

        labels = list(quintile_data.keys())
        win_rates = [quintile_data[label]["win_rate"] * 100 for label in labels]

        fig.add_trace(
            go.Bar(
                x=labels,
                y=win_rates,
                marker_color="#74c476",
                name="Win Rate",
                hovertemplate="Quintile: %{x}<br>Win Rate: %{y:.1f}%<extra></extra>",
            ),
            row=2,
            col=2,
        )

        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=2)

    def _update_layout(self, fig: go.Figure):
        """Update figure layout."""
        fig.update_layout(
            title=dict(
                text="<b>Signal Correlation Analysis</b>",
                font=dict(color="#333333", size=18),
                x=0.5,
                xanchor="center",
            ),
            height=self.chart_height,
            width=self.chart_width,
            template="plotly_white",
            showlegend=True,
            barmode="group",
            paper_bgcolor="#fafafa",
            plot_bgcolor="#ffffff",
        )

        fig.update_yaxes(title_text="Correlation", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=2)
        fig.update_xaxes(title_text="Signal Strength", row=1, col=2)
        fig.update_yaxes(title_text="Mean Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=2)


@dataclass
@sf_component(name="timing")
class SignalTimingMetric(SignalMetric):
    """Analyze optimal holding period for signals.

    Evaluates signal performance at different holding periods to find
    optimal exit timing based on mean return, Sharpe ratio, or win rate.
    """

    max_look_ahead: int = 1440
    sample_points: int = 48
    chart_height: int = 800
    chart_width: int = 1200

    def compute(
        self,
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute optimal timing metrics."""
        if "spot" in raw_data:
            price_df = raw_data["spot"]
        elif "futures" in raw_data:
            price_df = raw_data["futures"]
        else:
            logger.error("No price data found in raw_data")
            return None, {}

        signals_df = signals.value
        active_signals = signals_df.filter(pl.col("signal") != 0)

        if active_signals.height == 0:
            logger.warning("No non-zero signals found for timing analysis")
            return None, {}

        time_points = np.linspace(1, self.max_look_ahead, self.sample_points).astype(int)

        mean_returns = []
        sharpe_at_time = []
        win_rate_at_time = []
        std_returns = []

        for t in time_points:
            returns = self._get_returns_at_time(active_signals, price_df, t)

            if len(returns) > 0:
                mean_ret = np.mean(returns)
                std_ret = np.std(returns)
                sharpe = mean_ret / std_ret if std_ret > 0 else 0
                win_rate = np.mean(returns > 0)

                mean_returns.append(float(mean_ret))
                std_returns.append(float(std_ret))
                sharpe_at_time.append(float(sharpe))
                win_rate_at_time.append(float(win_rate))
            else:
                mean_returns.append(0.0)
                std_returns.append(0.0)
                sharpe_at_time.append(0.0)
                win_rate_at_time.append(0.0)

        optimal_time_mean = int(time_points[np.argmax(mean_returns)])
        optimal_time_sharpe = int(time_points[np.argmax(sharpe_at_time)])
        optimal_time_winrate = int(time_points[np.argmax(win_rate_at_time)])

        computed_metrics = {
            "quant": {
                "optimal_hold_time_mean": optimal_time_mean,
                "optimal_hold_time_sharpe": optimal_time_sharpe,
                "optimal_hold_time_winrate": optimal_time_winrate,
                "peak_mean_return": float(np.max(mean_returns)) * 100,
                "peak_sharpe": float(np.max(sharpe_at_time)),
                "peak_win_rate": float(np.max(win_rate_at_time)) * 100,
                "total_signals": active_signals.height,
            },
            "series": {
                "time_points": time_points.tolist(),
                "mean_returns": [r * 100 for r in mean_returns],
                "std_returns": [r * 100 for r in std_returns],
                "sharpe_at_time": sharpe_at_time,
                "win_rate_at_time": [r * 100 for r in win_rate_at_time],
            },
        }

        logger.info(
            f"Timing analysis: optimal hold time by mean={optimal_time_mean}min, "
            f"by Sharpe={optimal_time_sharpe}min, peak return={np.max(mean_returns) * 100:.2f}%"
        )

        return computed_metrics, {}

    def _get_returns_at_time(
        self,
        signals_df: pl.DataFrame,
        price_df: pl.DataFrame,
        look_ahead: int,
    ) -> np.ndarray:
        """Get returns for all signals at specified look-ahead."""
        returns = []
        pairs = signals_df["pair"].unique().to_list()

        for pair in pairs:
            pair_price = price_df.filter(pl.col("pair") == pair).sort("timestamp")
            pair_signals = signals_df.filter(pl.col("pair") == pair)

            if pair_price.height == 0:
                continue

            price_pd = pair_price.to_pandas().set_index("timestamp")

            for signal_row in pair_signals.iter_rows(named=True):
                signal_ts = signal_row["timestamp"]

                try:
                    signal_idx = price_pd.index.get_loc(signal_ts)
                except KeyError:
                    continue

                if signal_idx + look_ahead < len(price_pd):
                    signal_price = price_pd.iloc[signal_idx]["close"]
                    future_price = price_pd.iloc[signal_idx + look_ahead]["close"]

                    direction = 1 if signal_row["signal"] > 0 else -1
                    ret = direction * (future_price / signal_price - 1.0)
                    returns.append(ret)

        return np.array(returns)

    def plot(
        self,
        computed_metrics: dict[str, Any],
        plots_context: dict[str, Any],
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ) -> go.Figure:
        """Generate timing optimization visualization."""
        if computed_metrics is None:
            logger.error("No metrics available for plotting")
            return None

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "Mean Return Over Time",
                "Sharpe Ratio Over Time",
                "Win Rate Over Time",
            ),
            row_heights=[0.35, 0.35, 0.30],
        )

        series = computed_metrics["series"]
        quant = computed_metrics["quant"]
        time_points = series["time_points"]

        # Mean return with std band
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=series["mean_returns"],
                mode="lines",
                name="Mean Return",
                line=dict(color="#2171b5", width=2),
                fill="tozeroy",
                fillcolor="rgba(33, 113, 181, 0.1)",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[quant["optimal_hold_time_mean"]],
                y=[quant["peak_mean_return"]],
                mode="markers+text",
                name="Optimal (Mean)",
                marker=dict(color="red", size=12, symbol="star"),
                text=[f"{quant['optimal_hold_time_mean']}min"],
                textposition="top center",
            ),
            row=1,
            col=1,
        )

        # Sharpe ratio
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=series["sharpe_at_time"],
                mode="lines",
                name="Sharpe Ratio",
                line=dict(color="#31a354", width=2),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[quant["optimal_hold_time_sharpe"]],
                y=[quant["peak_sharpe"]],
                mode="markers+text",
                name="Optimal (Sharpe)",
                marker=dict(color="red", size=12, symbol="star"),
                text=[f"{quant['optimal_hold_time_sharpe']}min"],
                textposition="top center",
            ),
            row=2,
            col=1,
        )

        # Win rate
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=series["win_rate_at_time"],
                mode="lines",
                name="Win Rate",
                line=dict(color="#756bb1", width=2),
            ),
            row=3,
            col=1,
        )

        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=3, col=1)

        fig.add_trace(
            go.Scatter(
                x=[quant["optimal_hold_time_winrate"]],
                y=[quant["peak_win_rate"]],
                mode="markers+text",
                name="Optimal (Win Rate)",
                marker=dict(color="red", size=12, symbol="star"),
                text=[f"{quant['optimal_hold_time_winrate']}min"],
                textposition="top center",
            ),
            row=3,
            col=1,
        )

        fig.update_layout(
            title=dict(
                text=f"<b>Signal Timing Analysis</b><br>"
                f"<sub>Optimal hold: {quant['optimal_hold_time_mean']}min (mean), "
                f"{quant['optimal_hold_time_sharpe']}min (Sharpe)</sub>",
                font=dict(color="#333333", size=18),
                x=0.5,
                xanchor="center",
            ),
            height=self.chart_height,
            width=self.chart_width,
            template="plotly_white",
            showlegend=True,
            hovermode="x unified",
            paper_bgcolor="#fafafa",
            plot_bgcolor="#ffffff",
        )

        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe", row=2, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=3, col=1)
        fig.update_xaxes(title_text="Hold Time (minutes)", row=3, col=1)

        return fig
