from __future__ import annotations

import bisect
import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from loguru import logger
from plotly.subplots import make_subplots
from scipy import stats

from signalflow.analytic.base import StrategyMetric
from signalflow.core import RawData, StrategyState, sf_component


@dataclass
@sf_component(name="result_main", override=True)
class StrategyMainResult(StrategyMetric):
    """Strategy-level visualization based on results['metrics_df'] (Polars DataFrame)."""

    def compute(
        self,
        state: StrategyState,
        prices: dict[str, float],
        **kwargs,
    ) -> dict[str, float]:
        """Compute metric values."""
        return {}

    def plot(
        self,
        results: dict,
        state: StrategyState | None = None,
        raw_data: RawData | None = None,
        **kwargs,
    ) -> list[go.Figure] | go.Figure | None:
        metrics_df: pl.DataFrame | None = results.get("metrics_df")
        if metrics_df is None or metrics_df.height == 0:
            logger.warning("No metrics_df to plot")
            return None

        ts = metrics_df.to_pandas()["timestamp"].to_list()

        main_fig = self._plot_main(metrics_df=metrics_df, ts=ts, results=results)
        detailed_fig = self._plot_detailed(metrics_df=metrics_df, ts=ts, results=results)

        figs: list[go.Figure] = [main_fig]
        if detailed_fig is not None:
            figs.append(detailed_fig)
        return figs

    def _plot_main(self, *, metrics_df: pl.DataFrame, ts: list, results: dict) -> go.Figure:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("Strategy Performance", "Position Metrics", "Balance Allocation"),
            row_heights=[0.45, 0.30, 0.25],
        )

        # Strategy return (%)
        if "total_return" in metrics_df.columns:
            returns_pct = (metrics_df.get_column("total_return") * 100).to_list()
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=returns_pct,
                    mode="lines",
                    name="Strategy Return",
                    hovertemplate="Return: %{y:.2f}%<extra></extra>",
                ),
                row=1,
                col=1,
            )

        fig.add_hline(y=0, line_dash="dash", line_width=1, row=1, col=1)

        # Positions
        if "open_positions" in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=metrics_df.get_column("open_positions").to_list(),
                    mode="lines",
                    name="Open Positions",
                    fill="tozeroy",
                    hovertemplate="Open: %{y}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        if "closed_positions" in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=metrics_df.get_column("closed_positions").to_list(),
                    mode="lines",
                    name="Closed Positions",
                    line=dict(dash="dot"),
                    hovertemplate="Closed: %{y}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # Allocation + Total Return overlay
        if "cash" in metrics_df.columns and "equity" in metrics_df.columns:
            cash = metrics_df.get_column("cash").to_list()
            equity = metrics_df.get_column("equity").to_list()
            initial_capital = results.get("initial_capital", equity[0] if equity else 10000)

            allocated_pct = [(eq - c) / eq if eq > 0 else 0.0 for eq, c in zip(equity, cash, strict=False)]
            free_pct = [c / eq if eq > 0 else 0.0 for eq, c in zip(equity, cash, strict=False)]
            total_balance_pct = [(eq / initial_capital - 1.0) * 100.0 for eq in equity]

            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=free_pct,
                    mode="lines",
                    name="Free Cash",
                    line=dict(width=0),
                    fill="tozeroy",
                    stackgroup="balance",
                    hovertemplate="Free: %{y:.1%}<extra></extra>",
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=allocated_pct,
                    mode="lines",
                    name="In Positions",
                    line=dict(width=0),
                    fill="tonexty",
                    stackgroup="balance",
                    hovertemplate="Allocated: %{y:.1%}<extra></extra>",
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=total_balance_pct,
                    mode="lines",
                    name="Total Return",
                    yaxis="y4",
                    hovertemplate="Total: %{y:.2f}%<extra></extra>",
                ),
                row=3,
                col=1,
            )

            fig.update_yaxes(title_text="Allocation", row=3, col=1, tickformat=".0%", range=[0, 1])
            fig.update_layout(
                yaxis4=dict(
                    title="Total Return (%)",
                    overlaying="y3",
                    side="right",
                    showgrid=False,
                )
            )

        final_return = results.get("final_return", 0.0) * 100.0
        fig.update_layout(
            title=dict(text=f"Backtest Results | Total Return: {final_return:.2f}%"),
            template="plotly_white",
            height=900,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.01, x=0),
            showlegend=True,
        )
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        return fig

    def _plot_detailed(self, *, metrics_df: pl.DataFrame, ts: list, results: dict) -> go.Figure | None:
        has_dd = "current_drawdown" in metrics_df.columns
        has_util = "capital_utilization" in metrics_df.columns
        if not (has_dd or has_util):
            return None

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Drawdown Analysis", "Capital Utilization"),
            row_heights=[0.6, 0.4],
        )

        if has_dd:
            drawdown = metrics_df.get_column("current_drawdown").to_list()
            drawdown_pct = [-d * 100 for d in drawdown]
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=drawdown_pct,
                    mode="lines",
                    name="Drawdown",
                    fill="tozeroy",
                    hovertemplate="DD: %{y:.2f}%<extra></extra>",
                ),
                row=1,
                col=1,
            )

            max_dd = results.get("max_drawdown", 0.0) * 100.0
            if max_dd > 0:
                fig.add_hline(
                    y=-max_dd,
                    line_dash="dash",
                    line_width=1.5,
                    annotation_text=f"Max DD: {max_dd:.2f}%",
                    annotation_position="right",
                    row=1,
                    col=1,
                )

        if has_util:
            util = metrics_df.get_column("capital_utilization").to_list()
            util_pct = [u * 100 for u in util]
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=util_pct,
                    mode="lines",
                    name="Capital Utilization",
                    fill="tozeroy",
                    hovertemplate="Util: %{y:.1f}%<extra></extra>",
                ),
                row=2,
                col=1,
            )
            fig.add_hline(y=100, line_dash="dot", line_width=1, row=2, col=1)

        fig.update_layout(
            template="plotly_white",
            height=600,
            hovermode="x unified",
            showlegend=True,
        )
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=1)
        fig.update_yaxes(title_text="Utilization (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        return fig


@dataclass
@sf_component(name="result_distribution", override=True)
class StrategyDistributionResult(StrategyMetric):
    """Returns distribution and monthly heatmap visualization."""

    def compute(
        self,
        state: StrategyState,
        prices: dict[str, float],
        **kwargs,
    ) -> dict[str, float]:
        return {}

    def plot(
        self,
        results: dict,
        state: StrategyState | None = None,
        raw_data: RawData | None = None,
        **kwargs,
    ) -> list[go.Figure] | go.Figure | None:
        metrics_df: pl.DataFrame | None = results.get("metrics_df")
        if metrics_df is None or metrics_df.height == 0:
            logger.warning("No metrics_df for distribution plot")
            return None

        figs = []

        dist_fig = self._plot_returns_distribution(metrics_df=metrics_df)
        if dist_fig:
            figs.append(dist_fig)

        heatmap_fig = self._plot_returns_heatmap(metrics_df=metrics_df)
        if heatmap_fig:
            figs.append(heatmap_fig)

        return figs if figs else None

    def _plot_returns_distribution(self, *, metrics_df: pl.DataFrame) -> go.Figure | None:
        """Plot returns distribution histogram with normal fit."""
        if "total_return" not in metrics_df.columns:
            return None

        returns = metrics_df.get_column("total_return").to_list()
        if len(returns) < 10:
            return None

        returns_diff = np.diff(returns)

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Returns Distribution", "Returns QQ Plot"),
            horizontal_spacing=0.12,
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns_diff,
                nbinsx=50,
                name="Returns",
                marker_color="#1f77b4",
                opacity=0.7,
                histnorm="probability density",
            ),
            row=1,
            col=1,
        )

        # Normal distribution overlay
        x_range = np.linspace(np.min(returns_diff), np.max(returns_diff), 100)
        normal_y = stats.norm.pdf(x_range, np.mean(returns_diff), np.std(returns_diff))
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_y,
                mode="lines",
                name="Normal Fit",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=1,
        )

        # QQ Plot
        sorted_returns = np.sort(returns_diff)
        n = len(sorted_returns)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))

        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_returns,
                mode="markers",
                name="QQ",
                marker=dict(color="#1f77b4", size=4),
            ),
            row=1,
            col=2,
        )

        # Reference line
        fig.add_trace(
            go.Scatter(
                x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                y=[sorted_returns.min(), sorted_returns.max()],
                mode="lines",
                name="Reference",
                line=dict(color="red", dash="dash"),
            ),
            row=1,
            col=2,
        )

        # Statistics
        skew = float(stats.skew(returns_diff))
        kurtosis = float(stats.kurtosis(returns_diff))
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Skew: {skew:.3f}<br>Kurtosis: {kurtosis:.3f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11),
        )

        fig.update_layout(
            title=dict(text="<b>Returns Distribution Analysis</b>"),
            template="plotly_white",
            height=450,
            showlegend=True,
        )
        fig.update_xaxes(title_text="Return", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

        return fig

    def _plot_returns_heatmap(self, *, metrics_df: pl.DataFrame) -> go.Figure | None:
        """Plot monthly returns heatmap."""
        if "total_return" not in metrics_df.columns or "timestamp" not in metrics_df.columns:
            return None

        df = metrics_df.to_pandas()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month

        # Calculate monthly returns
        monthly = (
            df.groupby(["year", "month"])
            .agg(
                first_return=("total_return", "first"),
                last_return=("total_return", "last"),
            )
            .reset_index()
        )
        monthly["return"] = (monthly["last_return"] - monthly["first_return"]) * 100

        if monthly.empty:
            return None

        # Pivot for heatmap
        pivot = monthly.pivot(index="year", columns="month", values="return")
        pivot = pivot.reindex(columns=range(1, 13))

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        # Create text annotations
        text_matrix = []
        for _, row in pivot.iterrows():
            text_row = []
            for val in row:
                if pd.isna(val):
                    text_row.append("")
                else:
                    text_row.append(f"{val:.1f}%")
            text_matrix.append(text_row)

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=month_names,
                y=pivot.index.astype(str),
                colorscale="RdYlGn",
                zmid=0,
                text=text_matrix,
                texttemplate="%{text}",
                textfont=dict(size=10),
                hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
                colorbar=dict(title="Return (%)"),
            )
        )

        fig.update_layout(
            title=dict(text="<b>Monthly Returns Heatmap</b>"),
            xaxis_title="Month",
            yaxis_title="Year",
            template="plotly_white",
            height=max(300, len(pivot) * 50 + 100),
        )

        return fig


@dataclass
@sf_component(name="result_equity", override=True)
class StrategyEquityResult(StrategyMetric):
    """Equity curve with optional benchmark comparison."""

    benchmark_returns: list[float] | None = None
    benchmark_name: str = "Benchmark"

    def compute(
        self,
        state: StrategyState,
        prices: dict[str, float],
        **kwargs,
    ) -> dict[str, float]:
        return {}

    def plot(
        self,
        results: dict,
        state: StrategyState | None = None,
        raw_data: RawData | None = None,
        **kwargs,
    ) -> go.Figure | None:
        metrics_df: pl.DataFrame | None = results.get("metrics_df")
        if metrics_df is None or metrics_df.height == 0:
            logger.warning("No metrics_df for equity plot")
            return None

        if "equity" not in metrics_df.columns:
            return None

        ts = metrics_df.to_pandas()["timestamp"].to_list()
        equity = metrics_df.get_column("equity").to_list()
        initial = equity[0] if equity else 1
        normalized_equity = [(e / initial - 1) * 100 for e in equity]

        fig = go.Figure()

        # Strategy equity curve
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=normalized_equity,
                mode="lines",
                name="Strategy",
                line=dict(color="#2171b5", width=2),
                fill="tozeroy",
                fillcolor="rgba(33, 113, 181, 0.1)",
                hovertemplate="Return: %{y:.2f}%<extra></extra>",
            )
        )

        # Peak equity line
        peak_equity = np.maximum.accumulate(normalized_equity)
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=peak_equity,
                mode="lines",
                name="Peak",
                line=dict(color="rgba(33, 113, 181, 0.4)", width=1, dash="dash"),
                hovertemplate="Peak: %{y:.2f}%<extra></extra>",
            )
        )

        # Benchmark if provided
        if self.benchmark_returns is not None and len(self.benchmark_returns) > 0:
            benchmark_cumulative = (np.cumprod(1 + np.array(self.benchmark_returns)) - 1) * 100
            benchmark_ts = ts[: len(benchmark_cumulative)]
            fig.add_trace(
                go.Scatter(
                    x=benchmark_ts,
                    y=benchmark_cumulative.tolist(),
                    mode="lines",
                    name=self.benchmark_name,
                    line=dict(color="#d94701", width=2, dash="dot"),
                    hovertemplate=f"{self.benchmark_name}: %{{y:.2f}}%<extra></extra>",
                )
            )

        # Final return annotation
        final_return = normalized_equity[-1] if normalized_equity else 0
        fig.add_annotation(
            x=ts[-1] if ts else 0,
            y=final_return,
            text=f"<b>{final_return:.2f}%</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            ax=40,
            ay=-30,
            font=dict(size=12, color="#2171b5"),
        )

        fig.update_layout(
            title=dict(text=f"<b>Equity Curve</b> | Final Return: {final_return:.2f}%"),
            xaxis_title="Date",
            yaxis_title="Return (%)",
            template="plotly_white",
            height=500,
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", y=1.02, x=0),
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

        return fig


@dataclass
@sf_component(name="result_pair", override=True)
class StrategyPairResult(StrategyMetric):
    """Pair visualization with price line, entry/exit markers, and net position size."""

    pairs: list[str] = field(default_factory=list)
    price_col: str = "close"
    ts_col: str = "timestamp"
    pair_col: str = "pair"

    trade_id_col: str = "id"
    entry_ts_col: str = "entry_ts"
    exit_ts_col: str = "exit_ts"
    size_col: str = "size"

    height: int = 760
    template: str = "plotly_white"
    hovermode: str = "x unified"

    def compute(self, state: StrategyState, prices: dict[str, float], **kwargs) -> dict[str, float]:
        return {}

    def plot(
        self,
        results: dict,
        state: StrategyState | None = None,
        raw_data: RawData | None = None,
        **kwargs,
    ) -> list[go.Figure] | go.Figure | None:
        if not self.pairs:
            logger.warning("StrategyPairResult.plot: pairs is empty")
            return None

        figs: list[go.Figure] = []
        for pair in self.pairs:
            try:
                fig = self._plot_pair(pair=pair, state=state, raw_data=raw_data)
                if fig is not None:
                    figs.append(fig)
            except Exception as e:
                logger.exception(f"StrategyPairResult.plot failed for pair={pair}: {e}")

        return figs

    def _plot_pair(
        self,
        *,
        pair: str,
        state: StrategyState | None,
        raw_data: RawData | None,
    ) -> go.Figure | None:
        df = self._get_pair_df(raw_data=raw_data, pair=pair)
        if df is None or df.height == 0:
            logger.warning(f"StrategyPairResult: no spot data for pair={pair}")
            return None

        if self.ts_col not in df.columns:
            logger.warning(f"StrategyPairResult: ts_col='{self.ts_col}' not found for pair={pair}")
            return None
        if self.price_col not in df.columns:
            logger.warning(f"StrategyPairResult: price_col='{self.price_col}' not found for pair={pair}")
            return None

        ts_dt, ts_s, price = self._normalize_timeseries(df=df)

        if len(ts_s) == 0:
            logger.warning(f"StrategyPairResult: empty time axis for pair={pair}")
            return None

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{pair} Price + Trades", "Net Position Size"),
        )

        fig.add_trace(
            go.Scatter(x=ts_dt, y=price, mode="lines", name="Price"),
            row=1,
            col=1,
        )

        if state is None:
            return self._finalize_fig(fig)

        trades = self._extract_trades(state=state, pair=pair)
        if not trades:
            return self._finalize_fig(fig)

        entry_x: list[dt.datetime] = []
        entry_y: list[float] = []
        entry_cd: list[list[Any]] = []

        exit_x: list[dt.datetime] = []
        exit_y: list[float] = []
        exit_cd: list[list[Any]] = []

        deltas: dict[int, float] = {}

        for tr in trades:
            tid = tr.get(self.trade_id_col)
            et = self._to_int(tr.get(self.entry_ts_col))
            xt = self._to_int(tr.get(self.exit_ts_col))
            size = float(tr.get(self.size_col, 0.0) or 0.0)

            if et is not None:
                p0 = self._nearest_price(epoch_s=et, ts_s=ts_s, price=price)
                x0 = self._epoch_to_dt(et)
                if p0 is not None:
                    entry_x.append(x0)
                    entry_y.append(p0)
                    entry_cd.append([tid, size, dt.datetime.utcfromtimestamp(et).isoformat()])

                deltas[et] = deltas.get(et, 0.0) + size

            if xt is not None:
                p1 = self._nearest_price(epoch_s=xt, ts_s=ts_s, price=price)
                x1 = self._epoch_to_dt(xt)
                if p1 is not None:
                    exit_x.append(x1)
                    exit_y.append(p1)
                    exit_cd.append([tid, size, dt.datetime.utcfromtimestamp(xt).isoformat()])

                deltas[xt] = deltas.get(xt, 0.0) - size

            if et is not None and xt is not None:
                p0 = self._nearest_price(epoch_s=et, ts_s=ts_s, price=price)
                p1 = self._nearest_price(epoch_s=xt, ts_s=ts_s, price=price)
                if p0 is not None and p1 is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[self._epoch_to_dt(et), self._epoch_to_dt(xt)],
                            y=[p0, p1],
                            mode="lines",
                            showlegend=False,
                            opacity=0.35,
                            customdata=[[tid, size, et, xt]],
                            hovertemplate=(
                                "id=%{customdata[0]}<br>"
                                "size=%{customdata[1]}<br>"
                                "entry=%{customdata[2]}<br>"
                                "exit=%{customdata[3]}<br>"
                                "<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=1,
                    )

        if entry_x:
            fig.add_trace(
                go.Scatter(
                    x=entry_x,
                    y=entry_y,
                    mode="markers",
                    name="Entry",
                    marker=dict(size=10, symbol="triangle-up", color="green"),
                    customdata=entry_cd,
                    hovertemplate=(
                        "Entry<br>"
                        "id=%{customdata[0]}<br>"
                        "size=%{customdata[1]}<br>"
                        "ts=%{customdata[2]}<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

        if exit_x:
            fig.add_trace(
                go.Scatter(
                    x=exit_x,
                    y=exit_y,
                    mode="markers",
                    name="Exit",
                    marker=dict(size=10, symbol="triangle-down", color="red"),
                    customdata=exit_cd,
                    hovertemplate=(
                        "Exit<br>id=%{customdata[0]}<br>size=%{customdata[1]}<br>ts=%{customdata[2]}<br><extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

        if deltas:
            ev_ts = sorted(deltas.keys())
            cum = 0.0
            size_x: list[dt.datetime] = []
            size_y: list[float] = []
            for t in ev_ts:
                cum += deltas[t]
                size_x.append(self._epoch_to_dt(t))
                size_y.append(cum)

            fig.add_trace(
                go.Scatter(
                    x=size_x,
                    y=size_y,
                    mode="lines",
                    name="Net Position Size",
                    line_shape="hv",
                    hovertemplate="ts=%{x}<br>net=%{y}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        return self._finalize_fig(fig)

    def _get_pair_df(self, *, raw_data: RawData | None, pair: str) -> pl.DataFrame | None:
        if raw_data is None:
            return None
        spot = raw_data.get("spot")
        if not isinstance(spot, pl.DataFrame):
            return None
        if self.pair_col not in spot.columns:
            return spot
        return spot.filter(pl.col(self.pair_col) == pair)

    def _normalize_timeseries(self, *, df: pl.DataFrame) -> tuple[list[dt.datetime], list[int], list[float]]:
        """Returns ts_dt (datetime), ts_s (epoch seconds), price (float) - all sorted."""
        ts = df.get_column(self.ts_col)
        px = df.get_column(self.price_col).cast(pl.Float64)

        if ts.dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
            max_v = ts.max()
            ms = bool(max_v is not None and int(max_v) > 10_000_000_000)
            ts_s = (ts.cast(pl.Int64) // (1000 if ms else 1)).alias("ts_s")
        elif ts.dtype == pl.Datetime:
            ts_s = ts.dt.epoch(time_unit="s").alias("ts_s")
        else:
            ts_dt = ts.cast(pl.Utf8).str.to_datetime(strict=False)
            ts_s = ts_dt.dt.epoch(time_unit="s").alias("ts_s")

        tmp = pl.DataFrame({"ts_s": ts_s, "price": px}).drop_nulls().sort("ts_s")

        ts_s_list = tmp["ts_s"].cast(pl.Int64).to_list()
        price_list = tmp["price"].to_list()
        ts_dt_list = [self._epoch_to_dt(int(t)) for t in ts_s_list]

        return ts_dt_list, [int(t) for t in ts_s_list], [float(p) for p in price_list]

    def _extract_trades(self, *, state: StrategyState, pair: str) -> list[dict[str, Any]]:
        """Extract trades with unique entry ids from strategy state."""
        out: list[dict[str, Any]] = []

        positions = getattr(getattr(state, "portfolio", None), "positions", None)
        if not isinstance(positions, dict):
            return out

        for p in positions.values():
            if getattr(p, "pair", None) != pair:
                continue

            pid = getattr(p, "id", None)

            entry_time = getattr(p, "entry_time", None)
            exit_time = getattr(p, "last_time", None) if getattr(p, "is_closed", False) else None

            entry_ts = self._to_epoch(entry_time)
            exit_ts = self._to_epoch(exit_time)

            size = getattr(p, "quantity", None)
            if size is None:
                size = getattr(p, "size", None)
            if size is None:
                size = 0.0

            out.append(
                {
                    self.trade_id_col: str(pid) if pid is not None else None,
                    self.entry_ts_col: entry_ts,
                    self.exit_ts_col: exit_ts,
                    self.size_col: float(size or 0.0),
                }
            )

        return out

    def _nearest_price(self, *, epoch_s: int, ts_s: list[int], price: list[float]) -> float | None:
        if epoch_s is None or not ts_s:
            return None
        i = bisect.bisect_left(ts_s, int(epoch_s))
        if i <= 0:
            return float(price[0])
        if i >= len(ts_s):
            return float(price[-1])
        lt, rt = ts_s[i - 1], ts_s[i]
        lp, rp = price[i - 1], price[i]
        return float(lp if abs(epoch_s - lt) <= abs(rt - epoch_s) else rp)

    def _epoch_to_dt(self, t: int) -> dt.datetime:
        return dt.datetime.utcfromtimestamp(int(t))

    def _to_epoch(self, t: Any) -> int | None:
        if t is None:
            return None
        if hasattr(t, "timestamp"):
            return int(t.timestamp())
        if isinstance(t, int):
            return int(t)
        return None

    def _to_int(self, x: Any) -> int | None:
        if x is None:
            return None
        try:
            return int(x)
        except Exception:
            return None

    def _finalize_fig(self, fig: go.Figure) -> go.Figure:
        fig.update_layout(
            template=self.template,
            height=self.height,
            hovermode=self.hovermode,
            showlegend=True,
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Net Size", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        return fig
