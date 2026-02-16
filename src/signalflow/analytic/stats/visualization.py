"""Visualization functions for statistical validation results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from signalflow.analytic.stats.results import (
        BootstrapResult,
        MonteCarloResult,
        ValidationResult,
    )


def plot_monte_carlo(result: MonteCarloResult) -> list[go.Figure]:
    """Generate Monte Carlo simulation plots.

    Creates three figures:
    1. Final Equity Distribution histogram
    2. Max Drawdown Distribution histogram
    3. Risk Metrics (consecutive losses & drawdown duration)

    Args:
        result: MonteCarloResult to visualize

    Returns:
        List of Plotly Figure objects
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figs = []

    # 1. Final Equity Distribution
    fig1 = go.Figure()
    fig1.add_trace(
        go.Histogram(
            x=result.final_equity_dist,
            nbinsx=50,
            name="Simulated Final Equity",
            marker_color="steelblue",
            opacity=0.7,
        )
    )
    fig1.add_vline(
        x=result.original_final_equity,
        line_dash="dash",
        line_color="red",
        annotation_text="Original",
        annotation_position="top right",
    )
    for pct, val in sorted(result.equity_percentiles.items()):
        fig1.add_vline(
            x=val,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"P{pct * 100:.0f}",
            annotation_position="bottom right",
        )
    fig1.update_layout(
        title="Monte Carlo: Final Equity Distribution",
        xaxis_title="Final Equity ($)",
        yaxis_title="Frequency",
        template="plotly_white",
        showlegend=False,
    )
    figs.append(fig1)

    # 2. Max Drawdown Distribution
    fig2 = go.Figure()
    fig2.add_trace(
        go.Histogram(
            x=result.max_drawdown_dist * 100,
            nbinsx=50,
            name="Simulated Max Drawdown",
            marker_color="indianred",
            opacity=0.7,
        )
    )
    fig2.add_vline(
        x=result.original_max_drawdown * 100,
        line_dash="dash",
        line_color="darkred",
        annotation_text="Original",
        annotation_position="top right",
    )
    fig2.add_vline(
        x=result.ruin_threshold * 100,
        line_dash="solid",
        line_color="black",
        annotation_text=f"Ruin ({result.risk_of_ruin * 100:.1f}% risk)",
        annotation_position="top left",
    )
    fig2.update_layout(
        title="Monte Carlo: Max Drawdown Distribution",
        xaxis_title="Max Drawdown (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        showlegend=False,
    )
    figs.append(fig2)

    # 3. Risk Metrics Panel
    fig3 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Max Consecutive Losses", "Longest Drawdown Duration"),
    )
    fig3.add_trace(
        go.Histogram(
            x=result.max_consecutive_losses_dist,
            name="Consecutive Losses",
            marker_color="orange",
            nbinsx=20,
        ),
        row=1,
        col=1,
    )
    fig3.add_trace(
        go.Histogram(
            x=result.longest_drawdown_duration_dist,
            name="DD Duration",
            marker_color="purple",
            nbinsx=20,
        ),
        row=1,
        col=2,
    )
    fig3.update_layout(
        title="Monte Carlo: Risk Metrics Distribution",
        template="plotly_white",
        showlegend=False,
    )
    fig3.update_xaxes(title_text="Consecutive Losses", row=1, col=1)
    fig3.update_xaxes(title_text="Duration (bars)", row=1, col=2)
    fig3.update_yaxes(title_text="Frequency", row=1, col=1)
    fig3.update_yaxes(title_text="Frequency", row=1, col=2)
    figs.append(fig3)

    return figs


def plot_bootstrap(result: BootstrapResult) -> go.Figure:
    """Generate bootstrap confidence interval plot.

    Creates a forest plot showing point estimates with error bars
    representing confidence intervals for each metric.

    Args:
        result: BootstrapResult to visualize

    Returns:
        Plotly Figure object
    """
    import plotly.graph_objects as go

    metrics = list(result.intervals.keys())
    point_estimates = [result.intervals[m].point_estimate for m in metrics]
    lowers = [result.intervals[m].lower for m in metrics]
    uppers = [result.intervals[m].upper for m in metrics]

    # Determine colors based on significance (excludes 0)
    colors = []
    for m in metrics:
        ci = result.intervals[m]
        if ci.lower > 0 or ci.upper < 0:
            colors.append("green")  # Significant
        else:
            colors.append("gray")  # Not significant

    fig = go.Figure()

    # Error bars showing confidence intervals
    fig.add_trace(
        go.Scatter(
            x=point_estimates,
            y=metrics,
            mode="markers",
            marker=dict(size=12, color=colors),
            name="Point Estimate",
            error_x=dict(
                type="data",
                symmetric=False,
                array=[u - p for u, p in zip(uppers, point_estimates)],
                arrayminus=[p - l for p, l in zip(point_estimates, lowers)],
                color="darkgray",
                thickness=2,
            ),
        )
    )

    # Zero reference line
    fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)

    fig.update_layout(
        title=f"Bootstrap Confidence Intervals ({result.method.upper()}, n={result.n_bootstrap:,})",
        xaxis_title="Value",
        yaxis_title="Metric",
        template="plotly_white",
        height=max(300, len(metrics) * 50),
    )

    return fig


def plot_validation_summary(result: ValidationResult) -> go.Figure:
    """Generate comprehensive validation summary plot.

    Creates a 2x2 subplot with:
    1. Final Equity Distribution (Monte Carlo)
    2. Confidence Intervals (Bootstrap)
    3. Max Drawdown Distribution (Monte Carlo)
    4. Statistical Significance (PSR gauge)

    Args:
        result: ValidationResult to visualize

    Returns:
        Plotly Figure object
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Final Equity Distribution",
            "Confidence Intervals",
            "Max Drawdown Distribution",
            "Statistical Significance",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "indicator"}],
        ],
    )

    # 1. Monte Carlo Equity Distribution
    if result.monte_carlo:
        fig.add_trace(
            go.Histogram(
                x=result.monte_carlo.final_equity_dist,
                nbinsx=30,
                marker_color="steelblue",
                opacity=0.7,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # 2. Bootstrap Confidence Intervals
    if result.bootstrap:
        metrics = list(result.bootstrap.intervals.keys())[:5]  # Top 5
        estimates = [result.bootstrap.intervals[m].point_estimate for m in metrics]
        lowers = [result.bootstrap.intervals[m].lower for m in metrics]
        uppers = [result.bootstrap.intervals[m].upper for m in metrics]

        fig.add_trace(
            go.Scatter(
                x=estimates,
                y=metrics,
                mode="markers",
                marker=dict(size=8, color="steelblue"),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[u - p for u, p in zip(uppers, estimates)],
                    arrayminus=[p - l for p, l in zip(estimates, lowers)],
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # 3. Monte Carlo Drawdown Distribution
    if result.monte_carlo:
        fig.add_trace(
            go.Histogram(
                x=result.monte_carlo.max_drawdown_dist * 100,
                nbinsx=30,
                marker_color="indianred",
                opacity=0.7,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # 4. PSR Gauge
    if result.statistical_tests and result.statistical_tests.psr is not None:
        psr_value = result.statistical_tests.psr * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=psr_value,
                title={"text": "PSR (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "green" if result.statistical_tests.psr_is_significant else "red"},
                    "threshold": {
                        "line": {"color": "black", "width": 2},
                        "thickness": 0.75,
                        "value": result.statistical_tests.confidence_level
                        if hasattr(result.statistical_tests, "confidence_level")
                        else 95,
                    },
                },
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title="Comprehensive Statistical Validation Summary",
        template="plotly_white",
        height=700,
        showlegend=False,
    )

    # Update axis labels
    fig.update_xaxes(title_text="Final Equity ($)", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=2)
    fig.update_xaxes(title_text="Max Drawdown (%)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)

    return fig
