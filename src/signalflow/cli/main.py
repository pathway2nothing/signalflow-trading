"""
SignalFlow CLI - Main entry point.

Commands:
    sf run config.yaml        Run backtest from YAML config
    sf list detectors         List available detectors
    sf list metrics           List available metrics
    sf validate config.yaml   Validate config without running
    sf init                   Create sample config file
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from signalflow.api.result import BacktestResult


# =============================================================================
# CLI Group
# =============================================================================


@click.group()
@click.version_option(package_name="signalflow-trading")
def cli():
    """SignalFlow - Trading signal framework CLI."""
    pass


# =============================================================================
# Run Command
# =============================================================================


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Save results to JSON file")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
@click.option("--plot", "-p", is_flag=True, help="Show plots after backtest")
def run(config_path: str, output: str | None, quiet: bool, plot: bool):
    """
    Run backtest from YAML configuration file.

    \b
    Example:
        sf run config.yaml
        sf run config.yaml --output results.json
        sf run config.yaml --quiet --plot
    """
    from signalflow.cli.config import BacktestConfig

    # Load and validate config
    click.echo(f"Loading config: {config_path}")
    try:
        config = BacktestConfig.from_yaml(config_path)
    except Exception as e:
        click.secho(f"Error loading config: {e}", fg="red", err=True)
        sys.exit(1)

    # Validate
    issues = config.validate()
    errors = [i for i in issues if i.startswith("ERROR")]
    warnings = [i for i in issues if i.startswith("WARNING")]

    for warning in warnings:
        click.secho(warning, fg="yellow")

    if errors:
        for error in errors:
            click.secho(error, fg="red", err=True)
        sys.exit(1)

    # Run backtest
    click.echo(f"Running backtest: {config.strategy_id}")
    try:
        if quiet:
            config.show_progress = False

        builder = config.to_builder()
        result = builder.run()

        # Display results
        _display_result(result)

        # Save to file if requested
        if output:
            _save_result(result, output)
            click.echo(f"Results saved to: {output}")

        # Show plots if requested
        if plot:
            _show_plots(result)

    except Exception as e:
        click.secho(f"Error during backtest: {e}", fg="red", err=True)
        sys.exit(1)


def _display_result(result: BacktestResult) -> None:
    """Display backtest result summary."""
    click.echo()
    click.echo("=" * 50)
    click.secho("         BACKTEST RESULT", bold=True)
    click.echo("=" * 50)

    # Return with color
    return_color = "green" if result.total_return >= 0 else "red"
    click.echo(f"  Total Return:    ", nl=False)
    click.secho(f"{result.total_return:+.2%}", fg=return_color, bold=True)

    click.echo("-" * 50)
    click.echo(f"  Trades:          {result.n_trades:>10}")
    click.echo(f"  Win Rate:        {result.win_rate:>10.1%}")
    click.echo(f"  Profit Factor:   {result.profit_factor:>10.2f}")
    click.echo("-" * 50)
    click.echo(f"  Initial Capital: ${result.initial_capital:>12,.2f}")
    click.echo(f"  Final Capital:   ${result.final_capital:>12,.2f}")

    # Additional metrics
    m = result.metrics
    if m.get("max_drawdown"):
        click.echo(f"  Max Drawdown:    {m['max_drawdown']:>10.1%}")
    if m.get("sharpe_ratio"):
        click.echo(f"  Sharpe Ratio:    {m['sharpe_ratio']:>10.2f}")

    click.echo("=" * 50)
    click.echo()


def _save_result(result: BacktestResult, path: str) -> None:
    """Save result to JSON file."""
    import json
    from datetime import datetime

    data = result.to_dict()

    # Convert datetime objects for JSON
    def convert(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=convert)


def _show_plots(result: BacktestResult) -> None:
    """Show result plots."""
    figs = result.plot()
    if figs:
        for fig in figs:
            fig.show()
    else:
        click.echo("No plots available")


# =============================================================================
# List Commands
# =============================================================================


@cli.group()
def list():
    """List available components from registry."""
    pass


@list.command("detectors")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed info")
def list_detectors(verbose: bool):
    """List available signal detectors."""
    from signalflow.core import default_registry, SfComponentType

    detectors = default_registry.list(SfComponentType.DETECTOR)

    if not detectors:
        click.echo("No detectors registered.")
        click.echo("Install signalflow-ta for built-in detectors:")
        click.secho("  pip install signalflow-ta", fg="cyan")
        return

    click.echo(f"Available detectors ({len(detectors)}):")
    click.echo("-" * 40)

    for name in sorted(detectors):
        click.secho(f"  {name}", fg="green")
        if verbose:
            try:
                cls = default_registry.get(SfComponentType.DETECTOR, name)
                if cls.__doc__:
                    doc = cls.__doc__.split("\n")[0].strip()
                    click.echo(f"    {doc}")
            except Exception:
                pass


@list.command("metrics")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed info")
def list_metrics(verbose: bool):
    """List available strategy metrics."""
    from signalflow.core import default_registry, SfComponentType

    metrics = default_registry.list(SfComponentType.STRATEGY_METRIC)

    if not metrics:
        click.echo("No metrics registered.")
        return

    click.echo(f"Available metrics ({len(metrics)}):")
    click.echo("-" * 40)

    for name in sorted(metrics):
        click.secho(f"  {name}", fg="green")
        if verbose:
            try:
                cls = default_registry.get(SfComponentType.STRATEGY_METRIC, name)
                if cls.__doc__:
                    doc = cls.__doc__.split("\n")[0].strip()
                    click.echo(f"    {doc}")
            except Exception:
                pass


@list.command("features")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed info")
def list_features(verbose: bool):
    """List available features."""
    from signalflow.core import default_registry, SfComponentType

    features = default_registry.list(SfComponentType.FEATURE)

    if not features:
        click.echo("No features registered.")
        return

    click.echo(f"Available features ({len(features)}):")
    click.echo("-" * 40)

    for name in sorted(features):
        click.secho(f"  {name}", fg="green")


@list.command("all")
def list_all():
    """List all registered components."""
    from signalflow.core import default_registry, SfComponentType

    click.echo("SignalFlow Registry")
    click.echo("=" * 50)

    for comp_type in SfComponentType:
        components = default_registry.list(comp_type)
        if components:
            type_name = comp_type.name.replace("_", " ").title()
            click.echo(f"\n{type_name} ({len(components)}):")
            click.echo("-" * 40)
            for name in sorted(components):
                click.echo(f"  {name}")


# =============================================================================
# Validate Command
# =============================================================================


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path: str):
    """
    Validate configuration file without running.

    \b
    Example:
        sf validate config.yaml
    """
    from signalflow.cli.config import BacktestConfig

    click.echo(f"Validating: {config_path}")

    try:
        config = BacktestConfig.from_yaml(config_path)
    except Exception as e:
        click.secho(f"YAML parse error: {e}", fg="red", err=True)
        sys.exit(1)

    issues = config.validate()

    if not issues:
        click.secho("Config is valid!", fg="green")
        sys.exit(0)

    errors = [i for i in issues if i.startswith("ERROR")]
    warnings = [i for i in issues if i.startswith("WARNING")]

    for warning in warnings:
        click.secho(warning, fg="yellow")

    for error in errors:
        click.secho(error, fg="red")

    if errors:
        click.echo()
        click.secho(f"Found {len(errors)} error(s)", fg="red")
        sys.exit(1)
    else:
        click.secho("Config is valid (with warnings)", fg="yellow")


# =============================================================================
# Init Command
# =============================================================================


@cli.command()
@click.option("--output", "-o", default="backtest.yaml", help="Output filename")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing file")
def init(output: str, force: bool):
    """
    Create sample configuration file.

    \b
    Example:
        sf init
        sf init --output my_config.yaml
    """
    from signalflow.cli.config import generate_sample_config

    path = Path(output)

    if path.exists() and not force:
        click.secho(f"File already exists: {path}", fg="red", err=True)
        click.echo("Use --force to overwrite")
        sys.exit(1)

    content = generate_sample_config()
    path.write_text(content)

    click.secho(f"Created: {path}", fg="green")
    click.echo()
    click.echo("Next steps:")
    click.echo("  1. Edit the config file to match your setup")
    click.echo("  2. Run: sf validate backtest.yaml")
    click.echo("  3. Run: sf run backtest.yaml")


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
