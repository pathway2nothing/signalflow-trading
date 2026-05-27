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
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from signalflow.api.result import BacktestResult


# =============================================================================
# CLI Group
# =============================================================================


@click.group()
@click.version_option(package_name="signalflow-trading")
def cli() -> None:
    """SignalFlow - Trading signal framework CLI."""
    pass


# =============================================================================
# Run Command
# =============================================================================


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--mode", "-m", type=click.Choice(["backtest", "train", "optimize", "live"]), help="Execution mode")
@click.option("--output", "-o", type=click.Path(), help="Save results to JSON file")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
@click.option("--plot", "-p", is_flag=True, help="Show plots after backtest")
@click.option("--set", "params", multiple=True, help="Override parameters (e.g., --set tp=0.03)")
@click.option("--dry-run", is_flag=True, help="Validate without executing")
def run(
    config_path: str,
    mode: str | None,
    output: str | None,
    quiet: bool,
    plot: bool,
    params: tuple[str, ...],
    dry_run: bool,
) -> None:
    """
    Run flow from YAML configuration file.

    \b
    Example:
        sf run config.yaml
        sf run config.yaml --mode backtest
        sf run config.yaml --mode train --output results.json
        sf run flows/grid_sma.yml --mode backtest --set tp=0.03 --set sl=0.02
        sf run flows/grid_sma/ --mode live
    """
    from signalflow.cli.config import BacktestConfig
    from signalflow.config.mode_config import FlowMode, resolve_flow_config

    config_path_obj = Path(config_path)

    # Parse --set parameters
    param_overrides: dict[str, str | int | float] = {}
    for p in params:
        if "=" in p:
            key, value = p.split("=", 1)
            # Try to parse as number
            try:
                if "." in value:
                    param_overrides[key] = float(value)
                else:
                    param_overrides[key] = int(value)
            except ValueError:
                param_overrides[key] = value

    # Try new flow config format first
    try:
        flow_mode = FlowMode(mode) if mode else None
        resolved_config = resolve_flow_config(config_path_obj, flow_mode, param_overrides)

        # Check if it's a flow-style config
        if "detectors" in resolved_config or "strategy" in resolved_config:
            _run_flow_config(resolved_config, output, quiet, plot, dry_run)
            return
    except Exception:
        pass  # Fall back to legacy config

    # Legacy BacktestConfig
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

    if dry_run:
        click.secho("Config is valid! (dry run)", fg="green")
        return

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


def _run_flow_config(config: dict, output: str | None, quiet: bool, plot: bool, dry_run: bool) -> None:
    """Run flow-style configuration."""
    from signalflow.config.validation import validate_flow_config

    flow_id = config.get("flow_id", "unknown")
    mode = config.get("_mode", "backtest")

    click.echo(f"Loading flow: {flow_id} (mode: {mode})")

    # Validate
    errors, warnings = validate_flow_config(config)

    for warning in warnings:
        click.secho(f"WARNING: {warning}", fg="yellow")

    if errors:
        for error in errors:
            click.secho(f"ERROR: {error}", fg="red", err=True)
        sys.exit(1)

    if dry_run:
        click.secho("Flow config is valid! (dry run)", fg="green")
        _display_flow_info(config)
        return

    # Run based on mode
    click.echo(f"Running {mode}: {flow_id}")

    try:
        if mode == "backtest":
            _run_backtest_mode(config, output, quiet, plot)
        elif mode == "train":
            _run_train_mode(config, output, quiet)
        elif mode == "optimize":
            _run_optimize_mode(config, output, quiet)
        elif mode == "live":
            click.secho("Live mode not yet implemented via CLI", fg="yellow")
            sys.exit(1)
    except Exception as e:
        click.secho(f"Error during {mode}: {e}", fg="red", err=True)
        sys.exit(1)


def _display_flow_info(config: dict) -> None:
    """Display flow configuration info."""
    click.echo()
    click.echo("Flow Configuration:")
    click.echo("-" * 40)
    click.echo(f"  ID:       {config.get('flow_id', 'unknown')}")
    click.echo(f"  Name:     {config.get('flow_name', config.get('flow_id', ''))}")
    click.echo(f"  Mode:     {config.get('_mode', 'backtest')}")

    detectors = config.get("detectors", [])
    click.echo(f"  Detectors: {len(detectors)}")
    for d in detectors:
        click.echo(f"    - {d.get('id', 'unknown')}: {d.get('logic', {}).get('type', 'unknown')}")

    validators = config.get("validators", [])
    if validators:
        click.echo(f"  Validators: {len(validators)}")

    strategy = config.get("strategy", {})
    if strategy:
        reconcile = strategy.get("reconcile", {})
        click.echo(f"  Reconcile: {reconcile.get('mode', 'any')}")

    click.echo()


def _run_backtest_mode(config: dict, output: str | None, quiet: bool, plot: bool) -> None:
    """Run backtest mode."""
    from signalflow import Backtest

    # Convert flow config to backtest config
    backtest_config = _flow_to_backtest_config(config)

    result = Backtest.from_dict(backtest_config).run()

    _display_result(result)

    if output:
        _save_result(result, output)
        click.echo(f"Results saved to: {output}")

    if plot:
        _show_plots(result)


def _run_train_mode(config: dict, output: str | None, quiet: bool) -> None:
    """Run train mode."""
    click.secho("Train mode execution...", fg="cyan")

    # TODO: Implement full training pipeline
    # For now, just validate the training config
    train_config = config.get("train", {})
    if not train_config:
        click.secho("WARNING: No train configuration found", fg="yellow")
        return

    labeling = train_config.get("labeling", {})
    click.echo(f"  Labeling: {labeling.get('type', 'unknown')}")

    output_dir = train_config.get("output", {}).get("models_dir", "artifacts/models/")
    click.echo(f"  Output: {output_dir}")

    click.secho("Training pipeline not fully implemented yet", fg="yellow")


def _run_optimize_mode(config: dict, output: str | None, quiet: bool) -> None:
    """Run optimize mode."""
    click.secho("Optimize mode execution...", fg="cyan")

    optimize_config = config.get("optimize", {})
    if not optimize_config:
        click.secho("WARNING: No optimize configuration found", fg="yellow")
        return

    search_space = optimize_config.get("search_space", {})
    click.echo(f"  Parameters: {list(search_space.keys())}")

    optimizer = optimize_config.get("optimizer", {})
    click.echo(f"  Trials: {optimizer.get('n_trials', 100)}")

    click.secho("Optimization pipeline not fully implemented yet", fg="yellow")


def _flow_to_backtest_config(config: dict) -> dict:
    """Convert flow config to BacktestBuilder config."""
    result = {
        "strategy_id": config.get("flow_id", "backtest"),
    }

    # Data config
    data = config.get("data", {})
    if data:
        result["pairs"] = data.get("pairs", ["BTCUSDT"])
        result["timeframe"] = data.get("timeframe", "1h")

    # Capital from backtest mode config
    backtest = config.get("backtest", {})
    capital = backtest.get("capital", {})
    result["capital"] = capital.get("initial", config.get("capital", {}).get("initial", 10000))

    # Costs
    costs = backtest.get("costs", {})
    result["fee"] = costs.get("fee_rate", 0.001)

    # Detectors
    detectors = config.get("detectors", [])
    if detectors:
        d = detectors[0]
        result["detectors"] = {
            d.get("id", "main"): {
                "class_name": d.get("logic", {}).get("type", ""),
                "params": {k: v for k, v in d.get("logic", {}).items() if k != "type"},
            }
        }

    # Strategy
    strategy = config.get("strategy", {})
    entries = strategy.get("entries", {})
    exits = strategy.get("exits", {})

    if entries.get("rules"):
        rule = entries["rules"][0]
        result["entry"] = {
            "size": rule.get("base_size", 100),
            **{k: v for k, v in rule.items() if k not in {"type", "base_size"}},
        }

    if exits.get("rules"):
        rule = exits["rules"][0]
        if rule.get("type") == "tp_sl":
            result["exit"] = {
                "tp": rule.get("take_profit_pct", 0.02),
                "sl": rule.get("stop_loss_pct", 0.01),
            }

    return result


def _display_result(result: BacktestResult) -> None:
    """Display backtest result summary."""
    click.echo()
    click.echo("=" * 50)
    click.secho("         BACKTEST RESULT", bold=True)
    click.echo("=" * 50)

    # Return with color
    return_color = "green" if result.total_return >= 0 else "red"
    click.echo("  Total Return:    ", nl=False)
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
    def convert(obj: Any) -> str:
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
def list() -> None:
    """List available components from registry."""
    pass


@list.command("detectors")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed info")
def list_detectors(verbose: bool) -> None:
    """List available signal detectors."""
    from signalflow.core import SfComponentType, default_registry

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
def list_metrics(verbose: bool) -> None:
    """List available strategy metrics."""
    from signalflow.core import SfComponentType, default_registry

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
def list_features(verbose: bool) -> None:
    """List available features."""
    from signalflow.core import SfComponentType, default_registry

    features = default_registry.list(SfComponentType.FEATURE)

    if not features:
        click.echo("No features registered.")
        return

    click.echo(f"Available features ({len(features)}):")
    click.echo("-" * 40)

    for name in sorted(features):
        click.secho(f"  {name}", fg="green")


@list.command("all")
def list_all() -> None:
    """List all registered components."""
    from signalflow.core import SfComponentType, default_registry

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
def validate(config_path: str) -> None:
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
def init(output: str, force: bool) -> None:
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
# Preset Commands
# =============================================================================


@cli.group()
def preset() -> None:
    """Browse and use strategy presets."""
    pass


@preset.command("list")
@click.option(
    "--difficulty",
    "-d",
    type=click.Choice(["beginner", "intermediate", "advanced"]),
    help="Filter by difficulty",
)
@click.option("--tag", "-t", type=str, help="Filter by tag")
def preset_list(difficulty: str | None, tag: str | None) -> None:
    """List available strategy presets.

    \b
    Example:
        sf preset list
        sf preset list --difficulty beginner
        sf preset list --tag momentum
    """
    try:
        from signalflow.ta.presets import list_presets
    except ImportError:
        click.secho("signalflow-ta not installed. Install it for presets:", fg="red")
        click.secho("  pip install signalflow-ta", fg="cyan")
        return

    presets = list_presets(difficulty=difficulty, tag=tag)

    if not presets:
        click.echo("No presets found matching filters.")
        return

    click.echo(f"Available presets ({len(presets)}):")
    click.echo("-" * 60)
    for p in presets:
        diff_color = {"beginner": "green", "intermediate": "yellow", "advanced": "red"}.get(p.difficulty, "white")
        click.echo(f"  {p.name:<22} ", nl=False)
        click.secho(f"[{p.difficulty}]", fg=diff_color, nl=False)
        click.echo(f"  {p.description[:40]}")


@preset.command("show")
@click.argument("name")
def preset_show(name: str) -> None:
    """Show details of a strategy preset.

    \b
    Example:
        sf preset show grid
        sf preset show momentum
    """
    try:
        from signalflow.ta.presets import get_preset
    except ImportError:
        click.secho("signalflow-ta not installed.", fg="red")
        return

    try:
        p = get_preset(name)
    except KeyError as e:
        click.secho(str(e), fg="red")
        return

    click.echo(f"  {p.display_name}")
    click.echo(f"  {'=' * len(p.display_name)}")
    click.echo(f"  {p.description}")
    click.echo()
    click.echo(f"  Difficulty:  {p.difficulty}")
    click.echo(f"  Tags:        {', '.join(p.tags)}")
    click.echo(f"  Detector:    {p.detector}")
    click.echo(f"  Capital:     ${p.capital:,.0f}")
    click.echo(f"  Fee:         {p.fee:.4f}")
    if p.notes:
        click.echo(f"  Notes:       {p.notes}")


@preset.command("init")
@click.argument("name")
@click.option("--output", "-o", default=None, help="Output YAML filename")
def preset_init(name: str, output: str | None) -> None:
    """Generate a YAML config from a preset.

    \b
    Example:
        sf preset init grid
        sf preset init momentum --output my_momentum.yaml
    """
    import yaml

    try:
        from signalflow.ta.presets import get_preset
    except ImportError:
        click.secho("signalflow-ta not installed.", fg="red")
        return

    try:
        p = get_preset(name)
    except KeyError as e:
        click.secho(str(e), fg="red")
        return

    config = {
        "strategy_id": f"preset/{p.name}",
        "detector": p.detector,
        "detector_params": dict(p.detector_params),
        "entry": dict(p.entry_params),
        "exit": dict(p.exit_params),
        "capital": p.capital,
        "fee": p.fee,
    }
    if p.features:
        config["features"] = p.features

    out_path = Path(output) if output else Path(f"{p.name}_preset.yaml")
    out_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
    click.secho(f"Created: {out_path}", fg="green")


# =============================================================================
# Compare Command
# =============================================================================


@cli.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--metric", "-m", default="sharpe_ratio", help="Metric to highlight best/worst")
def compare(files: tuple[str, ...], metric: str) -> None:
    """Compare multiple backtest result JSON files.

    \b
    Example:
        sf compare run1.json run2.json
        sf compare results/*.json --metric total_return
    """
    import json

    if len(files) < 2:
        click.secho("Need at least 2 result files to compare.", fg="red")
        return

    from signalflow.analytic.compare import compare_results

    class _DictResult:
        def __init__(self, name: str, metrics: dict[str, float]) -> None:
            self.strategy_id = name
            self.metrics = metrics

    results = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            metrics = data.get("metrics", {})
            name = data.get("config", {}).get("strategy_id", Path(f).stem)
            results.append(_DictResult(name, metrics))
        except Exception as e:
            click.secho(f"Error reading {f}: {e}", fg="red")
            return

    cmp = compare_results(*results)
    click.echo(cmp.summary())

    try:
        click.echo()
        click.echo(f"Best {metric}: ", nl=False)
        click.secho(cmp.best(metric), fg="green", bold=True)
    except KeyError:
        pass


# =============================================================================
# Help Command
# =============================================================================


@cli.command("help")
@click.argument("term", required=False, default="")
def help_cmd(term: str) -> None:
    """Look up component or metric documentation.

    \b
    Example:
        sf help sharpe_ratio
        sf help sma_cross
        sf help detectors
    """
    from signalflow._help import help_system

    help_system(term)


# =============================================================================
# Optional Extension Commands
# =============================================================================

try:
    from signalflow.cli.benchmark import benchmark_ta

    cli.add_command(benchmark_ta)
except ImportError:
    pass


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
