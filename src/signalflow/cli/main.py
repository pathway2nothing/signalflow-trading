"""
``sf`` - the SignalFlow command-line interface (click group).

Four commands: ``list`` (enumerate the registry), ``run`` (load a flow.yaml,
build a dataset, backtest, print the scorecard), ``promote`` (validate + show the
would-be registry op; the real promotion happens in sf-prod), and ``version``.

The registry autodiscovers components lazily, so importing :mod:`signalflow` and
touching ``registry.snapshot()`` is enough to populate ``sf list``.
"""

import contextlib
import sys

import click

import signalflow as sf
from signalflow.enums import ComponentType

_TYPE_MAP = {t.value: t for t in ComponentType}
_TYPE_CHOICES = sorted(_TYPE_MAP)


def _configure_stdio() -> None:
    """Force UTF-8 output so component docs with non-cp1252 glyphs never crash the console."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def _emit(line: str = "") -> None:
    """Print a single line (kept tiny so CliRunner output is predictable)."""
    click.echo(line)


def _print_table(title: str, rows: list[tuple[str, str]], headers: tuple[str, str]) -> None:
    """Render a two-column table with rich if available, else aligned plain text."""
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=title)
        table.add_column(headers[0], style="cyan", no_wrap=True)
        table.add_column(headers[1])
        for left, right in rows:
            table.add_row(left, right)
        Console().print(table)
        return
    except Exception:
        pass

    if title:
        _emit(title)
    width = max((len(left) for left, _ in rows), default=len(headers[0]))
    width = max(width, len(headers[0]))
    _emit(f"{headers[0]:<{width}}  {headers[1]}")
    _emit(f"{'-' * width}  {'-' * len(headers[1])}")
    for left, right in rows:
        _emit(f"{left:<{width}}  {right}")


@click.group()
@click.version_option(version=sf.__version__, prog_name="sf", message="%(prog)s %(version)s")
def main() -> None:
    """SignalFlow - research, backtest, and promote trading flows."""
    _configure_stdio()


@main.command(name="list")
@click.argument("type_", metavar="[TYPE]", required=False)
def list_(type_: str | None) -> None:
    """List registered components."""
    if type_ is None:
        snapshot = sf.registry.snapshot()
        if not snapshot:
            _emit("(registry is empty)")
            return
        for type_name in sorted(snapshot):
            names = snapshot[type_name]
            _emit(f"{type_name} ({len(names)})")
            for name in names:
                _emit(f"  {name}")
            _emit()
        return

    key = type_.strip().lower()
    if key not in _TYPE_MAP:
        raise click.BadArgumentUsage(f"unknown type {type_!r}; choose from: {', '.join(_TYPE_CHOICES)}")
    component_type = _TYPE_MAP[key]
    names = sf.registry.list(component_type)
    if not names:
        _emit(f"no {key} components registered")
        return

    rows: list[tuple[str, str]] = []
    for name in names:
        try:
            schema = sf.registry.get_schema(component_type, name)
            summary = schema.get("description", "") or ""
        except Exception:
            summary = ""
        rows.append((name, summary))
    _print_table(f"{key} components ({len(names)})", rows, ("name", "summary"))


def _describe_instance(component_type: ComponentType, name: str) -> tuple[str, str]:
    """Return ``(outputs, warmup)`` from a default-constructed instance, ``n/a`` on failure."""
    try:
        obj = sf.registry.create(component_type, name)
        outputs = getattr(obj, "outputs", None)
        warmup = getattr(obj, "warmup", None)
        return (str(outputs) if outputs else "n/a", str(warmup) if warmup is not None else "n/a")
    except Exception:
        return "n/a", "n/a"


@main.command()
@click.argument("type_", metavar="TYPE")
@click.argument("name")
def info(type_: str, name: str) -> None:
    """Show a component's schema: description, role, module, parameters, outputs, warmup."""
    key = type_.strip().lower()
    if key not in _TYPE_MAP:
        raise click.BadArgumentUsage(f"unknown type {type_!r}; choose from: {', '.join(_TYPE_CHOICES)}")
    component_type = _TYPE_MAP[key]
    try:
        schema = sf.registry.get_schema(component_type, name)
    except Exception as exc:
        raise click.BadArgumentUsage(str(exc)) from exc

    _emit(f"{schema['name']} ({schema['class_name']})")
    if schema.get("description"):
        _emit(schema["description"])
    _emit(f"role: {schema.get('role', '') or 'n/a'}")
    _emit(f"module: {schema.get('module', '') or 'n/a'}")
    _emit()

    rows = [
        (
            p["name"],
            f"{p['type']}  default={p['default']!r}  required={p['required']}",
        )
        for p in schema.get("parameters", [])
    ]
    if rows:
        _print_table("parameters", rows, ("param", "type / default"))
    else:
        _emit("parameters: (none)")

    outputs, warmup = _describe_instance(component_type, name)
    _emit()
    _emit(f"outputs: {outputs}")
    _emit(f"warmup: {warmup}")


@main.command()
@click.argument("flow_yaml", type=click.Path(exists=True, dir_okay=False))
@click.option("--source", default="memory", show_default=True, help="Registered data source name.")
@click.option("--pairs", default="BTCUSDT", show_default=True, help="Comma-separated trading pairs.")
@click.option("--start", default="2023-01-01", show_default=True, help="Start date (ISO).")
@click.option("--end", default=None, help="End date (ISO); optional.")
@click.option("--interval", default="1h", show_default=True, help="Bar interval, e.g. 1m/1h/1d.")
@click.option("--capital", default=50000.0, show_default=True, type=float, help="Starting capital.")
def run(
    flow_yaml: str,
    source: str,
    pairs: str,
    start: str,
    end: str | None,
    interval: str,
    capital: float,
) -> None:
    """Load FLOW_YAML, build a dataset, backtest, and print the scorecard."""
    flow = sf.Flow.load(flow_yaml)
    pair_list = [p.strip() for p in pairs.split(",") if p.strip()]
    dataset = sf.data(source, pairs=pair_list, start=start, end=end, interval=interval)
    run_result = flow.backtest(dataset, capital=capital)

    scorecard = run_result.scorecard()
    rows = [(str(k), str(v)) for k, v in scorecard.items()]
    _print_table(f"scorecard - {flow.name}", rows, ("metric", "value"))


@main.command()
@click.argument("flow_yaml", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--to",
    "target",
    required=True,
    type=click.Choice(["shadow", "live"]),
    help="Target stage to promote into.",
)
def promote(flow_yaml: str, target: str) -> None:
    """Validate FLOW_YAML and show the registry op that promotion WOULD perform."""
    flow = sf.Flow.load(flow_yaml)
    _emit(f"validated flow {flow.name!r} from {flow_yaml}")
    _emit(f"would register: stage={target} flow={flow.name!r} quote={flow.quote}")
    _emit("no server contacted - real promotion happens in sf-prod.")


@main.command()
def version() -> None:
    """Print the installed signalflow version."""
    _emit(sf.__version__)


if __name__ == "__main__":
    main()
