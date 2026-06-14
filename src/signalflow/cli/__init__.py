"""
SignalFlow command-line interface.

The ``sf`` console script (see ``[project.scripts]`` in ``pyproject.toml``)
points at :data:`signalflow.cli.main.main`, the click group. Run it via the
installed ``sf`` command or ``python -m signalflow.cli.main``.

This package intentionally does NOT eagerly import :mod:`signalflow.cli.main`:
doing so would place the module in ``sys.modules`` before ``runpy`` executes it,
producing a spurious ``RuntimeWarning`` under ``python -m signalflow.cli.main``.
Import the group explicitly with ``from signalflow.cli.main import main``.
"""


__all__: list[str] = []
