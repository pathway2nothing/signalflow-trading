"""
Experiment - the research lifecycle wrapper around a Flow run.

An Experiment runs a Flow's backtest, keeps the resulting :class:`Run`, and logs
timestamped lifecycle stages. ``.scorecard()`` produces a stable-shape summary
(optionally vs a baseline).

Determinism note: importable code paths used by tests never call
``datetime.now`` themselves - the caller supplies timestamps (or a ``clock``
callable). The default clock is only invoked when no timestamp is given, and the
tests pass explicit values, so behaviour stays reproducible.
"""

from collections.abc import Callable

from signalflow.experiment.scorecard import Scorecard


class Experiment:
    """Lifecycle wrapper: run a Flow, store the Run, score it against a baseline."""

    def __init__(self, name: str, baseline=None, *, clock: Callable[[], object] | None = None) -> None:
        self.name = name
        self.baseline = baseline
        self._clock = clock
        self.last_run = None
        self._baseline_run = None
        self._last_run_args: dict | None = None

        self.log: dict = {"created": None, "first_result": None, "stages": []}

    def stamp(self, stage: str, ts=None):
        """Record ``stage`` with timestamp ``ts`` (or the clock's value)."""
        if ts is None and self._clock is not None:
            ts = self._clock()
        self.log["stages"].append({"stage": stage, "ts": ts})
        if stage in self.log and self.log[stage] is None:
            self.log[stage] = ts
        return ts

    def run(self, flow, data, capital, target=None, *, oos: bool = False, broker=None, ts=None):
        """Backtest ``flow`` on ``data``, store the Run, and stamp ``first_result``."""
        if self.log["created"] is None:
            self.stamp("created", ts=ts)
        run = flow.backtest(data, capital, target=target, broker=broker, oos=oos)
        self.last_run = run
        self._last_run_args = {"data": data, "capital": capital, "target": target}
        self.stamp("first_result", ts=ts)
        return run

    def _resolve_baseline_run(self):
        if self.baseline is None:
            return None
        if self._baseline_run is not None:
            return self._baseline_run

        if hasattr(self.baseline, "equity_curve"):
            self._baseline_run = self.baseline
        elif hasattr(self.baseline, "backtest"):
            if self._last_run_args is None:
                raise RuntimeError("cannot backtest baseline Flow before .run() supplies data/capital")
            a = self._last_run_args
            self._baseline_run = self.baseline.backtest(a["data"], a["capital"], target=a["target"])
        else:
            raise TypeError(f"baseline must be a Run, a Flow, or None; got {type(self.baseline)!r}")
        return self._baseline_run

    def scorecard(self, *, n: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
        if self.last_run is None:
            raise RuntimeError("no run yet; call .run(flow, data, capital) first")
        baseline_run = self._resolve_baseline_run()
        return Scorecard.from_run(self.last_run, baseline_run, n=n, alpha=alpha, seed=seed)
