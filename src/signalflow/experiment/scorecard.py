"""Scorecard - one stable-shape summary of a Run."""

from signalflow.experiment.stats import bootstrap_ci, monte_carlo_bounds

SCORECARD_KEYS = (
    "name",
    "mode",
    "total_return",
    "sharpe",
    "max_drawdown",
    "n_fills",
    "bootstrap_ci",
    "monte_carlo",
    "delta",
    "promotable",
    "oos",
)


class Scorecard:
    """Namespace for scorecard construction; the product is a plain dict."""

    @staticmethod
    def from_run(run, baseline=None, *, n: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
        """Summarize ``run`` into a stable-shape dict, optionally vs ``baseline``."""
        returns = run.returns
        lo, hi = bootstrap_ci(returns, n=n, alpha=alpha, seed=seed)
        mc = monte_carlo_bounds(returns, n=n, seed=seed)

        card: dict = {
            "name": run.name,
            "mode": run.mode,
            "total_return": float(run.total_return),
            "sharpe": float(run.sharpe()),
            "max_drawdown": float(run.max_drawdown),
            "n_fills": len(run.fills),
            "bootstrap_ci": {"low": lo, "high": hi, "alpha": alpha},
            "monte_carlo": mc,
            "delta": None,
            "promotable": bool(run.promotable),
            "oos": bool(run.oos),
        }
        if getattr(run, "oos_coverage", None) is not None:
            card["oos_coverage"] = float(run.oos_coverage)

        if baseline is not None:
            card["delta"] = {
                "total_return": float(run.total_return - baseline.total_return),
                "sharpe": float(run.sharpe() - baseline.sharpe()),
                "max_drawdown": float(run.max_drawdown - baseline.max_drawdown),
                "n_fills": len(run.fills) - len(baseline.fills),
            }
        return card
