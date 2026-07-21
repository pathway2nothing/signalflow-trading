"""Run - the result of executing a Flow."""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from signalflow.engine.types import Fill


@dataclass
class Run:
    """Result of executing a Flow: equity curve, fills, and derived metrics.

    ``promotable`` is false for an in-sample model backtest; ``oos`` flags a
    leak-free out-of-sample run.
    """

    name: str
    mode: str
    equity_curve: pl.DataFrame
    fills: list[Fill] = field(default_factory=list)
    target: str = "USDT"
    promotable: bool = True
    oos: bool = False
    oos_coverage: "float | None" = None

    @property
    def initial_equity(self) -> float:
        return float(self.equity_curve.get_column("equity")[0]) if self.equity_curve.height else 0.0

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.get_column("equity")[-1]) if self.equity_curve.height else 0.0

    @property
    def total_return(self) -> float:
        e0 = self.initial_equity
        return (self.final_equity / e0 - 1.0) if e0 else 0.0

    @property
    def returns(self) -> np.ndarray:
        eq = self.equity_curve.get_column("equity").to_numpy()
        if eq.size < 2:
            return np.array([])
        return np.diff(eq) / eq[:-1]

    @property
    def max_drawdown(self) -> float:
        eq = self.equity_curve.get_column("equity").to_numpy()
        if eq.size == 0:
            return 0.0
        peak = np.maximum.accumulate(eq)
        return float(np.max((peak - eq) / peak)) if np.all(peak > 0) else 0.0

    def periods_per_year(self) -> float:
        """Annualization factor derived from the equity curve's median bar spacing."""
        seconds_per_year = 365.0 * 24.0 * 3600.0
        ec = self.equity_curve
        if ec.height >= 3 and "ts" in ec.columns:
            secs = ec.get_column("ts").sort().diff().drop_nulls().dt.total_seconds()
            positive = secs.filter(secs > 0)
            if positive.len() > 0:
                median_dt = float(positive.median())
                if median_dt > 0:
                    return seconds_per_year / median_dt
        return 8760.0

    def sharpe(self, periods_per_year: float | None = None) -> float:
        """Annualized Sharpe of per-bar equity returns (zero risk-free rate).

        ``periods_per_year`` defaults to the factor implied by the equity curve's median
        bar spacing (:meth:`periods_per_year`), so hourly bars annualize by ~8760.
        """
        r = self.returns
        if r.size < 2 or r.std() == 0:
            return 0.0
        ppy = self.periods_per_year() if periods_per_year is None else periods_per_year
        return float(r.mean() / r.std() * np.sqrt(ppy))

    def scorecard(self) -> dict:
        """Standard metric dict: ``name``, ``mode``, ``target``, ``promotable``, ``oos``,
        ``n_fills``, ``initial_equity``, ``final_equity``, ``total_return``,
        ``max_drawdown``, and ``sharpe``.
        """
        card = {
            "name": self.name,
            "mode": self.mode,
            "target": self.target,
            "promotable": self.promotable,
            "oos": self.oos,
            "n_fills": len(self.fills),
            "initial_equity": round(self.initial_equity, 2),
            "final_equity": round(self.final_equity, 2),
            "total_return": round(self.total_return, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe": round(self.sharpe(), 3),
        }
        if self.oos_coverage is not None:
            card["oos_coverage"] = round(self.oos_coverage, 4)
        return card

    def __repr__(self) -> str:
        sc = self.scorecard()
        return (
            f"Run({self.name}, {self.mode}, ret={sc['total_return']:.2%}, "
            f"dd={sc['max_drawdown']:.2%}, sharpe={sc['sharpe']}, fills={sc['n_fills']})"
        )
