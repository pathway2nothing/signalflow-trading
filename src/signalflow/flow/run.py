"""Run - the result of executing a Flow."""


from dataclasses import dataclass, field

import numpy as np
import polars as pl

from signalflow.engine.types import Fill


@dataclass
class Run:
    name: str
    mode: str
    equity_curve: pl.DataFrame
    fills: list[Fill] = field(default_factory=list)
    target: str = "USDT"
    promotable: bool = True


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

    def sharpe(self, periods_per_year: int = 365 * 24) -> float:
        r = self.returns
        if r.size < 2 or r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(periods_per_year))

    def scorecard(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "promotable": self.promotable,
            "n_fills": len(self.fills),
            "initial_equity": round(self.initial_equity, 2),
            "final_equity": round(self.final_equity, 2),
            "total_return": round(self.total_return, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe": round(self.sharpe(), 3),
        }

    def __repr__(self) -> str:
        sc = self.scorecard()
        return (
            f"Run({self.name}, {self.mode}, ret={sc['total_return']:.2%}, "
            f"dd={sc['max_drawdown']:.2%}, sharpe={sc['sharpe']}, fills={sc['n_fills']})"
        )
