"""Observation - what every strategy model sees each bar."""


from dataclasses import dataclass, field

import numpy as np
import polars as pl

from signalflow.engine.types import PortfolioSnapshot

OBSERVATION_SCHEMA_VERSION = 1


@dataclass
class Observation:
    """Validated signals + portfolio + market state + constraints at one ts."""

    ts: object
    signals: pl.DataFrame
    portfolio: PortfolioSnapshot
    mandate: dict = field(default_factory=dict)
    schema_version: int = OBSERVATION_SCHEMA_VERSION


    def to_vector(self) -> np.ndarray:
        """Stable fixed-length numeric summary for RL policies."""
        sig = self.signals
        n = max(sig.height, 1)
        rise = (sig.get_column("signal") == "rise").sum() / n if "signal" in sig.columns else 0.0
        p_succ = (
            float(sig.get_column("p_success").fill_null(0.0).mean() or 0.0)
            if "p_success" in sig.columns
            else 0.0
        )
        eq = self.portfolio.equity
        cash = self.portfolio.balances.get(self.portfolio.target, 0.0)
        return np.array(
            [
                float(eq),
                float(cash / eq) if eq else 0.0,
                float(len(self.portfolio.positions)),
                float(rise),
                float(p_succ),
            ],
            dtype=np.float64,
        )


    def to_prompt_context(self) -> dict:
        """JSON-able structured context for an LLM strategy (no raw candles)."""
        cols = [c for c in ("pair", "signal", "p_success") if c in self.signals.columns]
        return {
            "ts": str(self.ts),
            "mandate": self.mandate,
            "equity": self.portfolio.equity,
            "positions": {
                p: {"qty": pos.qty, "avg_price": pos.avg_price} for p, pos in self.portfolio.positions.items()
            },
            "signals": self.signals.select(cols).to_dicts() if cols else [],
        }
