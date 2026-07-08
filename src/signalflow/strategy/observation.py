"""Observation - what every strategy model sees each bar."""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from signalflow.engine.types import PortfolioSnapshot, Position
from signalflow.errors import SchemaVersionError

OBSERVATION_SCHEMA_VERSION = 1


@dataclass
class Observation:
    """Validated signals + portfolio + market state + constraints at one ts."""

    ts: object
    signals: pl.DataFrame
    portfolio: PortfolioSnapshot
    mandate: dict = field(default_factory=dict)
    schema_version: int = OBSERVATION_SCHEMA_VERSION

    def to_dict(self) -> dict:
        """Serialize to a JSON-able dict tagged with the schema version."""
        port = self.portfolio
        return {
            "schema_version": self.schema_version,
            "ts": str(self.ts),
            "mandate": self.mandate,
            "signals": self.signals.to_dicts(),
            "portfolio": {
                "ts": str(port.ts),
                "target": port.target,
                "balances": dict(port.balances),
                "equity": port.equity,
                "prices": dict(port.prices),
                "positions": {
                    p: {"pair": pos.pair, "qty": pos.qty, "avg_price": pos.avg_price, "opened_ts": str(pos.opened_ts)}
                    for p, pos in port.positions.items()
                },
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Observation":
        """Rebuild an Observation, rejecting a schema version it was not trained on."""
        version = payload.get("schema_version")
        if version != OBSERVATION_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Observation schema_version {version!r} does not match runtime {OBSERVATION_SCHEMA_VERSION!r}"
            )
        pf = payload.get("portfolio") or {}
        portfolio = PortfolioSnapshot(
            ts=pf.get("ts"),
            target=pf.get("target", ""),
            balances=dict(pf.get("balances", {})),
            positions={
                p: Position(pair=v["pair"], qty=v["qty"], avg_price=v["avg_price"])
                for p, v in (pf.get("positions") or {}).items()
            },
            equity=pf.get("equity", 0.0),
            prices=dict(pf.get("prices", {})),
        )
        return cls(
            ts=payload.get("ts"),
            signals=pl.DataFrame(payload.get("signals") or []),
            portfolio=portfolio,
            mandate=payload.get("mandate") or {},
            schema_version=version,
        )

    def to_vector(self) -> np.ndarray:
        """Stable fixed-length numeric summary for RL policies."""
        sig = self.signals
        n = max(sig.height, 1)
        rise = (sig.get_column("signal") == "rise").sum() / n if "signal" in sig.columns else 0.0
        p_succ = float(sig.get_column("p_success").fill_null(0.0).mean() or 0.0) if "p_success" in sig.columns else 0.0
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
