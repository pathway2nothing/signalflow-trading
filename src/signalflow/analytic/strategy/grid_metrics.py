"""Grid-specific strategy metrics."""

from __future__ import annotations

from dataclasses import dataclass

from signalflow.analytic.base import StrategyMetric
from signalflow.core import PositionType, StrategyState, sf_component


@dataclass
@sf_component(name="grid_metrics", override=True)
class GridMetrics(StrategyMetric):
    """Track grid-specific performance metrics per bar.

    Emits:
        - grid_open_levels: number of open positions
        - grid_closed_levels: number of closed positions
        - grid_total_levels: total positions (open + closed)
        - grid_total_pnl: sum of all position PnL (realized + unrealized)
        - grid_avg_pnl_per_level: average PnL per position
        - grid_best_level_pnl: best individual position PnL
        - grid_worst_level_pnl: worst individual position PnL
        - grid_capital_deployed: total notional of open positions
        - grid_efficiency: total_pnl / capital_deployed
        - grid_price_spread: (max_entry - min_entry) / min_entry for open positions
        - grid_avg_entry_price: qty-weighted average entry price of open positions
    """

    def compute(self, state: StrategyState, prices: dict[str, float], **kwargs) -> dict[str, float]:
        all_positions = list(state.portfolio.positions.values())
        open_positions = [p for p in all_positions if not p.is_closed]
        closed_positions = [p for p in all_positions if p.is_closed]

        n_open = len(open_positions)
        n_closed = len(closed_positions)
        n_total = len(all_positions)

        # PnL: realized for closed, unrealized for open
        pnls: list[float] = []
        for pos in closed_positions:
            pnls.append(pos.realized_pnl)
        for pos in open_positions:
            price = prices.get(pos.pair, 0.0)
            if price > 0:
                if pos.position_type == PositionType.LONG:
                    pnls.append((price - pos.entry_price) * pos.qty)
                else:
                    pnls.append((pos.entry_price - price) * pos.qty)
            else:
                pnls.append(0.0)

        total_pnl = sum(pnls)
        avg_pnl = total_pnl / n_total if n_total > 0 else 0.0
        best_pnl = max(pnls) if pnls else 0.0
        worst_pnl = min(pnls) if pnls else 0.0

        # Capital deployed (open positions notional)
        capital_deployed = sum(p.entry_price * p.qty for p in open_positions)
        efficiency = total_pnl / capital_deployed if capital_deployed > 0 else 0.0

        # Price spread across open positions
        if n_open >= 2:
            entries = [p.entry_price for p in open_positions]
            min_entry = min(entries)
            max_entry = max(entries)
            price_spread = (max_entry - min_entry) / min_entry if min_entry > 0 else 0.0
        else:
            price_spread = 0.0

        # Weighted average entry price
        total_qty = sum(p.qty for p in open_positions)
        if total_qty > 0:
            avg_entry = sum(p.entry_price * p.qty for p in open_positions) / total_qty
        else:
            avg_entry = 0.0

        return {
            "grid_open_levels": float(n_open),
            "grid_closed_levels": float(n_closed),
            "grid_total_levels": float(n_total),
            "grid_total_pnl": total_pnl,
            "grid_avg_pnl_per_level": avg_pnl,
            "grid_best_level_pnl": best_pnl,
            "grid_worst_level_pnl": worst_pnl,
            "grid_capital_deployed": capital_deployed,
            "grid_efficiency": efficiency,
            "grid_price_spread": price_spread,
            "grid_avg_entry_price": avg_entry,
        }
