"""Portfolio-level metrics for multi-asset tracking."""

from __future__ import annotations

from dataclasses import dataclass

from signalflow.analytic.base import StrategyMetric
from signalflow.core import StrategyState, sf_component


@dataclass
@sf_component(name="portfolio_exposure", override=True)
class PortfolioExposureMetric(StrategyMetric):
    """Track portfolio exposure, leverage, and concentration per bar.

    Emits the following metrics each bar:
        - gross_exposure: sum of |position_notional|
        - net_exposure: signed sum of position notionals
        - leverage: gross_exposure / equity
        - n_pairs: number of distinct pairs with open positions
        - max_pair_pct: largest single-pair share of gross exposure
    """

    def compute(self, state: StrategyState, prices: dict[str, float], **kwargs) -> dict[str, float]:
        portfolio = state.portfolio
        gross = portfolio.gross_exposure(prices=prices)
        net = portfolio.net_exposure(prices=prices)
        lev = portfolio.leverage(prices=prices)
        conc = portfolio.concentration(prices=prices)

        return {
            "gross_exposure": gross,
            "net_exposure": net,
            "leverage": lev,
            "n_pairs": len(conc),
            "max_pair_pct": max(conc.values()) if conc else 0.0,
        }


@dataclass
@sf_component(name="portfolio_pnl_breakdown", override=True)
class PortfolioPnLBreakdownMetric(StrategyMetric):
    """Per-pair PnL breakdown for multi-asset strategies.

    Emits:
        - per_pair_realized: sum of realized PnL across pairs
        - best_pair_pnl: pair with highest total PnL
        - worst_pair_pnl: pair with lowest total PnL
        - pair_count_open: number of pairs with open positions
    """

    def compute(self, state: StrategyState, prices: dict[str, float], **kwargs) -> dict[str, float]:
        pair_pnl: dict[str, float] = {}
        for p in state.portfolio.positions.values():
            pnl = p.total_pnl
            pair_pnl[p.pair] = pair_pnl.get(p.pair, 0.0) + pnl

        if not pair_pnl:
            return {
                "per_pair_realized": 0.0,
                "best_pair_pnl": 0.0,
                "worst_pair_pnl": 0.0,
                "pair_count_open": 0.0,
            }

        pnl_values = list(pair_pnl.values())
        open_pairs = state.portfolio.positions_by_pair(open_only=True)

        return {
            "per_pair_realized": sum(pnl_values),
            "best_pair_pnl": max(pnl_values),
            "worst_pair_pnl": min(pnl_values),
            "pair_count_open": float(len(open_pairs)),
        }
