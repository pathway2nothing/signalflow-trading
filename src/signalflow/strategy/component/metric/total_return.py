from dataclasses import dataclass
from signalflow.core import StrategyState
from signalflow.core.decorators import sf_component
from signalflow.strategy.component.base import StrategyMetric

@dataclass
@sf_component(name='total_return', override=True)
class TotalReturnMetric(StrategyMetric):
    """Computes total return metrics for the portfolio."""
    
    initial_capital: float = 10000.0
    
    @property
    def name(self) -> str:
        return 'total_return'
    
    def compute(
        self,
        state: StrategyState,
        prices: dict[str, float]
    ) -> dict[str, float]:
        equity = state.portfolio.equity(prices=prices)
        cash = state.portfolio.cash
        
        total_realized = sum(p.realized_pnl for p in state.portfolio.positions.values())
        total_unrealized = sum(p.unrealized_pnl for p in state.portfolio.open_positions())
        total_fees = sum(p.fees_paid for p in state.portfolio.positions.values())
        
        total_return = (equity - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0.0
        
        return {
            'equity': equity,
            'cash': cash,
            'total_return': total_return,
            'realized_pnl': total_realized,
            'unrealized_pnl': total_unrealized,
            'total_fees': total_fees,
            'open_positions': len(state.portfolio.open_positions()),
            'closed_positions': len([p for p in state.portfolio.positions.values() if p.is_closed]),
        }
