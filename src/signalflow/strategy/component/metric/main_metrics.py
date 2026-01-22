from __future__ import annotations
from dataclasses import dataclass, field
from signalflow.core import StrategyState
from signalflow.core.decorators import sf_component
from signalflow.strategy.component.base import StrategyMetric
import numpy as np


@dataclass
@sf_component(name='total_return', override=True)
class TotalReturnMetric(StrategyMetric):
    """Computes total return metrics for the portfolio."""
    
    initial_capital: float = 10000.0

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

@dataclass
@sf_component(name='balance_allocation', override=True)
class BalanceAllocationMetric(StrategyMetric):
    
    initial_capital: float = 10000.0
    
    def compute(self, state: StrategyState, prices: dict[str, float]) -> dict[str, float]:
        equity = state.portfolio.equity(prices=prices)
        cash = state.portfolio.cash
        
        positions_value = equity - cash
        capital_utilization = positions_value / equity if equity > 0 else 0.0
        free_balance_pct = cash / equity if equity > 0 else 0.0
        allocation_vs_initial = positions_value / self.initial_capital if self.initial_capital > 0 else 0.0
        
        return {
            'capital_utilization': capital_utilization,      
            'free_balance_pct': free_balance_pct,           
            'allocated_value': positions_value,             
            'allocation_vs_initial': allocation_vs_initial 
        }


@dataclass
@sf_component(name='drawdown', override=True)
class DrawdownMetric(StrategyMetric):
    
    _peak_equity: float = 0.0
    _max_drawdown: float = 0.0
    
    @property
    def name(self) -> str:
        return 'drawdown'
    
    def compute(self, state: StrategyState, prices: dict[str, float]) -> dict[str, float]:
        equity = state.portfolio.equity(prices=prices)
        
        if equity > self._peak_equity:
            self._peak_equity = equity
        
        current_drawdown = 0.0
        if self._peak_equity > 0:
            current_drawdown = (self._peak_equity - equity) / self._peak_equity
        
        if current_drawdown > self._max_drawdown:
            self._max_drawdown = current_drawdown
        
        return {
            'current_drawdown': current_drawdown,
            'max_drawdown': self._max_drawdown,
            'peak_equity': self._peak_equity
        }


@dataclass
@sf_component(name='win_rate', override=True)
class WinRateMetric(StrategyMetric):
    def compute(self, state: StrategyState, prices: dict[str, float]) -> dict[str, float]:
        closed_positions = [
            p for p in state.portfolio.positions.values() 
            if p.is_closed
        ]
        
        if not closed_positions:
            return {
                'win_rate': 0.0,
                'winning_trades': 0,
                'losing_trades': 0
            }
        
        winning = sum(1 for p in closed_positions if p.realized_pnl > 0)
        losing = sum(1 for p in closed_positions if p.realized_pnl <= 0)
        
        win_rate = winning / len(closed_positions) if closed_positions else 0.0
        
        return {
            'win_rate': win_rate,
            'winning_trades': winning,
            'losing_trades': losing
        }


@dataclass
@sf_component(name='sharpe_ratio', override=True)
class SharpeRatioMetric(StrategyMetric):

    initial_capital: float = 10000.0
    window_size: int = 100  
    risk_free_rate: float = 0.0 
    _returns_history: list[float] = None
    
    def __post_init__(self):
        self._returns_history = []
    
    @property
    def name(self) -> str:
        return 'sharpe_ratio'
    
    def compute(self, state: StrategyState, prices: dict[str, float]) -> dict[str, float]:
        import numpy as np
        
        equity = state.portfolio.equity(prices=prices)
        current_return = (equity - self.initial_capital) / self.initial_capital
        
        self._returns_history.append(current_return)
        
        if len(self._returns_history) > self.window_size:
            self._returns_history.pop(0)
        
        if len(self._returns_history) < 2:
            return {'sharpe_ratio': 0.0}
        
        returns_array = np.array(self._returns_history)
        returns_diff = np.diff(returns_array)
        
        if len(returns_diff) < 2:
            return {'sharpe_ratio': 0.0}
        
        mean_return = np.mean(returns_diff)
        std_return = np.std(returns_diff)
        
        if std_return == 0:
            return {'sharpe_ratio': 0.0}
        
        sharpe = (mean_return - self.risk_free_rate) / std_return
        
        return {'sharpe_ratio': sharpe}