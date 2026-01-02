"""Strategy metrics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from signalflow.core import sf_component
from signalflow.core.enums import SfComponentType
from signalflow.strategy.component.base import StrategyMetric
from signalflow.strategy.state import StrategyState
from signalflow.strategy.types import StrategyContext


@dataclass
@sf_component(name='std_metrics')
class StandardMetrics(StrategyMetric):
    """Standard portfolio metrics.
    
    Computes:
    - open_positions: Number of open positions
    - unrealized_pnl: Total unrealized PnL
    - realized_pnl: Total realized PnL
    - total_fees: Total fees paid
    - portfolio_value: Unrealized + Realized - Fees
    """
    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_METRIC
    
    def compute(
        self,
        *,
        state: StrategyState,
        context: StrategyContext,
    ) -> dict[str, float]:
        """Compute standard metrics."""
        open_positions = state.portfolio.open_positions()
        all_positions = state.portfolio.positions.values()
        
        unrealized_pnl = sum(p.unrealized_pnl for p in open_positions)
        realized_pnl = sum(p.realized_pnl for p in all_positions)
        total_fees = sum(p.fees_paid for p in all_positions)
        
        return {
            'open_positions': float(len(open_positions)),
            'unrealized_pnl': float(unrealized_pnl),
            'realized_pnl': float(realized_pnl),
            'total_fees': float(total_fees),
            'portfolio_value': float(unrealized_pnl + realized_pnl - total_fees),
        }


@dataclass
@sf_component(name='exposure_metrics')
class ExposureMetrics(StrategyMetric):
    """Exposure and risk metrics.
    
    Computes:
    - total_exposure: Sum of position notionals
    - avg_position_size: Average position size
    - max_position_pnl: Highest unrealized PnL
    - min_position_pnl: Lowest unrealized PnL
    - positions_by_pair: Count by trading pair
    """
    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_METRIC
    
    def compute(
        self,
        *,
        state: StrategyState,
        context: StrategyContext,
    ) -> dict[str, float]:
        """Compute exposure metrics."""
        open_positions = state.portfolio.open_positions()
        
        if not open_positions:
            return {
                'total_exposure': 0.0,
                'avg_position_size': 0.0,
                'max_position_pnl': 0.0,
                'min_position_pnl': 0.0,
                'num_pairs': 0.0,
            }
        
        exposures = [p.last_price * p.qty for p in open_positions]
        pnls = [p.unrealized_pnl for p in open_positions]
        pairs = set(p.pair for p in open_positions)
        
        return {
            'total_exposure': sum(exposures),
            'avg_position_size': sum(exposures) / len(exposures),
            'max_position_pnl': max(pnls),
            'min_position_pnl': min(pnls),
            'num_pairs': float(len(pairs)),
        }


@dataclass
@sf_component(name='drawdown_metrics')
class DrawdownMetrics(StrategyMetric):
    """Drawdown tracking metrics.
    
    Tracks:
    - peak_value: Maximum portfolio value seen
    - current_drawdown: Current drawdown from peak
    - max_drawdown: Maximum drawdown ever seen
    
    Stores state in runtime dict.
    """
    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_METRIC
    
    def compute(
        self,
        *,
        state: StrategyState,
        context: StrategyContext,
    ) -> dict[str, float]:
        """Compute drawdown metrics."""
        # Current portfolio value
        open_positions = state.portfolio.open_positions()
        all_positions = state.portfolio.positions.values()
        
        unrealized = sum(p.unrealized_pnl for p in open_positions)
        realized = sum(p.realized_pnl for p in all_positions)
        fees = sum(p.fees_paid for p in all_positions)
        current_value = unrealized + realized - fees
        
        # Get tracked state
        peak = state.runtime.get('_peak_value', current_value)
        max_dd = state.runtime.get('_max_drawdown', 0.0)
        
        # Update peak
        if current_value > peak:
            peak = current_value
            state.runtime['_peak_value'] = peak
        
        # Calculate current drawdown
        if peak > 0:
            current_dd = (peak - current_value) / peak
        else:
            current_dd = 0.0
        
        # Update max drawdown
        if current_dd > max_dd:
            max_dd = current_dd
            state.runtime['_max_drawdown'] = max_dd
        
        return {
            'peak_value': float(peak),
            'current_drawdown': float(current_dd),
            'max_drawdown': float(max_dd),
        }


@dataclass
@sf_component(name='win_rate_metrics')
class WinRateMetrics(StrategyMetric):
    """Win rate and trade statistics.
    
    Computes:
    - total_trades: Number of closed positions
    - winning_trades: Number of profitable trades
    - losing_trades: Number of losing trades
    - win_rate: Winning / Total
    - avg_win: Average profit on winning trades
    - avg_loss: Average loss on losing trades
    - profit_factor: Total wins / Total losses
    """
    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_METRIC
    
    def compute(
        self,
        *,
        state: StrategyState,
        context: StrategyContext,
    ) -> dict[str, float]:
        """Compute win rate metrics."""
        closed = [p for p in state.portfolio.positions.values() if p.is_closed]
        
        if not closed:
            return {
                'total_trades': 0.0,
                'winning_trades': 0.0,
                'losing_trades': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
            }
        
        winners = [p for p in closed if p.realized_pnl > 0]
        losers = [p for p in closed if p.realized_pnl <= 0]
        
        total_wins = sum(p.realized_pnl for p in winners)
        total_losses = abs(sum(p.realized_pnl for p in losers))
        
        return {
            'total_trades': float(len(closed)),
            'winning_trades': float(len(winners)),
            'losing_trades': float(len(losers)),
            'win_rate': len(winners) / len(closed) if closed else 0.0,
            'avg_win': total_wins / len(winners) if winners else 0.0,
            'avg_loss': total_losses / len(losers) if losers else 0.0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
        }