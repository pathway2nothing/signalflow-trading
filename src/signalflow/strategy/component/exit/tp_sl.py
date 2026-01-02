"""Exit rules for strategy."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from signalflow.core import sf_component
from signalflow.core.enums import SfComponentType
from signalflow.core.containers import Position
from signalflow.strategy.component.base import StrategyExitRule
from signalflow.strategy.state import StrategyState
from signalflow.strategy.types import StrategyContext


@dataclass
@sf_component(name='tp_sl')
class TakeProfitStopLossExit(StrategyExitRule):
    """Take profit and stop loss exit rule.
    
    Closes position when:
    - Profit reaches tp threshold (take profit)
    - Loss reaches sl threshold (stop loss)
    
    Attributes:
        tp: Take profit threshold as fraction (e.g., 0.02 = 2%)
        sl: Stop loss threshold as fraction (e.g., 0.01 = 1%)
    """
    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_EXIT_RULE
    
    tp: float = 0.02
    sl: float = 0.01
    
    def should_exit(
        self,
        *,
        position: Position,
        state: StrategyState,
        context: StrategyContext,
    ) -> tuple[bool, str]:
        """Check if position should be closed."""
        if position.qty <= 0 or position.entry_price <= 0:
            return False, ''
        
        # Get current price
        price = context.prices.get(position.pair, position.last_price)
        
        # Calculate return (considering position direction)
        ret = position.side_sign * (price - position.entry_price) / position.entry_price
        
        if ret >= self.tp:
            return True, 'TP'
        
        if ret <= -self.sl:
            return True, 'SL'
        
        return False, ''


@dataclass
@sf_component(name='time_exit')
class TimeBasedExit(StrategyExitRule):
    """Time-based exit rule.
    
    Closes position after maximum holding time.
    
    Attributes:
        max_minutes: Maximum holding time in minutes
    """
    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_EXIT_RULE
    
    max_minutes: int = 1440  # 24 hours default
    
    def should_exit(
        self,
        *,
        position: Position,
        state: StrategyState,
        context: StrategyContext,
    ) -> tuple[bool, str]:
        """Check if position exceeded max holding time."""
        if position.entry_time is None:
            return False, ''
        
        duration = (context.ts - position.entry_time).total_seconds() / 60
        
        if duration >= self.max_minutes:
            return True, 'TIME'
        
        return False, ''


@dataclass
@sf_component(name='pnl_velocity_exit')
class PnLVelocityExit(StrategyExitRule):
    """Exit based on PnL change velocity.
    
    Closes position if PnL changes too quickly (both up and down).
    Useful for detecting unusual market conditions.
    
    Requires metrics: pnl_velocity_{position_id} or similar tracking.
    
    Attributes:
        max_pnl_change_pct: Maximum PnL change per interval (e.g., 0.02 = 2%)
        check_interval_minutes: Interval for velocity calculation
    """
    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_EXIT_RULE
    
    max_pnl_change_pct: float = 0.02
    check_interval_minutes: int = 5
    
    def should_exit(
        self,
        *,
        position: Position,
        state: StrategyState,
        context: StrategyContext,
    ) -> tuple[bool, str]:
        """Check if PnL is changing too fast."""
        # This requires tracking previous PnL in runtime state
        # Key format: pnl_history_{position_id}
        history_key = f'pnl_history_{position.id}'
        
        pnl_history: list[tuple] = state.runtime.get(history_key, [])
        
        current_pnl_pct = 0.0
        if position.entry_price > 0 and position.qty > 0:
            price = context.prices.get(position.pair, position.last_price)
            current_pnl_pct = position.side_sign * (price - position.entry_price) / position.entry_price
        
        # Add current to history
        pnl_history.append((context.ts, current_pnl_pct))
        
        # Keep only relevant history (last N minutes)
        cutoff = context.ts.timestamp() - self.check_interval_minutes * 60
        pnl_history = [(ts, pnl) for ts, pnl in pnl_history if ts.timestamp() > cutoff]
        state.runtime[history_key] = pnl_history
        
        # Check velocity
        if len(pnl_history) >= 2:
            oldest_pnl = pnl_history[0][1]
            pnl_change = abs(current_pnl_pct - oldest_pnl)
            
            if pnl_change >= self.max_pnl_change_pct:
                return True, 'PNL_VELOCITY'
        
        return False, ''


@dataclass
@sf_component(name='trailing_tp')
class TrailingTakeProfitExit(StrategyExitRule):
    """Trailing take profit exit.
    
    Tracks maximum profit and exits when price retraces by threshold.
    
    Attributes:
        activation_pct: Profit level to activate trailing (e.g., 0.01 = 1%)
        trail_pct: Retrace from max to trigger exit (e.g., 0.005 = 0.5%)
    """
    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_EXIT_RULE
    
    activation_pct: float = 0.01
    trail_pct: float = 0.005
    
    def should_exit(
        self,
        *,
        position: Position,
        state: StrategyState,
        context: StrategyContext,
    ) -> tuple[bool, str]:
        """Check trailing take profit condition."""
        if position.entry_price <= 0:
            return False, ''
        
        price = context.prices.get(position.pair, position.last_price)
        current_pnl_pct = position.side_sign * (price - position.entry_price) / position.entry_price
        
        # Track max PnL
        max_key = f'max_pnl_{position.id}'
        max_pnl = state.runtime.get(max_key, 0.0)
        
        if current_pnl_pct > max_pnl:
            max_pnl = current_pnl_pct
            state.runtime[max_key] = max_pnl
        
        # Check if activated and retraced
        if max_pnl >= self.activation_pct:
            retrace = max_pnl - current_pnl_pct
            if retrace >= self.trail_pct:
                return True, 'TRAILING_TP'
        
        return False, ''