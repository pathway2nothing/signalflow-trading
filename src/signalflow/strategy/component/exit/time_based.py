from dataclasses import dataclass
from signalflow.core import Position, Order, StrategyState, PositionType, sf_component
from signalflow.strategy.component.base import ExitRule

@dataclass
@sf_component(name='time_exit')
class TimeBasedExit(ExitRule):
    """Exit positions after a fixed holding period."""
    
    max_bars: int = 60  
    bar_col: str = 'bar_count' 
    
    def check_exits(
        self,
        positions: list[Position],
        prices: dict[str, float],
        state: StrategyState
    ) -> list[Order]:
        orders: list[Order] = []
        
        for pos in positions:
            if pos.is_closed:
                continue
                
            price = prices.get(pos.pair)
            if price is None:
                continue
            
            bar_count = pos.meta.get(self.bar_col, 0) + 1
            pos.meta[self.bar_col] = bar_count
            
            if bar_count >= self.max_bars:
                side = 'SELL' if pos.position_type == PositionType.LONG else 'BUY'
                order = Order(
                    pair=pos.pair,
                    side=side,
                    order_type='MARKET',
                    qty=pos.qty,
                    position_id=pos.id,
                    meta={
                        'exit_reason': 'time_exit',
                        'bars_held': bar_count,
                    }
                )
                orders.append(order)
        
        return orders
