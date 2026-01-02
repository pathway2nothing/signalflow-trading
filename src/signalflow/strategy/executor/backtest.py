
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
import uuid

from signalflow.core import Position, Trade
from signalflow.strategy.types import NewPositionOrder, ClosePositionOrder
from signalflow.strategy.order_executor.base import OrderExecutor


@dataclass
class BacktestExecutor(OrderExecutor):
    """Backtest executor with instant fills at order price.
    
    Attributes:
        fee_rate: Trading fee as fraction of notional (e.g., 0.001 = 0.1%)
    """
    fee_rate: float = 0.001  
    
    def execute_entry(
        self,
        order: NewPositionOrder,
    ) -> tuple[Position, Trade]:
        """Execute entry order with instant fill.
        
        Args:
            order: Entry order with pair, qty, price
            
        Returns:
            Tuple of (new Position, entry Trade)
        """
        position_id = str(uuid.uuid4())
        trade_id = str(uuid.uuid4())
        
        notional = order.price * order.qty
        fee = notional * self.fee_rate
        
        position = Position(
            id=position_id,
            is_closed=False,
            pair=order.pair,
            position_type=order.position_type,
            signal_strength=order.signal_strength,
            entry_time=order.ts,
            last_time=order.ts,
            entry_price=order.price,
            last_price=order.price,
            qty=order.qty,
            fees_paid=fee,
            realized_pnl=0.0,
            meta=dict(order.meta) if order.meta else {},
        )
        
        trade = Trade(
            id=trade_id,
            position_id=position_id,
            pair=order.pair,
            side='BUY',
            ts=order.ts,
            price=order.price,
            qty=order.qty,
            fee=fee,
            meta={'order_type': 'entry'},
        )
        
        return position, trade
    
    def execute_exit(
        self,
        position: Position,
        order: ClosePositionOrder,
    ) -> Trade:
        """Execute exit order with instant fill.
        
        Updates position state (closes it) and returns exit trade.
        
        Args:
            position: Position to close
            order: Exit order with price and reason
            
        Returns:
            Exit Trade
        """
        trade_id = str(uuid.uuid4())
        
        notional = order.price * position.qty
        fee = notional * self.fee_rate
        
        # Calculate realized PnL
        pnl = position.side_sign * (order.price - position.entry_price) * position.qty
        
        # Update position (mutable)
        position.last_time = order.ts
        position.last_price = order.price
        position.fees_paid += fee
        position.realized_pnl = pnl
        position.is_closed = True
        position.meta['exit_reason'] = order.reason
        position.meta['exit_time'] = order.ts
        position.meta['exit_price'] = order.price
        
        trade = Trade(
            id=trade_id,
            position_id=position.id,
            pair=position.pair,
            side='SELL',
            ts=order.ts,
            price=order.price,
            qty=position.qty,
            fee=fee,
            meta={'order_type': 'exit', 'reason': order.reason},
        )
        
        return trade


