from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

TradeSide = Literal["BUY", "SELL"]


@dataclass(frozen=True, slots=True)
class Trade:
    """Immutable domain event representing an executed trade.

    A Trade is a fact: it happened, it cannot be changed.
    All position accounting flows from trades.

    Trade → Position relationship:
        - Trades are immutable events
        - Position state is derived from applying trades
        - One position can have multiple trades (entry, partial exits, full exit)

    Attributes:
        id (str): Unique trade identifier.
        position_id (str | None): ID of the position this trade belongs to.
        pair (str): Trading pair (e.g. "BTCUSDT").
        side (TradeSide): Trade side - "BUY" or "SELL".
        ts (datetime | None): Execution timestamp.
        price (float): Execution price.
        qty (float): Executed quantity (always positive).
        fee (float): Transaction fee paid (always positive).
        meta (dict[str, Any]): Additional metadata (e.g., order_id, fill_type).

    Example:
        ```python
        from signalflow.core import Trade
        from datetime import datetime

        # Entry trade
        entry = Trade(
            position_id="pos_123",
            pair="BTCUSDT",
            side="BUY",
            ts=datetime.now(),
            price=45000.0,
            qty=0.5,
            fee=22.5,
            meta={"type": "entry", "signal": "sma_cross"}
        )

        # Exit trade
        exit = Trade(
            position_id="pos_123",
            pair="BTCUSDT",
            side="SELL",
            ts=datetime.now(),
            price=46000.0,
            qty=0.5,
            fee=23.0,
            meta={"type": "exit", "reason": "take_profit"}
        )

        # Calculate PnL
        pnl = (exit.price - entry.price) * entry.qty
        total_fees = entry.fee + exit.fee
        net_pnl = pnl - total_fees
        print(f"Net PnL: ${net_pnl:.2f}")

        # Check notional value
        print(f"Entry notional: ${entry.notional:.2f}")
        print(f"Exit notional: ${exit.notional:.2f}")
        ```

    Note:
        Trades are immutable events. Position state is derived from trades.
        qty and fee are always positive regardless of side.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    position_id: str | None = None

    pair: str = ""
    side: TradeSide = "BUY"
    ts: datetime | None = None

    price: float = 0.0
    qty: float = 0.0
    fee: float = 0.0

    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        """Calculate notional value of the trade.

        Notional = price * qty

        Returns:
            float: Notional value in currency units.

        Example:
            ```python
            trade = Trade(price=45000.0, qty=0.5)
            assert trade.notional == 22500.0  # 45000 * 0.5

            # Track total volume
            trades = [trade1, trade2, trade3]
            total_volume = sum(t.notional for t in trades)
            ```
        """
        return float(self.price) * float(self.qty)
