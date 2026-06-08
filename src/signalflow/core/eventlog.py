"""Event log: the canonical source of truth for portfolio state.

The invariant "portfolio changes ONLY through fills" makes this an
event-sourced system by definition: a fill is an event, the portfolio is the
fold of the event stream. This module materialises that fold.

Design (REFACTOR_PLAN.md §1, §4.3):
    - The canonical event is :class:`~signalflow.core.containers.trade.Trade`
      (already frozen, already persisted, ``Trade.id == fill.id``).
    - ``apply_fill`` is the SINGLE place where a fill mutates the portfolio
      (find-or-create position + ``position.apply_trade`` + cash accounting).
      Brokers delegate here instead of each re-implementing position math.
    - ``fold`` replays an ordered event stream into a ``Portfolio`` — the
      formal ``state = fold(events)``.
    - The persisted snapshot becomes a CACHE, verifiable by replay
      (``portfolios_match`` compares the trade-derived fields).

Cash accounting is parameterised by :class:`CashPolicy` rather than duplicated
across broker subclasses (``track_cash=False`` is the "unlimited" mode).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from signalflow.core.containers.portfolio import Portfolio
from signalflow.core.containers.position import Position
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.containers.trade import Trade
from signalflow.core.enums import PositionType

# Control keys carried in ``Trade.meta`` that are NOT part of position.meta.
_CONTROL_META_KEYS = ("type", "signal_strength")

# Position fields reconstructable from the event stream alone. ``last_price`` /
# ``last_time`` are excluded: they come from mark-to-market, which is a derived
# view, not an event — so the fold cannot (and need not) reproduce them.
_REPLAY_NUMERIC_FIELDS = ("qty", "entry_price", "fees_paid", "realized_pnl")


@dataclass(frozen=True, slots=True)
class CashPolicy:
    """Declares how a fill mutates portfolio cash.

    Attributes:
        track_cash: When True (default), BUY debits ``notional + fee`` and SELL
            credits ``notional - fee``. When False, cash is left untouched
            (the "unlimited" / balance-agnostic backtest mode).
    """

    track_cash: bool = True


DEFAULT_CASH_POLICY = CashPolicy()


def _position_from_trade(trade: Trade) -> Position:
    """Build a fresh Position from an entry trade (mirrors the old broker logic).

    Reconstructs ``signal_strength`` and ``meta`` from the trade's metadata so
    that a fold over the event stream yields the exact same position the broker
    created live.
    """
    position_type = PositionType.LONG if trade.side == "BUY" else PositionType.SHORT
    signal_strength = float(trade.meta.get("signal_strength", 1.0))
    meta = {k: v for k, v in trade.meta.items() if k not in _CONTROL_META_KEYS}
    return Position(
        id=trade.position_id or str(uuid.uuid4()),
        is_closed=False,
        pair=trade.pair,
        position_type=position_type,
        signal_strength=signal_strength,
        entry_time=trade.ts,
        last_time=trade.ts,
        entry_price=trade.price,
        last_price=trade.price,
        qty=trade.qty,
        fees_paid=trade.fee,
        realized_pnl=0.0,
        meta=meta,
    )


def _apply_cash(portfolio: Portfolio, trade: Trade, policy: CashPolicy) -> None:
    if not policy.track_cash:
        return
    notional = trade.notional
    if trade.side == "SELL":
        portfolio.cash += notional - trade.fee
    else:  # BUY
        portfolio.cash -= notional + trade.fee


def apply_fill(
    portfolio: Portfolio,
    trade: Trade,
    *,
    policy: CashPolicy = DEFAULT_CASH_POLICY,
) -> Position:
    """Fold ONE event into the portfolio. The only home of fill->position logic.

    If ``trade.position_id`` already names a position, the trade is applied to it
    (increase / partial / full close). Otherwise a new position is created. Cash
    is adjusted per ``policy``.

    Args:
        portfolio: Portfolio to mutate in place.
        trade: The canonical event to apply.
        policy: Cash-accounting policy.

    Returns:
        The position affected by this event.
    """
    pid = trade.position_id
    if pid is not None and pid in portfolio.positions:
        position = portfolio.positions[pid]
        position.apply_trade(trade)
    else:
        position = _position_from_trade(trade)
        portfolio.positions[position.id] = position

    _apply_cash(portfolio, trade, policy)
    return position


def fold(
    events: Iterable[Trade],
    *,
    initial_cash: float = 0.0,
    policy: CashPolicy = DEFAULT_CASH_POLICY,
) -> Portfolio:
    """Deterministically replay an ordered event stream into a Portfolio.

    Events MUST be supplied in chronological (execution) order; the store's
    ``read_trades`` returns them ordered by ``(ts, trade_id)``.
    """
    portfolio = Portfolio(cash=initial_cash)
    for trade in events:
        apply_fill(portfolio, trade, policy=policy)
    return portfolio


def replay_state(
    strategy_id: str,
    events: Iterable[Trade],
    *,
    initial_cash: float = 0.0,
    policy: CashPolicy = DEFAULT_CASH_POLICY,
) -> StrategyState:
    """Replay the event stream into a StrategyState (portfolio only).

    Metrics and runtime are derived, not events, so they are left empty.
    """
    state = StrategyState(strategy_id=strategy_id)
    state.portfolio = fold(events, initial_cash=initial_cash, policy=policy)
    return state


def portfolios_match(a: Portfolio, b: Portfolio, *, tol: float = 1e-9) -> bool:
    """True if two portfolios agree on all trade-derived (foldable) fields.

    Compares cash and, per position, the fields that the event stream fully
    determines. Mark-to-market fields (``last_price`` / ``last_time``) are
    intentionally ignored — they are a derived view, not part of the log.
    """
    if abs(a.cash - b.cash) > tol:
        return False
    if set(a.positions) != set(b.positions):
        return False
    for pid, pa in a.positions.items():
        pb = b.positions[pid]
        if pa.pair != pb.pair or pa.position_type != pb.position_type or pa.is_closed != pb.is_closed:
            return False
        for f in _REPLAY_NUMERIC_FIELDS:
            if abs(float(getattr(pa, f)) - float(getattr(pb, f))) > tol:
                return False
    return True
