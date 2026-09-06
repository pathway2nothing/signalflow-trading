"""Exchange lot/tick quantization shared by the simulated and the live brokers.

Binance-style filters: ``stepSize`` (quantity step), ``tickSize`` (price step),
``minQty`` and ``minNotional``. The same arithmetic is used in the backtest so
paper and armed runs size orders identically.
"""

import decimal
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolFilters:
    """Order-size limits of one symbol; ``None`` steps mean 'no quantization'."""

    step_size: "str | float | None" = None
    tick_size: "str | float | None" = None
    min_qty: float = 0.0
    min_notional: float = 0.0

    @classmethod
    def from_binance(cls, info: dict) -> "SymbolFilters":
        """Parse an ``exchangeInfo`` payload (one symbol) into filters."""
        symbols = info.get("symbols") or []
        if not symbols:
            return cls()
        by_type = {f.get("filterType"): f for f in symbols[0].get("filters", [])}
        lot = by_type.get("LOT_SIZE", {})
        price = by_type.get("PRICE_FILTER", {})
        notional = by_type.get("NOTIONAL") or by_type.get("MIN_NOTIONAL") or {}
        return cls(
            step_size=lot.get("stepSize"),
            tick_size=price.get("tickSize"),
            min_qty=float(lot.get("minQty", 0.0) or 0.0),
            min_notional=float(notional.get("minNotional", 0.0) or 0.0),
        )

    @classmethod
    def from_mapping(cls, raw: "dict | SymbolFilters | None") -> "SymbolFilters":
        if raw is None:
            return cls()
        if isinstance(raw, SymbolFilters):
            return raw
        return cls(
            step_size=raw.get("stepSize", raw.get("step_size")),
            tick_size=raw.get("tickSize", raw.get("tick_size")),
            min_qty=float(raw.get("minQty", raw.get("min_qty", 0.0)) or 0.0),
            min_notional=float(raw.get("minNotional", raw.get("min_notional", 0.0)) or 0.0),
        )


def floor_step(value: float, step: "str | float | None") -> decimal.Decimal:
    """Floor ``value`` to a multiple of ``step`` (quantities never round up)."""
    d = decimal.Decimal(str(value))
    if not step:
        return d
    s = decimal.Decimal(str(step))
    return (d / s).to_integral_value(rounding=decimal.ROUND_DOWN) * s


def round_tick(value: float, tick: "str | float | None") -> decimal.Decimal:
    """Round ``value`` to the nearest multiple of ``tick`` (limit prices)."""
    d = decimal.Decimal(str(value))
    if not tick:
        return d
    t = decimal.Decimal(str(tick))
    return (d / t).to_integral_value(rounding=decimal.ROUND_HALF_UP) * t


def quantize_order(qty: float, price: float, limit_price: "float | None", filters: SymbolFilters):
    """``(qty, limit_price, ok)`` after the symbol's filters; ``ok`` is False below the minimums."""
    q = floor_step(qty, filters.step_size)
    lp = round_tick(limit_price, filters.tick_size) if limit_price is not None else None
    ref = float(lp) if lp is not None else float(price)
    notional = float(q) * ref
    ok = float(q) >= filters.min_qty and float(q) > 0 and not (filters.min_notional and notional < filters.min_notional)
    return float(q), (float(lp) if lp is not None else None), ok


__all__ = ["SymbolFilters", "floor_step", "quantize_order", "round_tick"]
