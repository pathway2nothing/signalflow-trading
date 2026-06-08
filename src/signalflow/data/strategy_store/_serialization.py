"""Shared JSON serialization helpers for strategy stores."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from signalflow.core import Portfolio, Position, PositionType, StrategyState, Trade


def to_json(obj: object) -> str:
    """Convert object to JSON string.

    Handles dataclasses by converting to dict first. Uses default=str
    for non-serializable types (e.g., datetime).
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    return json.dumps(obj, default=str, ensure_ascii=False)


def _parse_dt(value: Any) -> datetime | None:
    """Parse a datetime that ``to_json`` serialized via ``str(datetime)``."""
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return value


def _position_from_dict(data: dict[str, Any]) -> Position:
    data = dict(data)
    pt = data.get("position_type")
    if isinstance(pt, str):
        data["position_type"] = PositionType(pt)
    data["entry_time"] = _parse_dt(data.get("entry_time"))
    data["last_time"] = _parse_dt(data.get("last_time"))
    return Position(**data)


def _portfolio_from_dict(data: dict[str, Any]) -> Portfolio:
    positions = {pid: _position_from_dict(pd) for pid, pd in data.get("positions", {}).items()}
    return Portfolio(cash=data.get("cash", 0.0), positions=positions)


def state_from_json(payload: str) -> StrategyState:
    """Deserialize StrategyState from JSON, reconstructing nested dataclasses.

    The snapshot is a derived cache of the event log; deserialization must
    rebuild the canonical ``Portfolio``/``Position`` objects (not leave them as
    dicts) so the snapshot can be compared against an event-log replay.
    """
    data = json.loads(payload)

    data["last_ts"] = _parse_dt(data.get("last_ts"))

    portfolio = data.get("portfolio")
    if isinstance(portfolio, dict):
        data["portfolio"] = _portfolio_from_dict(portfolio)

    # metrics_phase_done is a tick-local cache; sets do not round-trip through
    # JSON, so coerce whatever survived back into a set.
    mpd = data.get("metrics_phase_done")
    data["metrics_phase_done"] = set(mpd) if isinstance(mpd, list) else set()

    return StrategyState(**data)


def trade_from_json(payload: str) -> Trade:
    """Deserialize a Trade from JSON (inverse of ``to_json``)."""
    data = json.loads(payload)
    data["ts"] = _parse_dt(data.get("ts"))
    return Trade(**data)
