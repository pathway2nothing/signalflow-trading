"""Shared JSON serialization helpers for strategy stores."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass

from signalflow.core import StrategyState


def to_json(obj: object) -> str:
    """Convert object to JSON string.

    Handles dataclasses by converting to dict first. Uses default=str
    for non-serializable types (e.g., datetime).
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    return json.dumps(obj, default=str, ensure_ascii=False)


def state_from_json(payload: str) -> StrategyState:
    """Deserialize StrategyState from JSON string."""
    data = json.loads(payload)
    return StrategyState(**data)
