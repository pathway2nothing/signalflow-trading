"""LLMStrategy + an open OpenAI-compatible client.

An LLM is just another StrategyModel: structured context in, intents out. It
never sees raw candles, only ``Observation.to_prompt_context()``. On any client
failure it falls back to a deterministic RulesStrategy.
"""

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, Field

from signalflow.decorators import strategy
from signalflow.engine.types import Intent
from signalflow.enums import IntentKind, Side
from signalflow.errors import KillSwitchTripped
from signalflow.strategy.base import Strategy, build_strategy
from signalflow.strategy.observation import Observation
from signalflow.strategy.rules import RulesStrategy

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.1"
LLM_API_KEY_ENV = "SIGNALFLOW_LLM_API_KEY"


class Decision(BaseModel):
    """One per-pair decision."""

    pair: str
    action: str = Field(description="one of: open, close, hold")
    size_pct: float = Field(default=0.0, description="fraction of equity for an open")


class Decisions(BaseModel):
    """The model's full set of per-pair decisions for one bar."""

    decisions: list[Decision] = Field(default_factory=list)


DECISIONS_SCHEMA = Decisions.model_json_schema()


@runtime_checkable
class LLMClient(Protocol):
    def decide(self, context: dict, schema: dict) -> "dict | None": ...


def _system_prompt(mandate: str) -> str:
    """Build the instruction prompt embedding the mandate and decision schema."""
    return (
        "You are a disciplined trading strategy. Each turn you receive a JSON "
        "context with the current timestamp, your portfolio (equity + open "
        "positions), and a table of validated signals (pair, signal, p_success). "
        "You NEVER see raw price candles. Decide an action per pair.\n\n"
        f"MANDATE:\n{mandate}\n\n"
        "Reply with ONLY a JSON object matching this schema (one entry per pair "
        "you act on); action is 'open' | 'close' | 'hold', size_pct is the "
        "fraction of equity for an 'open':\n"
        f"{json.dumps(DECISIONS_SCHEMA)}"
    )


@dataclass
class OpenAICompatClient:
    """Chat client for open OpenAI-compatible servers (Ollama, vLLM, LM Studio, TGI)."""

    base_url: str = ""
    model: str = ""
    api_key: str = field(default="not-needed", repr=False)
    max_tokens: int = 1024
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url or os.environ.get("SIGNALFLOW_LLM_BASE_URL", DEFAULT_BASE_URL)
        self.model = self.model or os.environ.get("SIGNALFLOW_LLM_MODEL", DEFAULT_MODEL)

    def decide(self, context: dict, schema: dict) -> "dict | None":
        """Call the chat endpoint and parse a JSON decisions object; ``None`` (logged) on failure."""
        try:
            import httpx

            mandate = context.get("mandate", "")
            mandate_txt = mandate if isinstance(mandate, str) else json.dumps(mandate)
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _system_prompt(mandate_txt)},
                    {"role": "user", "content": json.dumps(context)},
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": self.max_tokens,
                "temperature": 0,
            }
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            resp = httpx.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as exc:
            logger.warning(f"LLM client {self.model!r} at {self.base_url}: {type(exc).__name__}: {exc}")
            return None


@strategy("llm")
@dataclass
class LLMStrategy(Strategy):
    """Turn an LLM's structured decisions into intents.

    When the client fails or returns something that does not validate, the
    strategy falls back to ``fallback`` (a ``RulesStrategy`` by default) and logs
    a WARNING each time; ``fallbacks`` counts them and the live loop reports the
    count in ``Run.meta``. With ``fallback=None`` a failure raises
    :class:`KillSwitchTripped` instead, so an armed flow stops rather than
    silently trading rules it was not configured to trade.
    """

    client: LLMClient
    mandate: str = ""
    fallback: object = field(default_factory=RulesStrategy)
    cache_size: int = 10_000
    fallbacks: int = field(default=0, init=False, repr=False)
    _cache: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)

    def to_config(self) -> dict:
        """Serialize to a portable config; the client's API key is never written."""
        client = self.client
        client_cfg = {
            "base_url": getattr(client, "base_url", ""),
            "model": getattr(client, "model", ""),
            "max_tokens": getattr(client, "max_tokens", 1024),
            "timeout": getattr(client, "timeout", 60.0),
        }
        fallback_cfg = self.fallback.to_config() if isinstance(self.fallback, Strategy) else None
        return {
            "name": self.name,
            "params": {
                "client": client_cfg,
                "mandate": self.mandate,
                "fallback": fallback_cfg,
                "cache_size": self.cache_size,
            },
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "LLMStrategy":
        """Rebuild an OpenAI-compatible client from params, reading the API key from env."""
        params = cfg.get("params") or {}
        client_cfg = params.get("client") or {}
        client = OpenAICompatClient(
            base_url=client_cfg.get("base_url", ""),
            model=client_cfg.get("model", ""),
            api_key=os.environ.get(LLM_API_KEY_ENV, "not-needed"),
            max_tokens=client_cfg.get("max_tokens", 1024),
            timeout=client_cfg.get("timeout", 60.0),
        )
        fallback_cfg = params.get("fallback")
        fallback = build_strategy(fallback_cfg) if fallback_cfg else None
        return cls(
            client=client,
            mandate=params.get("mandate", ""),
            fallback=fallback,
            cache_size=int(params.get("cache_size", 10_000)),
        )

    def decide(self, obs: Observation) -> list[Intent]:
        context = obs.to_prompt_context()
        context["mandate"] = self.mandate

        key = (str(obs.ts), self._hash_context(context))
        if key in self._cache:
            raw = self._cache[key]
        else:
            try:
                raw = self.client.decide(context, DECISIONS_SCHEMA)
            except Exception as exc:
                logger.warning(f"LLMStrategy: client raised {type(exc).__name__}: {exc}")
                raw = None
            self._cache[key] = raw
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        if raw is None:
            return self._fall_back(obs, "no decision from the client")
        try:
            decisions = Decisions.model_validate(raw)
        except Exception as exc:
            return self._fall_back(obs, f"invalid decision payload ({type(exc).__name__})")
        return self._to_intents(decisions, obs)

    def _fall_back(self, obs: Observation, reason: str) -> list[Intent]:
        self.fallbacks += 1
        if self.fallback is None:
            raise KillSwitchTripped(f"LLMStrategy at {obs.ts}: {reason} and no fallback strategy is configured")
        name = getattr(self.fallback, "name", type(self.fallback).__name__)
        logger.warning(f"LLMStrategy at {obs.ts}: {reason}; falling back to {name!r} (fallback #{self.fallbacks})")
        return self.fallback.decide(obs)

    @staticmethod
    def _hash_context(context: dict) -> int:
        return hash(json.dumps(context, sort_keys=True, default=str))

    def _to_intents(self, decisions: Decisions, obs: Observation) -> list[Intent]:
        port = obs.portfolio
        held = port.positions
        intents: list[Intent] = []
        for d in decisions.decisions:
            action = (d.action or "").lower()
            if action == "open" and d.pair not in held:
                notional = (d.size_pct or 0.0) * port.equity
                if notional > 0:
                    intents.append(Intent(d.pair, IntentKind.OPEN, Side.BUY, notional=notional, reason="llm_open"))
            elif action == "close":
                pos = held.get(d.pair)
                if pos is not None and pos.qty > 0:
                    intents.append(Intent(d.pair, IntentKind.CLOSE, Side.SELL, qty=pos.qty, reason="llm_close"))
        return intents
