"""LLMStrategy + an open OpenAI-compatible client.

An LLM is just another StrategyModel: structured context in, intents out. It
never sees raw candles, only ``Observation.to_prompt_context()``. On any client
failure it falls back to a deterministic RulesStrategy.
"""


import json
import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from signalflow.decorators import strategy
from signalflow.engine.types import Intent
from signalflow.enums import IntentKind, Side
from signalflow.strategy.observation import Observation
from signalflow.strategy.rules import RulesStrategy

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.1"


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
    api_key: str = "not-needed"
    max_tokens: int = 1024
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url or os.environ.get("SIGNALFLOW_LLM_BASE_URL", DEFAULT_BASE_URL)
        self.model = self.model or os.environ.get("SIGNALFLOW_LLM_MODEL", DEFAULT_MODEL)

    def decide(self, context: dict, schema: dict) -> "dict | None":
        """Call the chat endpoint and parse a JSON decisions object; None on failure."""
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
            resp = httpx.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            return None


@strategy("llm")
@dataclass
class LLMStrategy:
    """Turn an LLM's structured decisions into intents, with a rules fallback."""

    client: LLMClient
    mandate: str = ""
    fallback: object = field(default_factory=RulesStrategy)
    thinking: str = "low"
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def decide(self, obs: Observation) -> list[Intent]:
        context = obs.to_prompt_context()
        context["mandate"] = self.mandate

        key = (str(obs.ts), self._hash_context(context))
        if key in self._cache:
            raw = self._cache[key]
        else:
            try:
                raw = self.client.decide(context, DECISIONS_SCHEMA)
            except Exception:
                raw = None
            self._cache[key] = raw

        if raw is None:
            return self.fallback.decide(obs)
        try:
            decisions = Decisions.model_validate(raw)
        except Exception:
            return self.fallback.decide(obs)
        return self._to_intents(decisions, obs)

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
