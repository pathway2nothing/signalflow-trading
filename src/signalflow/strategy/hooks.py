"""Event hooks system for live trading.

This module implements the HOOKS block from SFFLOW.md specification:
- on_entry: Triggered when position is opened
- on_exit: Triggered when position is closed
- on_error: Triggered on errors

Hooks support multiple notification backends:
- telegram: Telegram notifications
- slack: Slack notifications
- webhook: Generic HTTP webhooks
- log: Logging only

Example:
    >>> from signalflow.strategy.hooks import HooksManager, HookEvent
    >>> hooks = HooksManager.from_config({
    ...     "on_entry": [{"type": "telegram", "template": "Entry: {pair} @ {price}"}],
    ...     "on_exit": [{"type": "telegram", "template": "Exit: {pair} PnL: {pnl_pct}%"}],
    ... })
    >>> await hooks.trigger(HookEvent.ON_ENTRY, {"pair": "BTCUSDT", "price": 50000})
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from loguru import logger

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


class HookEvent(StrEnum):
    """Hook event types."""

    ON_ENTRY = "on_entry"
    ON_EXIT = "on_exit"
    ON_ERROR = "on_error"
    ON_SIGNAL = "on_signal"
    ON_RISK_LIMIT = "on_risk_limit"
    ON_CIRCUIT_BREAKER = "on_circuit_breaker"


class HookPriority(StrEnum):
    """Hook priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HookConfig:
    """Hook handler configuration.

    Attributes:
        type: Handler type (telegram, slack, webhook, log)
        template: Message template with {placeholders}
        priority: Message priority
        enabled: Whether hook is enabled
        params: Handler-specific parameters
    """

    type: str
    template: str = ""
    priority: HookPriority = HookPriority.NORMAL
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HookConfig:
        """Create from dict."""
        priority_str = data.get("priority", "normal")
        return cls(
            type=data.get("type", "log"),
            template=data.get("template", ""),
            priority=HookPriority(priority_str),
            enabled=data.get("enabled", True),
            params={k: v for k, v in data.items() if k not in {"type", "template", "priority", "enabled"}},
        )


class BaseHookHandler(ABC):
    """Base class for hook handlers."""

    def __init__(self, config: HookConfig):
        """Initialize handler.

        Args:
            config: Handler configuration
        """
        self.config = config

    @abstractmethod
    async def send(self, event: HookEvent, context: dict[str, Any]) -> bool:
        """Send notification.

        Args:
            event: Event type
            context: Event context data

        Returns:
            True if sent successfully
        """
        ...

    def format_message(self, context: dict[str, Any]) -> str:
        """Format message from template.

        Args:
            context: Context data for template

        Returns:
            Formatted message
        """
        if not self.config.template:
            return json.dumps(context, default=str)

        try:
            return self.config.template.format(**context)
        except KeyError as e:
            logger.warning(f"Missing template key: {e}")
            return self.config.template


class LogHookHandler(BaseHookHandler):
    """Log-only hook handler."""

    async def send(self, event: HookEvent, context: dict[str, Any]) -> bool:
        """Log the event."""
        message = self.format_message(context)
        level = {
            HookPriority.LOW: "DEBUG",
            HookPriority.NORMAL: "INFO",
            HookPriority.HIGH: "WARNING",
            HookPriority.CRITICAL: "ERROR",
        }.get(self.config.priority, "INFO")

        logger.log(level, f"[HOOK:{event}] {message}")
        return True


class TelegramHookHandler(BaseHookHandler):
    """Telegram notification handler."""

    def __init__(self, config: HookConfig):
        """Initialize with Telegram config.

        Args:
            config: Handler config with bot_token and chat_id in params
        """
        super().__init__(config)
        self.bot_token = config.params.get("bot_token")
        self.chat_id = config.params.get("chat_id")

    async def send(self, event: HookEvent, context: dict[str, Any]) -> bool:
        """Send Telegram message."""
        if httpx is None:
            logger.error("httpx is required for Telegram hooks: pip install httpx")
            return False
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram hook missing bot_token or chat_id")
            return False

        message = self.format_message(context)

        # Add emoji based on event/priority
        emoji = self._get_emoji(event)
        message = f"{emoji} {message}"

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json={
                        "chat_id": self.chat_id,
                        "text": message,
                        "parse_mode": "HTML",
                    },
                    timeout=10.0,
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def _get_emoji(self, event: HookEvent) -> str:
        """Get emoji for event type."""
        return {
            HookEvent.ON_ENTRY: "🟢",
            HookEvent.ON_EXIT: "🔴",
            HookEvent.ON_ERROR: "⚠️",
            HookEvent.ON_SIGNAL: "📊",
            HookEvent.ON_RISK_LIMIT: "🚨",
            HookEvent.ON_CIRCUIT_BREAKER: "🛑",
        }.get(event, "📬")


class SlackHookHandler(BaseHookHandler):
    """Slack notification handler."""

    def __init__(self, config: HookConfig):
        """Initialize with Slack config.

        Args:
            config: Handler config with webhook_url in params
        """
        super().__init__(config)
        self.webhook_url = config.params.get("webhook_url")

    async def send(self, event: HookEvent, context: dict[str, Any]) -> bool:
        """Send Slack message."""
        if httpx is None:
            logger.error("httpx is required for Slack hooks: pip install httpx")
            return False
        if not self.webhook_url:
            logger.warning("Slack hook missing webhook_url")
            return False

        message = self.format_message(context)

        # Build Slack payload
        color = {
            HookEvent.ON_ENTRY: "#36a64f",
            HookEvent.ON_EXIT: "#cc0000",
            HookEvent.ON_ERROR: "#ff9900",
        }.get(event, "#0099cc")

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"SF Flow: {event.value}",
                    "text": message,
                    "footer": "SignalFlow",
                }
            ]
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return False


class WebhookHookHandler(BaseHookHandler):
    """Generic HTTP webhook handler."""

    def __init__(self, config: HookConfig):
        """Initialize with webhook config.

        Args:
            config: Handler config with url, method, headers in params
        """
        super().__init__(config)
        self.url = config.params.get("url")
        self.method = config.params.get("method", "POST")
        self.headers = config.params.get("headers", {})

    async def send(self, event: HookEvent, context: dict[str, Any]) -> bool:
        """Send webhook request."""
        if httpx is None:
            logger.error("httpx is required for webhook hooks: pip install httpx")
            return False
        if not self.url:
            logger.warning("Webhook hook missing url")
            return False

        payload = {
            "event": event.value,
            "message": self.format_message(context),
            "context": context,
            "priority": self.config.priority.value,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    self.method,
                    self.url,
                    json=payload,
                    headers=self.headers,
                    timeout=10.0,
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Webhook send failed: {e}")
            return False


def create_handler(config: HookConfig) -> BaseHookHandler:
    """Create hook handler from config.

    Args:
        config: Handler configuration

    Returns:
        Configured handler instance
    """
    handlers = {
        "log": LogHookHandler,
        "telegram": TelegramHookHandler,
        "slack": SlackHookHandler,
        "webhook": WebhookHookHandler,
    }

    handler_class = handlers.get(config.type, LogHookHandler)
    return handler_class(config)


@dataclass
class HooksManager:
    """Manager for hook handlers.

    Coordinates multiple handlers for different events.

    Example:
        >>> manager = HooksManager.from_config({
        ...     "on_entry": [{"type": "telegram", "template": "Entry: {pair}"}],
        ...     "on_exit": [{"type": "log"}],
        ... })
        >>> await manager.trigger(HookEvent.ON_ENTRY, {"pair": "BTCUSDT"})
    """

    handlers: dict[HookEvent, list[BaseHookHandler]] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> HooksManager:
        """Create from configuration.

        Args:
            config: Dict with event names as keys and handler lists as values

        Returns:
            Configured HooksManager
        """
        handlers: dict[HookEvent, list[BaseHookHandler]] = {}

        for event in HookEvent:
            event_config = config.get(event.value, [])
            if isinstance(event_config, dict):
                event_config = [event_config]

            handlers[event] = [
                create_handler(HookConfig.from_dict(h))
                for h in event_config
                if h.get("enabled", True)
            ]

        return cls(handlers=handlers, enabled=config.get("enabled", True))

    async def trigger(
        self,
        event: HookEvent,
        context: dict[str, Any],
        *,
        wait: bool = False,
    ) -> list[bool]:
        """Trigger hooks for an event.

        Args:
            event: Event type
            context: Event context data
            wait: Whether to wait for all handlers to complete

        Returns:
            List of success/failure results
        """
        if not self.enabled:
            return []

        handlers = self.handlers.get(event, [])
        if not handlers:
            return []

        tasks = [handler.send(event, context) for handler in handlers]

        if wait:
            return await asyncio.gather(*tasks, return_exceptions=False)
        else:
            # Fire and forget
            for task in tasks:
                asyncio.create_task(task)
            return [True] * len(handlers)

    def trigger_sync(self, event: HookEvent, context: dict[str, Any]) -> list[bool]:
        """Trigger hooks synchronously.

        Args:
            event: Event type
            context: Event context data

        Returns:
            List of success/failure results
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule as task
                asyncio.create_task(self.trigger(event, context, wait=False))
                return [True]
            else:
                return loop.run_until_complete(self.trigger(event, context, wait=True))
        except RuntimeError:
            # No event loop
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.trigger(event, context, wait=True))
            finally:
                loop.close()

    def on_entry(self, context: dict[str, Any]) -> list[bool]:
        """Convenience method for entry events."""
        return self.trigger_sync(HookEvent.ON_ENTRY, context)

    def on_exit(self, context: dict[str, Any]) -> list[bool]:
        """Convenience method for exit events."""
        return self.trigger_sync(HookEvent.ON_EXIT, context)

    def on_error(self, context: dict[str, Any]) -> list[bool]:
        """Convenience method for error events."""
        return self.trigger_sync(HookEvent.ON_ERROR, context)


# Global hooks instance (can be configured once and used everywhere)
_global_hooks: HooksManager | None = None


def configure_hooks(config: dict[str, Any]) -> HooksManager:
    """Configure global hooks manager.

    Args:
        config: Hooks configuration

    Returns:
        Configured HooksManager
    """
    global _global_hooks
    _global_hooks = HooksManager.from_config(config)
    return _global_hooks


def get_hooks() -> HooksManager | None:
    """Get global hooks manager."""
    return _global_hooks
