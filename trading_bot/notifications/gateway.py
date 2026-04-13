"""
notifications/gateway.py — Provider factory + unified notification gateway.

Usage:
    from trading_bot.notifications.gateway import get_gateway

    gw = get_gateway()          # call once; reuse the singleton
    await gw.start()

    await gw.send_alert(alert)
    response = await gw.poll_reply(alert, timeout_seconds=300)

    await gw.stop()

The concrete provider is selected by the NOTIFICATION_PROVIDER env var:
  "console"   → ConsoleProvider   (default, no deps)
  "telegram"  → TelegramProvider  (python-telegram-bot)
  "openclaw"  → OpenClawProvider  (stub)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from trading_bot import config

from .base import ApprovalResponse, NotificationProvider, StockAlert

# ── Singleton ─────────────────────────────────────────────────────────────────
_gateway_instance: Optional["NotificationGateway"] = None


class NotificationGateway:
    """
    Thin façade over a concrete NotificationProvider.

    Forwards all calls to the active provider; adds structured logging so every
    outbound notification is traceable in the JSON log file.
    """

    def __init__(self, provider: NotificationProvider) -> None:
        self._provider = provider

    @property
    def provider_name(self) -> str:
        return type(self._provider).__name__

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        logger.info("NotificationGateway starting with provider: {}", self.provider_name)
        await self._provider.start()

    async def stop(self) -> None:
        await self._provider.stop()
        logger.info("NotificationGateway stopped.")

    # ── Message methods ───────────────────────────────────────────────────────

    async def send_message(self, text: str, parse_mode: str = "text") -> None:
        logger.debug("Gateway.send_message via {}", self.provider_name)
        await self._provider.send_message(text, parse_mode=parse_mode)

    async def send_document(self, path: Path, caption: str = "") -> None:
        logger.debug("Gateway.send_document: {}", path.name)
        await self._provider.send_document(path, caption=caption)

    async def send_alert(self, alert: StockAlert) -> None:
        logger.info(
            "Gateway.send_alert: {} {} confidence={:.0f}%",
            alert.action,
            alert.symbol,
            alert.confidence_score * 100,
        )
        await self._provider.send_alert(alert)

    async def poll_reply(
        self,
        alert: StockAlert,
        timeout_seconds: int = 300,
    ) -> ApprovalResponse:
        logger.info(
            "Gateway.poll_reply: waiting up to {}s for {} approval",
            timeout_seconds,
            alert.symbol,
        )
        response = await self._provider.poll_reply(alert, timeout_seconds=timeout_seconds)
        logger.info(
            "Gateway.poll_reply result: {} → {} (timed_out={})",
            alert.symbol,
            response.choice.value,
            response.timed_out,
        )
        return response

    # ── Convenience helpers ───────────────────────────────────────────────────

    async def notify_and_await_approval(
        self,
        alert: StockAlert,
        timeout_seconds: int = 300,
    ) -> ApprovalResponse:
        """send_alert + poll_reply in one call — most callers want this."""
        await self.send_alert(alert)
        return await self.poll_reply(alert, timeout_seconds=timeout_seconds)


# ── Factory ───────────────────────────────────────────────────────────────────

def _build_provider(name: str) -> NotificationProvider:
    name = name.strip().lower()

    if name == "telegram":
        from .telegram_provider import TelegramProvider
        return TelegramProvider()

    if name == "openclaw":
        from .openclaw_provider import OpenClawProvider
        return OpenClawProvider()

    if name in ("console", ""):
        from .console_provider import ConsoleProvider
        return ConsoleProvider()

    logger.warning(
        "Unknown NOTIFICATION_PROVIDER '{}' — falling back to console.", name
    )
    from .console_provider import ConsoleProvider
    return ConsoleProvider()


def get_gateway() -> NotificationGateway:
    """Return the process-level singleton gateway, creating it on first call."""
    global _gateway_instance
    if _gateway_instance is None:
        provider_name = config.NOTIFICATION_PROVIDER
        provider = _build_provider(provider_name)
        _gateway_instance = NotificationGateway(provider)
        logger.info(
            "NotificationGateway created: NOTIFICATION_PROVIDER='{}'", provider_name
        )
    return _gateway_instance
