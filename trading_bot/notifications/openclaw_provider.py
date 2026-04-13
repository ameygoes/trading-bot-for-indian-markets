"""
notifications/openclaw_provider.py — OpenClaw notification provider (STUB).

TODO: Replace all TODO sections with real OpenClaw API calls once the
      integration spec is finalised.

Env vars required:
  OPENCLAW_API_KEY      — API key for authentication
  OPENCLAW_WEBHOOK_URL  — Webhook URL for inbound replies
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from trading_bot import config

from .base import (
    ApprovalChoice,
    ApprovalResponse,
    NotificationProvider,
    StockAlert,
)


class OpenClawProvider(NotificationProvider):
    """
    Stub provider for OpenClaw integration.

    All methods log what *would* be sent and return safe defaults so the rest
    of the bot can run end-to-end without a live OpenClaw account.
    """

    def __init__(self) -> None:
        if not config.OPENCLAW_API_KEY:
            logger.warning(
                "[OPENCLAW] OPENCLAW_API_KEY is not set — provider is fully stubbed."
            )
        if not config.OPENCLAW_WEBHOOK_URL:
            logger.warning(
                "[OPENCLAW] OPENCLAW_WEBHOOK_URL is not set — replies will time-out."
            )
        self._api_key = config.OPENCLAW_API_KEY
        self._webhook_url = config.OPENCLAW_WEBHOOK_URL

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        # TODO: register webhook with OpenClaw so it can POST replies back
        # TODO: verify OPENCLAW_API_KEY is valid (health-check call)
        logger.info("[OPENCLAW] Provider started (stub — no real connection).")

    async def stop(self) -> None:
        # TODO: deregister webhook if OpenClaw supports it
        logger.info("[OPENCLAW] Provider stopped.")

    # ── Core interface ────────────────────────────────────────────────────────

    async def send_message(self, text: str, parse_mode: str = "text") -> None:
        # TODO: POST { "text": text } to OpenClaw messages endpoint
        # TODO: handle HTTP errors, rate limits, retry with tenacity
        logger.info("[OPENCLAW STUB] send_message: {!r}", text[:80])

    async def send_document(self, path: Path, caption: str = "") -> None:
        # TODO: multipart upload to OpenClaw file endpoint
        # TODO: include caption as metadata
        logger.info("[OPENCLAW STUB] send_document: {} (caption={!r})", path.name, caption)

    async def send_alert(self, alert: StockAlert) -> None:
        # TODO: construct OpenClaw-flavoured payload with actionable buttons
        # TODO: include symbol, action, prices, confidence in structured fields
        # TODO: attach inline YES/NO/WAIT/DETAILS action identifiers in payload
        logger.info(
            "[OPENCLAW STUB] send_alert: {} {} @ ₹{:.2f}",
            alert.action,
            alert.symbol,
            alert.entry_price,
        )
        logger.debug("[OPENCLAW STUB] Alert text:\n{}", alert.to_text())

    async def poll_reply(
        self,
        alert: StockAlert,
        timeout_seconds: int = 300,
    ) -> ApprovalResponse:
        # TODO: long-poll or WebSocket listen on OPENCLAW_WEBHOOK_URL for the
        #       user's button press keyed to alert.symbol
        # TODO: return the actual ApprovalChoice from the webhook payload
        logger.warning(
            "[OPENCLAW STUB] poll_reply called for {} — auto-returning NO (stub).",
            alert.symbol,
        )
        return ApprovalResponse(
            choice=ApprovalChoice.NO,
            symbol=alert.symbol,
            timed_out=True,
        )
