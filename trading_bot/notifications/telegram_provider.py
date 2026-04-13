"""
notifications/telegram_provider.py — Telegram provider via python-telegram-bot.

Features:
  • Sends structured stock alerts with inline YES / NO / WAIT / DETAILS buttons
  • poll_reply waits for the user to tap a button (asyncio.Event-based)
  • send_document uploads the research report JSON/PDF
  • Markdown V2 formatting for clean message layout

Requires env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from loguru import logger

from trading_bot import config

from .base import (
    ApprovalChoice,
    ApprovalResponse,
    NotificationProvider,
    StockAlert,
)

# ── Lazy import — only needed when Telegram provider is active ─────────────────
try:
    from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.constants import ParseMode
    from telegram.ext import Application, CallbackQueryHandler, ContextTypes
    _TELEGRAM_AVAILABLE = True
except ImportError:
    _TELEGRAM_AVAILABLE = False


_APPROVAL_BUTTONS = [
    [
        InlineKeyboardButton("✅ YES", callback_data="YES") if _TELEGRAM_AVAILABLE else None,
        InlineKeyboardButton("❌ NO", callback_data="NO") if _TELEGRAM_AVAILABLE else None,
    ],
    [
        InlineKeyboardButton("⏳ WAIT", callback_data="WAIT") if _TELEGRAM_AVAILABLE else None,
        InlineKeyboardButton("🔍 DETAILS", callback_data="DETAILS") if _TELEGRAM_AVAILABLE else None,
    ],
] if _TELEGRAM_AVAILABLE else []


def _alert_markdown(alert: StockAlert) -> str:
    """Format alert as Telegram MarkdownV2."""
    # MarkdownV2 requires escaping: . - + = ( ) | { } # ! ~
    def esc(s: str) -> str:
        for ch in r"\_*[]()~`>#+-=|{}.!":
            s = s.replace(ch, f"\\{ch}")
        return s

    rr = alert.risk_reward
    return (
        f"*{esc(alert.action)} {esc(alert.symbol)}* \\[{esc(alert.exchange)}\\]\n"
        f"Style: `{esc(alert.style)}`\n\n"
        f"Entry      ₹`{alert.entry_price:,.2f}`\n"
        f"Stop Loss  ₹`{alert.stop_loss:,.2f}`\n"
        f"Target 1   ₹`{alert.target_1:,.2f}`\n"
        f"Target 2   ₹`{alert.target_2:,.2f}`\n"
        f"Size       `{alert.position_size_pct*100:.1f}%` of portfolio\n"
        f"Confidence `{alert.confidence_score*100:.0f}%`\n"
        f"R\\:R        `1:{rr:.1f}`\n\n"
        f"_{esc(alert.reasoning)}_"
    )


class TelegramProvider(NotificationProvider):
    """
    Telegram notification provider.

    Lifecycle:
      start() — builds and starts the Application (begins polling)
      stop()  — stops the Application cleanly
    """

    def __init__(self) -> None:
        if not _TELEGRAM_AVAILABLE:
            raise RuntimeError(
                "python-telegram-bot is not installed. "
                "Run: pip install python-telegram-bot"
            )
        if not config.TELEGRAM_BOT_TOKEN:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
        if not config.TELEGRAM_CHAT_ID:
            raise RuntimeError("TELEGRAM_CHAT_ID is not set in .env")

        self._chat_id = config.TELEGRAM_CHAT_ID
        self._app: Optional[Application] = None

        # symbol → asyncio.Event + chosen value for poll_reply synchronisation
        self._pending: dict[str, tuple[asyncio.Event, list[Optional[ApprovalChoice]]]] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._app = (
            Application.builder()
            .token(config.TELEGRAM_BOT_TOKEN)
            .build()
        )
        self._app.add_handler(CallbackQueryHandler(self._on_button_press))
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("[TELEGRAM] Provider started — polling for updates.")

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("[TELEGRAM] Provider stopped.")

    # ── Core interface ────────────────────────────────────────────────────────

    async def send_message(self, text: str, parse_mode: str = "text") -> None:
        if not self._app:
            raise RuntimeError("TelegramProvider.start() not called.")
        pm = ParseMode.MARKDOWN_V2 if parse_mode == "markdown" else None
        await self._app.bot.send_message(
            chat_id=self._chat_id,
            text=text,
            parse_mode=pm,
        )
        logger.debug("[TELEGRAM] Message sent.")

    async def send_document(self, path: Path, caption: str = "") -> None:
        if not self._app:
            raise RuntimeError("TelegramProvider.start() not called.")
        with open(path, "rb") as fh:
            await self._app.bot.send_document(
                chat_id=self._chat_id,
                document=fh,
                filename=path.name,
                caption=caption or path.name,
            )
        logger.info("[TELEGRAM] Document sent: {}", path.name)

    async def send_alert(self, alert: StockAlert) -> None:
        if not self._app:
            raise RuntimeError("TelegramProvider.start() not called.")

        keyboard = InlineKeyboardMarkup(_APPROVAL_BUTTONS)
        text = _alert_markdown(alert)

        await self._app.bot.send_message(
            chat_id=self._chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard,
        )
        logger.info("[TELEGRAM] Alert sent for {}.", alert.symbol)

    async def poll_reply(
        self,
        alert: StockAlert,
        timeout_seconds: int = 300,
    ) -> ApprovalResponse:
        """Wait for the user to tap an inline button for *alert.symbol*."""
        event: asyncio.Event = asyncio.Event()
        choice_box: list[Optional[ApprovalChoice]] = [None]
        self._pending[alert.symbol] = (event, choice_box)

        try:
            await asyncio.wait_for(event.wait(), timeout=float(timeout_seconds))
            choice = choice_box[0] or ApprovalChoice.NO
            timed_out = False
        except asyncio.TimeoutError:
            choice = ApprovalChoice.NO
            timed_out = True
            logger.warning("[TELEGRAM] Approval timeout for {}.", alert.symbol)
        finally:
            self._pending.pop(alert.symbol, None)

        logger.info("[TELEGRAM] {} → {} (timed_out={})", alert.symbol, choice.value, timed_out)
        return ApprovalResponse(choice=choice, symbol=alert.symbol, timed_out=timed_out)

    # ── Internal callback ─────────────────────────────────────────────────────

    async def _on_button_press(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handles inline keyboard button presses from the user."""
        query = update.callback_query
        await query.answer()

        raw = (query.data or "").strip().upper()
        try:
            choice = ApprovalChoice(raw)
        except ValueError:
            logger.warning("[TELEGRAM] Unknown callback data: '{}'", raw)
            return

        # Match the pending alert — we only support one pending at a time per
        # symbol.  If multiple alerts are outstanding, the first match wins.
        for symbol, (event, choice_box) in list(self._pending.items()):
            choice_box[0] = choice
            event.set()
            await query.edit_message_reply_markup(reply_markup=None)
            logger.info("[TELEGRAM] Button '{}' pressed for {}.", choice.value, symbol)
            break
