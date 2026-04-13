"""
notifications/telegram_provider.py — Full Telegram approval flow.

Button behaviour:
  YES / NO  → edits the alert message to show a confirmation badge; resolves event
  WAIT      → snoozes the alert (edits message); caller re-alerts after a delay
  DETAILS   → sends research report (or full reasoning text); switches keyboard to
              YES / NO / WAIT (no second DETAILS press)

Additional commands:
  /status   → lists any currently-pending alerts

Requires env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from trading_bot import config

from .base import ApprovalChoice, ApprovalResponse, NotificationProvider, StockAlert

# ── Optional dependency ────────────────────────────────────────────────────────
try:
    from telegram import Bot, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.constants import ParseMode
    from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes
    _TELEGRAM_AVAILABLE = True
except ImportError:
    _TELEGRAM_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Keyboard layouts
# ══════════════════════════════════════════════════════════════════════════════

def _full_keyboard() -> "InlineKeyboardMarkup":
    """All four buttons — shown when the alert first arrives."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ YES",     callback_data="YES"),
            InlineKeyboardButton("❌ NO",      callback_data="NO"),
        ],
        [
            InlineKeyboardButton("⏳ WAIT",    callback_data="WAIT"),
            InlineKeyboardButton("🔍 DETAILS", callback_data="DETAILS"),
        ],
    ])


def _decision_keyboard() -> "InlineKeyboardMarkup":
    """YES / NO / WAIT only — shown after DETAILS has been tapped."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ YES",  callback_data="YES"),
            InlineKeyboardButton("❌ NO",   callback_data="NO"),
            InlineKeyboardButton("⏳ WAIT", callback_data="WAIT"),
        ],
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Pending-alert state
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _PendingAlert:
    """Mutable state for one outstanding alert awaiting user approval."""
    alert: StockAlert
    message_id: int
    event: asyncio.Event = field(default_factory=asyncio.Event)
    choice: list[Optional[ApprovalChoice]] = field(default_factory=lambda: [None])


# ══════════════════════════════════════════════════════════════════════════════
# Markdown helpers
# ══════════════════════════════════════════════════════════════════════════════

def _esc(s: str) -> str:
    """Escape all MarkdownV2 special characters in a plain string."""
    for ch in r"\_*[]()~`>#+-=|{}.!":
        s = s.replace(ch, f"\\{ch}")
    return s


def _alert_markdown(alert: StockAlert, prefix: str = "") -> str:
    """
    Render a StockAlert as Telegram MarkdownV2.

    Args:
        alert:  The stock alert to format.
        prefix: Optional pre-formatted MarkdownV2 header prepended before the
                alert body (e.g. "✅ *APPROVED*\\n\\n").  Must already be
                properly escaped — it is inserted verbatim.
    """
    rr = alert.risk_reward
    header = f"*{_esc(alert.action)} {_esc(alert.symbol)}* \\[{_esc(alert.exchange)}\\]\n"
    body = (
        f"Style: `{_esc(alert.style)}`\n\n"
        f"Entry      ₹`{alert.entry_price:,.2f}`\n"
        f"Stop Loss  ₹`{alert.stop_loss:,.2f}`\n"
        f"Target 1   ₹`{alert.target_1:,.2f}`\n"
        f"Target 2   ₹`{alert.target_2:,.2f}`\n"
        f"Size       `{alert.position_size_pct*100:.1f}%` of portfolio\n"
        f"Confidence `{alert.confidence_score*100:.0f}%`\n"
        f"R\\:R        `1:{rr:.1f}`\n\n"
        f"_{_esc(alert.reasoning)}_"
    )
    if alert.tags:
        body += f"\n\nTags: {_esc(', '.join(alert.tags))}"
    return prefix + header + body


# ══════════════════════════════════════════════════════════════════════════════
# Provider
# ══════════════════════════════════════════════════════════════════════════════

class TelegramProvider(NotificationProvider):
    """
    Telegram notification provider with a full four-button approval flow.

    Lifecycle:
        start()  → builds Application, registers handlers, begins polling
        stop()   → stops polling and shuts down cleanly
    """

    def __init__(self) -> None:
        if not _TELEGRAM_AVAILABLE:
            raise RuntimeError(
                "python-telegram-bot is not installed. "
                "Run: pip install 'python-telegram-bot'"
            )
        if not config.TELEGRAM_BOT_TOKEN:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
        if not config.TELEGRAM_CHAT_ID:
            raise RuntimeError("TELEGRAM_CHAT_ID is not set in .env")

        self._chat_id: str = config.TELEGRAM_CHAT_ID
        self._app: Optional[Application] = None
        # Active pending alerts keyed by symbol — one at a time per symbol.
        self._pending: dict[str, _PendingAlert] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        self._app.add_handler(CallbackQueryHandler(self._on_button_press))
        self._app.add_handler(CommandHandler("status", self._on_status_command))
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
        pm = ParseMode.MARKDOWN_V2 if parse_mode == "markdown" else None
        await self._bot().send_message(chat_id=self._chat_id, text=text, parse_mode=pm)
        logger.debug("[TELEGRAM] Message sent.")

    async def send_document(self, path: Path, caption: str = "") -> None:
        with open(path, "rb") as fh:
            await self._bot().send_document(
                chat_id=self._chat_id,
                document=fh,
                filename=path.name,
                caption=caption or path.name,
            )
        logger.info("[TELEGRAM] Document sent: {}", path.name)

    async def send_alert(self, alert: StockAlert) -> None:
        if alert.symbol in self._pending:
            logger.warning("[TELEGRAM] Overwriting existing pending entry for {}", alert.symbol)

        msg = await self._bot().send_message(
            chat_id=self._chat_id,
            text=_alert_markdown(alert),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=_full_keyboard(),
        )
        self._pending[alert.symbol] = _PendingAlert(
            alert=alert,
            message_id=msg.message_id,
        )
        logger.info("[TELEGRAM] Alert sent for {} (msg_id={})", alert.symbol, msg.message_id)

    async def poll_reply(
        self,
        alert: StockAlert,
        timeout_seconds: int = 300,
    ) -> ApprovalResponse:
        """
        Wait for the user to tap an inline button for *alert.symbol*.

        - send_alert() must be called first to register the pending entry.
        - WAIT resolves immediately — the caller is responsible for re-alerting.
        - Timeout → auto-NO; keyboard is cleared.
        """
        entry = self._pending.get(alert.symbol)
        if entry is None:
            logger.warning("[TELEGRAM] poll_reply called before send_alert for {}", alert.symbol)
            return ApprovalResponse(choice=ApprovalChoice.NO, symbol=alert.symbol, timed_out=True)

        try:
            await asyncio.wait_for(entry.event.wait(), timeout=float(timeout_seconds))
            choice = entry.choice[0] or ApprovalChoice.NO
            timed_out = False
        except asyncio.TimeoutError:
            choice = ApprovalChoice.NO
            timed_out = True
            logger.warning("[TELEGRAM] Approval timeout for {}", alert.symbol)
            await self._try_clear_keyboard(entry.message_id)
        finally:
            self._pending.pop(alert.symbol, None)

        logger.info(
            "[TELEGRAM] {} → {} (timed_out={})", alert.symbol, choice.value, timed_out
        )
        return ApprovalResponse(choice=choice, symbol=alert.symbol, timed_out=timed_out)

    # ── /status command ───────────────────────────────────────────────────────

    async def _on_status_command(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """/status — show all currently-pending alerts."""
        if not self._pending:
            text = "✅ No pending alerts\\."
        else:
            lines = ["*Pending alerts:*"]
            for sym, entry in self._pending.items():
                lines.append(f"  • {_esc(sym)} \\(msg `{entry.message_id}`\\)")
            text = "\n".join(lines)
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)

    # ── Callback router ───────────────────────────────────────────────────────

    async def _on_button_press(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Route inline keyboard presses to the matching pending alert."""
        query = update.callback_query
        await query.answer()

        raw = (query.data or "").strip().upper()
        try:
            choice = ApprovalChoice(raw)
        except ValueError:
            logger.warning("[TELEGRAM] Unknown callback data: '{}'", raw)
            return

        # Match by message_id so concurrent alerts for different symbols can't collide.
        msg_id = query.message.message_id
        entry: Optional[_PendingAlert] = None
        for pending in self._pending.values():
            if pending.message_id == msg_id:
                entry = pending
                break

        if entry is None:
            logger.warning("[TELEGRAM] No pending alert for msg_id={}", msg_id)
            await query.edit_message_reply_markup(reply_markup=None)
            return

        logger.info("[TELEGRAM] '{}' pressed for {}", choice.value, entry.alert.symbol)

        if choice == ApprovalChoice.DETAILS:
            await self._handle_details(query, entry)
        elif choice == ApprovalChoice.WAIT:
            await self._handle_wait(query, entry)
        else:  # YES or NO
            await self._handle_decision(query, entry, choice)

    # ── Per-button handlers ───────────────────────────────────────────────────

    async def _handle_details(
        self, query: "CallbackQuery", entry: _PendingAlert
    ) -> None:
        """
        DETAILS pressed:
          1. Send the research report file, or fall back to the reasoning text.
          2. Switch keyboard to YES / NO / WAIT (remove DETAILS button).
          3. Do NOT set the event — the user still needs to decide.
        """
        alert = entry.alert
        report_path = alert.research_report_path
        if report_path and Path(report_path).exists():
            await self.send_document(Path(report_path), caption=f"Research: {alert.symbol}")
        else:
            await self._bot().send_message(
                chat_id=self._chat_id,
                text=(
                    f"*{_esc(alert.symbol)} — Full Details*\n\n"
                    f"_{_esc(alert.reasoning)}_"
                ),
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        # Downgrade keyboard — DETAILS button removed
        await query.edit_message_reply_markup(reply_markup=_decision_keyboard())
        logger.debug("[TELEGRAM] Details sent for {}", alert.symbol)

    async def _handle_wait(
        self, query: "CallbackQuery", entry: _PendingAlert
    ) -> None:
        """
        WAIT pressed:
          1. Edit message to show a snooze badge (remove keyboard).
          2. Resolve the event with WAIT so the caller can re-alert later.
        """
        await query.edit_message_text(
            text=_alert_markdown(
                entry.alert,
                prefix="⏳ *SNOOZED* — will re\\-alert at next candle\n\n",
            ),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=None,
        )
        entry.choice[0] = ApprovalChoice.WAIT
        entry.event.set()

    async def _handle_decision(
        self,
        query: "CallbackQuery",
        entry: _PendingAlert,
        choice: ApprovalChoice,
    ) -> None:
        """
        YES or NO pressed:
          1. Edit message to show confirmation badge (remove keyboard).
          2. Resolve the event with the chosen value.
        """
        prefix = (
            "✅ *APPROVED* — executing trade\n\n"
            if choice == ApprovalChoice.YES
            else "❌ *REJECTED*\n\n"
        )
        await query.edit_message_text(
            text=_alert_markdown(entry.alert, prefix=prefix),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=None,
        )
        entry.choice[0] = choice
        entry.event.set()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _bot(self) -> "Bot":
        if not self._app:
            raise RuntimeError("TelegramProvider.start() not called.")
        return self._app.bot

    async def _try_clear_keyboard(self, message_id: int) -> None:
        """Best-effort removal of the inline keyboard (ignores API errors)."""
        try:
            await self._bot().edit_message_reply_markup(
                chat_id=self._chat_id,
                message_id=message_id,
                reply_markup=None,
            )
        except Exception as exc:
            logger.debug("[TELEGRAM] Could not clear keyboard for msg {}: {}", message_id, exc)
