"""
tests/test_telegram_provider.py — Unit tests for the Telegram approval flow.

All tests run fully offline (no real Telegram connection).  The
python-telegram-bot library is mocked at the Application/Bot level so we can
exercise every code path without network access.

Test categories:
  - Markdown formatting (_esc, _alert_markdown)
  - send_alert: stores pending entry, calls bot.send_message with keyboard
  - poll_reply: resolves on event set, handles missing entry, handles timeout
  - DETAILS button: sends report/fallback, updates keyboard, does NOT resolve event
  - WAIT button: edits message, resolves event with WAIT choice
  - YES / NO buttons: edit message with badge, resolve event with correct choice
  - Unknown callback data: logged and ignored
  - /status command: lists pending or reports none
  - Concurrent alerts for different symbols don't collide
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Skip entire module if python-telegram-bot is not installed ────────────────
telegram = pytest.importorskip(
    "telegram",
    reason="python-telegram-bot not installed — skipping TelegramProvider tests",
)

from trading_bot.notifications.base import ApprovalChoice, StockAlert
from trading_bot.notifications.telegram_provider import (
    TelegramProvider,
    _PendingAlert,
    _alert_markdown,
    _decision_keyboard,
    _esc,
    _full_keyboard,
)


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

DIXON_ALERT = StockAlert(
    symbol="DIXON",
    exchange="NSE",
    action="BUY",
    style="SWING",
    entry_price=14_250.00,
    stop_loss=13_600.00,
    target_1=15_500.00,
    target_2=16_800.00,
    position_size_pct=0.04,
    confidence_score=0.78,
    reasoning="Strong breakout on 2× volume with sector tailwinds.",
    tags=["breakout", "momentum"],
)

RELIANCE_ALERT = StockAlert(
    symbol="RELIANCE",
    exchange="NSE",
    action="BUY",
    style="SWING_LIGHT",
    entry_price=2_900.00,
    stop_loss=2_750.00,
    target_1=3_100.00,
    target_2=3_300.00,
    position_size_pct=0.03,
    confidence_score=0.65,
    reasoning="Consolidation breakout near 52-week high.",
)


def _make_provider() -> tuple[TelegramProvider, MagicMock]:
    """
    Return a TelegramProvider with its internal _app replaced by a mock.
    The mock bot's send_message returns a message with message_id=42 by default.
    """
    with (
        patch("trading_bot.config.TELEGRAM_BOT_TOKEN", "fake-token"),
        patch("trading_bot.config.TELEGRAM_CHAT_ID", "12345"),
    ):
        provider = TelegramProvider()

    mock_msg = MagicMock()
    mock_msg.message_id = 42

    mock_bot = MagicMock()
    mock_bot.send_message   = AsyncMock(return_value=mock_msg)
    mock_bot.send_document  = AsyncMock()
    mock_bot.edit_message_reply_markup = AsyncMock()
    mock_bot.edit_message_text         = AsyncMock()

    mock_app = MagicMock()
    mock_app.bot = mock_bot

    provider._app = mock_app
    provider._chat_id = "12345"
    return provider, mock_bot


def _make_callback_query(
    data: str,
    message_id: int = 42,
    edit_text_side_effect=None,
) -> MagicMock:
    """Build a mock CallbackQuery for the given button data."""
    query = AsyncMock()
    query.data = data
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    query.edit_message_text = AsyncMock(side_effect=edit_text_side_effect)
    query.message = MagicMock()
    query.message.message_id = message_id
    return query


def _make_update(query: MagicMock) -> MagicMock:
    update = MagicMock()
    update.callback_query = query
    return update


# ══════════════════════════════════════════════════════════════════════════════
# Markdown helpers
# ══════════════════════════════════════════════════════════════════════════════

def test_esc_special_characters() -> None:
    assert _esc("1.5") == "1\\.5"
    assert _esc("(x)") == "\\(x\\)"
    assert _esc("a-b") == "a\\-b"
    assert _esc("a_b") == "a\\_b"
    assert _esc("no specials") == "no specials"


def test_alert_markdown_contains_symbol() -> None:
    md = _alert_markdown(DIXON_ALERT)
    assert "DIXON" in md
    assert "NSE" in md
    assert "BUY" in md


def test_alert_markdown_contains_prices() -> None:
    md = _alert_markdown(DIXON_ALERT)
    assert "14,250.00" in md
    assert "13,600.00" in md
    assert "15,500.00" in md


def test_alert_markdown_tags_present() -> None:
    md = _alert_markdown(DIXON_ALERT)
    assert "breakout" in md
    assert "momentum" in md


def test_alert_markdown_prefix_prepended() -> None:
    md = _alert_markdown(DIXON_ALERT, prefix="✅ *APPROVED*\n\n")
    assert md.startswith("✅ *APPROVED*\n\n")
    assert "DIXON" in md


def test_alert_markdown_no_prefix() -> None:
    md = _alert_markdown(DIXON_ALERT, prefix="")
    assert md.startswith("*BUY")


def test_full_keyboard_has_four_buttons() -> None:
    kb = _full_keyboard()
    buttons = [btn for row in kb.inline_keyboard for btn in row]
    data_values = {b.callback_data for b in buttons}
    assert data_values == {"YES", "NO", "WAIT", "DETAILS"}


def test_decision_keyboard_has_three_buttons() -> None:
    kb = _decision_keyboard()
    buttons = [btn for row in kb.inline_keyboard for btn in row]
    data_values = {b.callback_data for b in buttons}
    assert data_values == {"YES", "NO", "WAIT"}
    assert "DETAILS" not in data_values


# ══════════════════════════════════════════════════════════════════════════════
# send_alert
# ══════════════════════════════════════════════════════════════════════════════

async def test_send_alert_registers_pending_entry() -> None:
    provider, bot = _make_provider()
    await provider.send_alert(DIXON_ALERT)
    assert "DIXON" in provider._pending
    assert provider._pending["DIXON"].message_id == 42


async def test_send_alert_calls_send_message_with_keyboard() -> None:
    provider, bot = _make_provider()
    await provider.send_alert(DIXON_ALERT)
    bot.send_message.assert_called_once()
    call_kwargs = bot.send_message.call_args.kwargs
    assert call_kwargs["reply_markup"] is not None


async def test_send_alert_overwrites_stale_pending() -> None:
    provider, bot = _make_provider()
    # First alert
    await provider.send_alert(DIXON_ALERT)
    first_msg_id = provider._pending["DIXON"].message_id
    # Simulate a second alert for the same symbol (e.g. re-alert after WAIT)
    bot.send_message.return_value.message_id = 99
    await provider.send_alert(DIXON_ALERT)
    assert provider._pending["DIXON"].message_id == 99


# ══════════════════════════════════════════════════════════════════════════════
# poll_reply — event already set (fast path)
# ══════════════════════════════════════════════════════════════════════════════

async def test_poll_reply_yes() -> None:
    provider, _ = _make_provider()
    await provider.send_alert(DIXON_ALERT)
    # Simulate user tapping YES
    entry = provider._pending["DIXON"]
    entry.choice[0] = ApprovalChoice.YES
    entry.event.set()

    response = await provider.poll_reply(DIXON_ALERT, timeout_seconds=5)
    assert response.choice == ApprovalChoice.YES
    assert response.timed_out is False
    assert response.symbol == "DIXON"


async def test_poll_reply_no() -> None:
    provider, _ = _make_provider()
    await provider.send_alert(DIXON_ALERT)
    entry = provider._pending["DIXON"]
    entry.choice[0] = ApprovalChoice.NO
    entry.event.set()

    response = await provider.poll_reply(DIXON_ALERT, timeout_seconds=5)
    assert response.choice == ApprovalChoice.NO
    assert response.timed_out is False


async def test_poll_reply_wait() -> None:
    provider, _ = _make_provider()
    await provider.send_alert(DIXON_ALERT)
    entry = provider._pending["DIXON"]
    entry.choice[0] = ApprovalChoice.WAIT
    entry.event.set()

    response = await provider.poll_reply(DIXON_ALERT, timeout_seconds=5)
    assert response.choice == ApprovalChoice.WAIT
    assert response.timed_out is False


async def test_poll_reply_removes_pending_after_resolve() -> None:
    provider, _ = _make_provider()
    await provider.send_alert(DIXON_ALERT)
    entry = provider._pending["DIXON"]
    entry.choice[0] = ApprovalChoice.YES
    entry.event.set()

    await provider.poll_reply(DIXON_ALERT, timeout_seconds=5)
    assert "DIXON" not in provider._pending


async def test_poll_reply_missing_entry_returns_no() -> None:
    """Calling poll_reply without prior send_alert returns NO immediately."""
    provider, _ = _make_provider()
    response = await provider.poll_reply(DIXON_ALERT, timeout_seconds=1)
    assert response.choice == ApprovalChoice.NO
    assert response.timed_out is True


async def test_poll_reply_timeout_returns_no() -> None:
    provider, bot = _make_provider()
    await provider.send_alert(DIXON_ALERT)
    # Don't set the event — let it time out
    response = await provider.poll_reply(DIXON_ALERT, timeout_seconds=0.05)
    assert response.choice == ApprovalChoice.NO
    assert response.timed_out is True


# ══════════════════════════════════════════════════════════════════════════════
# Button handler — YES / NO
# ══════════════════════════════════════════════════════════════════════════════

async def test_button_yes_resolves_event() -> None:
    provider, _ = _make_provider()
    await provider.send_alert(DIXON_ALERT)

    query  = _make_callback_query("YES", message_id=42)
    update = _make_update(query)
    await provider._on_button_press(update, MagicMock())

    entry = provider._pending.get("DIXON")
    # Entry is still present — poll_reply hasn't consumed it yet
    assert entry is not None
    assert entry.choice[0] == ApprovalChoice.YES
    assert entry.event.is_set()
    query.edit_message_text.assert_awaited_once()


async def test_button_no_resolves_event() -> None:
    provider, _ = _make_provider()
    await provider.send_alert(DIXON_ALERT)

    query  = _make_callback_query("NO", message_id=42)
    update = _make_update(query)
    await provider._on_button_press(update, MagicMock())

    entry = provider._pending["DIXON"]
    assert entry.choice[0] == ApprovalChoice.NO
    assert entry.event.is_set()


async def test_button_yes_edits_message_with_approved_badge() -> None:
    provider, _ = _make_provider()
    await provider.send_alert(DIXON_ALERT)

    query  = _make_callback_query("YES", message_id=42)
    update = _make_update(query)
    await provider._on_button_press(update, MagicMock())

    call_args = query.edit_message_text.call_args
    text = call_args.kwargs.get("text") or call_args.args[0]
    assert "APPROVED" in text
    # Keyboard should be cleared
    assert call_args.kwargs.get("reply_markup") is None


async def test_button_no_edits_message_with_rejected_badge() -> None:
    provider, _ = _make_provider()
    await provider.send_alert(DIXON_ALERT)

    query  = _make_callback_query("NO", message_id=42)
    update = _make_update(query)
    await provider._on_button_press(update, MagicMock())

    text = query.edit_message_text.call_args.kwargs.get("text") or ""
    assert "REJECTED" in text


# ══════════════════════════════════════════════════════════════════════════════
# Button handler — WAIT
# ══════════════════════════════════════════════════════════════════════════════

async def test_button_wait_resolves_event_with_wait() -> None:
    provider, _ = _make_provider()
    await provider.send_alert(DIXON_ALERT)

    query  = _make_callback_query("WAIT", message_id=42)
    update = _make_update(query)
    await provider._on_button_press(update, MagicMock())

    entry = provider._pending["DIXON"]
    assert entry.choice[0] == ApprovalChoice.WAIT
    assert entry.event.is_set()


async def test_button_wait_edits_message_with_snooze_text() -> None:
    provider, _ = _make_provider()
    await provider.send_alert(DIXON_ALERT)

    query  = _make_callback_query("WAIT", message_id=42)
    update = _make_update(query)
    await provider._on_button_press(update, MagicMock())

    text = query.edit_message_text.call_args.kwargs.get("text") or ""
    assert "SNOOZED" in text
    assert query.edit_message_text.call_args.kwargs.get("reply_markup") is None


# ══════════════════════════════════════════════════════════════════════════════
# Button handler — DETAILS
# ══════════════════════════════════════════════════════════════════════════════

async def test_button_details_does_not_resolve_event() -> None:
    provider, bot = _make_provider()
    await provider.send_alert(DIXON_ALERT)

    query  = _make_callback_query("DETAILS", message_id=42)
    update = _make_update(query)
    await provider._on_button_press(update, MagicMock())

    entry = provider._pending["DIXON"]
    assert not entry.event.is_set(), "DETAILS should not resolve the pending event"


async def test_button_details_switches_to_decision_keyboard() -> None:
    provider, bot = _make_provider()
    await provider.send_alert(DIXON_ALERT)

    query  = _make_callback_query("DETAILS", message_id=42)
    update = _make_update(query)
    await provider._on_button_press(update, MagicMock())

    query.edit_message_reply_markup.assert_awaited_once()
    kb = query.edit_message_reply_markup.call_args.kwargs.get("reply_markup")
    if kb is None:
        kb = query.edit_message_reply_markup.call_args.args[0]
    buttons = [b for row in kb.inline_keyboard for b in row]
    data = {b.callback_data for b in buttons}
    assert data == {"YES", "NO", "WAIT"}


async def test_button_details_sends_fallback_when_no_report(capsys) -> None:
    """With no research_report_path, DETAILS sends a plain text follow-up."""
    provider, bot = _make_provider()
    await provider.send_alert(DIXON_ALERT)  # DIXON_ALERT has no report path

    query  = _make_callback_query("DETAILS", message_id=42)
    update = _make_update(query)
    await provider._on_button_press(update, MagicMock())

    bot.send_message.assert_awaited()
    # The second call (after send_alert) should contain the symbol
    calls = bot.send_message.call_args_list
    texts = [c.kwargs.get("text", "") for c in calls]
    assert any("DIXON" in t for t in texts)


async def test_button_details_sends_document_when_report_exists(tmp_path: Path) -> None:
    report = tmp_path / "DIXON_report.json"
    report.write_text('{"symbol": "DIXON"}')

    alert_with_report = StockAlert(
        **{
            **DIXON_ALERT.__dict__,
            "research_report_path": report,
        }
    )

    provider, bot = _make_provider()
    await provider.send_alert(alert_with_report)

    query  = _make_callback_query("DETAILS", message_id=42)
    update = _make_update(query)
    await provider._on_button_press(update, MagicMock())

    bot.send_document.assert_awaited_once()


# ══════════════════════════════════════════════════════════════════════════════
# Unknown callback data
# ══════════════════════════════════════════════════════════════════════════════

async def test_unknown_callback_data_ignored() -> None:
    provider, _ = _make_provider()
    await provider.send_alert(DIXON_ALERT)

    query  = _make_callback_query("YOLO", message_id=42)
    update = _make_update(query)
    # Should not raise
    await provider._on_button_press(update, MagicMock())
    # Event must not be set
    assert not provider._pending["DIXON"].event.is_set()


async def test_stale_message_id_clears_keyboard() -> None:
    """Callback for a msg_id not in _pending should just clear the keyboard."""
    provider, _ = _make_provider()
    await provider.send_alert(DIXON_ALERT)

    query  = _make_callback_query("YES", message_id=999)  # wrong message_id
    update = _make_update(query)
    await provider._on_button_press(update, MagicMock())

    # Pending entry for DIXON should still be there (not accidentally consumed)
    assert "DIXON" in provider._pending
    assert not provider._pending["DIXON"].event.is_set()
    # Keyboard cleared on the orphan message
    query.edit_message_reply_markup.assert_awaited_once()


# ══════════════════════════════════════════════════════════════════════════════
# Concurrent alerts — different symbols don't collide
# ══════════════════════════════════════════════════════════════════════════════

async def test_concurrent_alerts_different_symbols() -> None:
    provider, bot = _make_provider()

    # Alert 1 gets msg_id=42, alert 2 gets msg_id=43
    msg1 = MagicMock(); msg1.message_id = 42
    msg2 = MagicMock(); msg2.message_id = 43
    bot.send_message.side_effect = [msg1, msg2]

    await provider.send_alert(DIXON_ALERT)
    await provider.send_alert(RELIANCE_ALERT)

    assert provider._pending["DIXON"].message_id    == 42
    assert provider._pending["RELIANCE"].message_id == 43

    # Press YES on RELIANCE's message — should only affect RELIANCE
    query  = _make_callback_query("YES", message_id=43)
    update = _make_update(query)
    await provider._on_button_press(update, MagicMock())

    assert provider._pending["RELIANCE"].choice[0] == ApprovalChoice.YES
    assert provider._pending["RELIANCE"].event.is_set()

    # DIXON should be untouched
    assert not provider._pending["DIXON"].event.is_set()


# ══════════════════════════════════════════════════════════════════════════════
# /status command
# ══════════════════════════════════════════════════════════════════════════════

async def test_status_command_no_pending() -> None:
    provider, _ = _make_provider()

    update = MagicMock()
    update.message.reply_text = AsyncMock()
    await provider._on_status_command(update, MagicMock())

    text = update.message.reply_text.call_args.args[0]
    assert "No pending" in text


async def test_status_command_with_pending() -> None:
    provider, _ = _make_provider()
    await provider.send_alert(DIXON_ALERT)

    update = MagicMock()
    update.message.reply_text = AsyncMock()
    await provider._on_status_command(update, MagicMock())

    text = update.message.reply_text.call_args.args[0]
    assert "DIXON" in text


# ══════════════════════════════════════════════════════════════════════════════
# Constructor validation
# ══════════════════════════════════════════════════════════════════════════════

def test_constructor_raises_without_token() -> None:
    with (
        patch("trading_bot.config.TELEGRAM_BOT_TOKEN", ""),
        patch("trading_bot.config.TELEGRAM_CHAT_ID", "12345"),
        pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"),
    ):
        TelegramProvider()


def test_constructor_raises_without_chat_id() -> None:
    with (
        patch("trading_bot.config.TELEGRAM_BOT_TOKEN", "tok"),
        patch("trading_bot.config.TELEGRAM_CHAT_ID", ""),
        pytest.raises(RuntimeError, match="TELEGRAM_CHAT_ID"),
    ):
        TelegramProvider()
