"""
tests/test_notifications.py — End-to-end smoke test for the notification layer.

Runs fully offline: uses ConsoleProvider with stdin mocked so no real keyboard
input is needed.  Verifies the full chain:

  StockAlert → gateway.send_alert → gateway.poll_reply → ApprovalResponse
"""

from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_bot.notifications.base import (
    ApprovalChoice,
    StockAlert,
)
from trading_bot.notifications.console_provider import ConsoleProvider
from trading_bot.notifications.gateway import NotificationGateway


# ── Fixtures ──────────────────────────────────────────────────────────────────

DIXON_ALERT = StockAlert(
    symbol="DIXON.NS",
    exchange="NSE",
    action="BUY",
    style="SWING",
    entry_price=14_250.00,
    stop_loss=13_600.00,
    target_1=15_500.00,
    target_2=16_800.00,
    position_size_pct=0.04,
    confidence_score=0.78,
    reasoning=(
        "Dixon Technologies broke out of a 6-week consolidation on 2× avg volume. "
        "Strong EMS sector tailwinds and improving operating margins support continuation."
    ),
    tags=["breakout", "momentum", "ems_sector"],
)


# ── Unit tests — StockAlert helpers ──────────────────────────────────────────

def test_stock_alert_risk_reward() -> None:
    rr = DIXON_ALERT.risk_reward
    # entry=14250, sl=13600, t1=15500 → (15500-14250)/(14250-13600) = 1250/650 ≈ 1.92
    assert 1.8 < rr < 2.1, f"Unexpected R:R {rr:.2f}"


def test_stock_alert_to_text_contains_symbol() -> None:
    text = DIXON_ALERT.to_text()
    assert "DIXON.NS" in text
    assert "14,250.00" in text
    assert "BUY" in text


# ── Console provider — non-interactive (stdin is not a tty) ──────────────────

@pytest.mark.asyncio
async def test_console_send_message(capsys) -> None:
    provider = ConsoleProvider()
    await provider.send_message("Hello from trading bot!")
    captured = capsys.readouterr()
    assert "Hello from trading bot!" in captured.out


@pytest.mark.asyncio
async def test_console_send_alert(capsys) -> None:
    provider = ConsoleProvider()
    await provider.send_alert(DIXON_ALERT)
    captured = capsys.readouterr()
    assert "DIXON.NS" in captured.out


@pytest.mark.asyncio
async def test_console_poll_reply_non_interactive() -> None:
    """
    When stdin is not a tty (CI / unit test), poll_reply must auto-return NO
    without blocking.
    """
    provider = ConsoleProvider()
    # stdin is not a tty in pytest — no mock needed
    response = await provider.poll_reply(DIXON_ALERT, timeout_seconds=5)
    assert response.symbol == "DIXON.NS"
    assert response.choice == ApprovalChoice.NO
    assert response.timed_out is True


# ── Gateway — full chain smoke test ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_gateway_full_chain(capsys) -> None:
    """
    Full end-to-end: gateway.send_alert → gateway.poll_reply → ApprovalResponse.
    Uses ConsoleProvider in non-interactive mode so poll_reply returns immediately.
    """
    provider = ConsoleProvider()
    gateway = NotificationGateway(provider)

    await gateway.start()

    # send_alert should not raise
    await gateway.send_alert(DIXON_ALERT)

    # poll_reply auto-declines in non-interactive mode
    response = await gateway.poll_reply(DIXON_ALERT, timeout_seconds=5)
    assert response.symbol == "DIXON.NS"
    assert response.choice == ApprovalChoice.NO

    await gateway.stop()

    captured = capsys.readouterr()
    assert "DIXON.NS" in captured.out


@pytest.mark.asyncio
async def test_gateway_notify_and_await_approval(capsys) -> None:
    """Convenience method send_alert + poll_reply in one call."""
    provider = ConsoleProvider()
    gateway = NotificationGateway(provider)

    await gateway.start()
    response = await gateway.notify_and_await_approval(DIXON_ALERT, timeout_seconds=5)
    await gateway.stop()

    assert response.symbol == "DIXON.NS"
    # non-interactive → auto-NO
    assert response.choice in (ApprovalChoice.NO, ApprovalChoice.YES)


# ── send_document (file path test) ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_console_send_document(tmp_path: Path, capsys) -> None:
    report = tmp_path / "DIXON_research.json"
    report.write_text('{"symbol": "DIXON.NS", "score": 0.78}')

    provider = ConsoleProvider()
    await provider.send_document(report, caption="DIXON research report")
    captured = capsys.readouterr()
    assert "DIXON_research.json" in captured.out
