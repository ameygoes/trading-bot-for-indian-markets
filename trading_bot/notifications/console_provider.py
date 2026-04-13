"""
notifications/console_provider.py — Stdout-only provider for local dev/testing.

No external dependencies.  Approval responses are read from stdin so you can
manually type YES / NO / WAIT / DETAILS during development.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from .base import (
    ApprovalChoice,
    ApprovalResponse,
    NotificationProvider,
    StockAlert,
)

_DIVIDER = "─" * 50


class ConsoleProvider(NotificationProvider):
    """Prints messages to stdout; reads approval replies from stdin."""

    # ── Core interface ────────────────────────────────────────────────────────

    async def send_message(self, text: str, parse_mode: str = "text") -> None:
        logger.info("[CONSOLE] {}", text)
        print(f"\n{_DIVIDER}\n{text}\n{_DIVIDER}\n")

    async def send_document(self, path: Path, caption: str = "") -> None:
        msg = f"[DOCUMENT] {path}"
        if caption:
            msg += f"\n  Caption: {caption}"
        logger.info(msg)
        print(f"\n{_DIVIDER}\n{msg}\n{_DIVIDER}\n")

    async def send_alert(self, alert: StockAlert) -> None:
        text = alert.to_text()
        logger.info("[ALERT] {}", alert.symbol)
        print(f"\n{text}\n")
        print("  Respond with: YES | NO | WAIT | DETAILS")

    async def poll_reply(
        self,
        alert: StockAlert,
        timeout_seconds: int = 300,
    ) -> ApprovalResponse:
        """
        Read a line from stdin within *timeout_seconds*.

        In non-interactive / automated contexts (e.g. unit tests) where stdin
        is not a tty, we immediately return NO to avoid hanging.
        """
        import sys

        if not sys.stdin.isatty():
            logger.warning(
                "[CONSOLE] Non-interactive stdin — auto-declining {}.", alert.symbol
            )
            return ApprovalResponse(
                choice=ApprovalChoice.NO,
                symbol=alert.symbol,
                timed_out=True,
            )

        print(f"\n  ⏱  You have {timeout_seconds}s to respond > ", end="", flush=True)

        try:
            raw = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, input),
                timeout=float(timeout_seconds),
            )
        except asyncio.TimeoutError:
            print("\n  [timeout] Treating as NO.")
            logger.warning("[CONSOLE] Approval timeout for {}.", alert.symbol)
            return ApprovalResponse(
                choice=ApprovalChoice.NO,
                symbol=alert.symbol,
                timed_out=True,
            )

        raw = raw.strip().upper()
        try:
            choice = ApprovalChoice(raw)
        except ValueError:
            logger.warning("[CONSOLE] Unrecognised reply '{}' — treating as NO.", raw)
            choice = ApprovalChoice.NO

        logger.info("[CONSOLE] {} → {}", alert.symbol, choice.value)
        return ApprovalResponse(choice=choice, symbol=alert.symbol)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        logger.info("[CONSOLE] Provider ready (stdout/stdin mode).")

    async def stop(self) -> None:
        logger.info("[CONSOLE] Provider stopped.")
