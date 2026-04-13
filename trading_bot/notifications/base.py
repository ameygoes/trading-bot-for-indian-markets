"""
notifications/base.py — Abstract base for all notification providers.

Every provider must implement:
  • send_message   — plain text or markdown message
  • send_document  — file attachment (e.g. research report PDF/JSON)
  • send_alert     — structured stock alert with approval buttons
  • poll_reply     — block until the user responds YES / NO / WAIT / DETAILS
                     (or a timeout elapses)

Providers are instantiated once by gateway.py and reused for the bot's lifetime.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class ApprovalChoice(str, enum.Enum):
    YES = "YES"
    NO = "NO"
    WAIT = "WAIT"          # "remind me at next candle"
    DETAILS = "DETAILS"    # "show me the full research report"


@dataclass
class StockAlert:
    """Structured payload sent to the user for every trade candidate."""

    symbol: str                    # e.g. "DIXON.NS"
    exchange: str                  # "NSE" | "BSE"
    action: str                    # "BUY" | "SELL" | "WATCH"
    style: str                     # "SWING" | "SWING_LIGHT" | "DAY_TRADE"

    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    position_size_pct: float       # fraction of portfolio, e.g. 0.04

    confidence_score: float        # 0–1 from research engine
    reasoning: str                 # 1–3 sentence human-readable rationale

    # Optional extras
    research_report_path: Optional[Path] = None
    tags: list[str] = field(default_factory=list)   # e.g. ["momentum", "breakout"]

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def risk_reward(self) -> float:
        if self.entry_price == self.stop_loss:
            return 0.0
        return (self.target_1 - self.entry_price) / abs(self.entry_price - self.stop_loss)

    def to_text(self) -> str:
        """Plain-text representation used by console and Telegram fallback."""
        rr = self.risk_reward
        lines = [
            f"{'='*48}",
            f"  ALERT: {self.action} {self.symbol} [{self.exchange}]",
            f"  Style : {self.style}",
            f"{'='*48}",
            f"  Entry      : ₹{self.entry_price:,.2f}",
            f"  Stop Loss  : ₹{self.stop_loss:,.2f}",
            f"  Target 1   : ₹{self.target_1:,.2f}",
            f"  Target 2   : ₹{self.target_2:,.2f}",
            f"  Size       : {self.position_size_pct*100:.1f}% of portfolio",
            f"  Confidence : {self.confidence_score*100:.0f}%",
            f"  R:R        : 1:{rr:.1f}",
            f"",
            f"  Reasoning: {self.reasoning}",
        ]
        if self.tags:
            lines.append(f"  Tags: {', '.join(self.tags)}")
        lines.append(f"{'='*48}")
        return "\n".join(lines)


@dataclass
class ApprovalResponse:
    choice: ApprovalChoice
    symbol: str
    timed_out: bool = False


class NotificationProvider(ABC):
    """Abstract base — all providers must implement these four methods."""

    # ── Core interface ────────────────────────────────────────────────────────

    @abstractmethod
    async def send_message(self, text: str, parse_mode: str = "text") -> None:
        """Send a plain (or markdown) text message."""

    @abstractmethod
    async def send_document(self, path: Path, caption: str = "") -> None:
        """Upload a file (research report JSON / PDF)."""

    @abstractmethod
    async def send_alert(self, alert: StockAlert) -> None:
        """Send a structured stock alert with actionable buttons."""

    @abstractmethod
    async def poll_reply(
        self,
        alert: StockAlert,
        timeout_seconds: int = 300,
    ) -> ApprovalResponse:
        """
        Block until the user responds to *alert* or *timeout_seconds* elapses.

        Returns an ApprovalResponse with timed_out=True on timeout (treated as NO).
        """

    # ── Optional lifecycle hooks (override as needed) ─────────────────────────

    async def start(self) -> None:
        """Called once at bot startup — e.g. set webhook, start polling."""

    async def stop(self) -> None:
        """Called once at graceful shutdown."""
