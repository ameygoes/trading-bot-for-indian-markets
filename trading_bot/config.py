"""
config.py — Centralised configuration loader.

Reads .env via python-dotenv and exposes typed settings used across all modules.
All other modules import from here — never read os.environ directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val


def _bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


# ── Timezone ──────────────────────────────────────────────────────────────────
IST = ZoneInfo("Asia/Kolkata")

# ── Market Hours (IST) ────────────────────────────────────────────────────────
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# ── Trading Mode ──────────────────────────────────────────────────────────────
LIVE_TRADING: bool = _bool("LIVE_TRADING", False)

# ── Notification Provider ─────────────────────────────────────────────────────
NOTIFICATION_PROVIDER: str = os.getenv("NOTIFICATION_PROVIDER", "console")

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── OpenClaw ──────────────────────────────────────────────────────────────────
OPENCLAW_API_KEY: str = os.getenv("OPENCLAW_API_KEY", "")
OPENCLAW_WEBHOOK_URL: str = os.getenv("OPENCLAW_WEBHOOK_URL", "")

# ── Broker ────────────────────────────────────────────────────────────────────
KITE_API_KEY: str = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET: str = os.getenv("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN: str = os.getenv("KITE_ACCESS_TOKEN", "")

UPSTOX_API_KEY: str = os.getenv("UPSTOX_API_KEY", "")
UPSTOX_API_SECRET: str = os.getenv("UPSTOX_API_SECRET", "")
UPSTOX_ACCESS_TOKEN: str = os.getenv("UPSTOX_ACCESS_TOKEN", "")

# ── Social / News ─────────────────────────────────────────────────────────────
REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "IndianMarketBot/1.0")

TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN", "")
SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")

# ── LLM ───────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# ── Database ──────────────────────────────────────────────────────────────────
TRADES_DB_PATH: Path = _ROOT / os.getenv("TRADES_DB_PATH", "data/trades.db")
RESEARCH_DB_PATH: Path = _ROOT / os.getenv("RESEARCH_DB_PATH", "data/research.db")
PORTFOLIO_DB_PATH: Path = _ROOT / os.getenv("PORTFOLIO_DB_PATH", "data/portfolio.db")
RESEARCH_REPORTS_DIR: Path = _ROOT / "data" / "research_reports"
SYSTEM_LEARNINGS_PATH: Path = _ROOT / "data" / "system_learnings.json"

# ── MCP Server ────────────────────────────────────────────────────────────────
MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", "8080"))

# ── Position Sizing ───────────────────────────────────────────────────────────
# Market-cap tiers → trading style → max position %
POSITION_LIMITS = {
    "SWING": 0.04,       # large cap > 20,000 Cr
    "SWING_LIGHT": 0.025,  # mid cap > 5,000 Cr
    "DAY_TRADE": 0.015,  # small cap < 5,000 Cr
}
RISK_PER_TRADE_PCT = 0.02  # 2% portfolio risk per trade

# ── Discovery / Research Intervals ───────────────────────────────────────────
DISCOVERY_INTERVAL_HOURS = 4
MACRO_REFRESH_INTERVAL_HOURS = 6
RESEARCH_REFRESH_INTERVAL_DAYS = 7
ML_REOPTIMIZE_INTERVAL_DAYS = 14
PORTFOLIO_REVIEW_INTERVAL_DAYS = 30

# ── Watcher Intervals (seconds) ───────────────────────────────────────────────
WATCHER_INTERVAL_SWING_S = 60
WATCHER_INTERVAL_DAY_TRADE_S = 30
WATCHER_INTERVAL_NEAR_LEVEL_S = 10   # within 2% of SL or target

# ── Scheduled Message Times (IST, 24h) ───────────────────────────────────────
PREMARKET_BRIEF_TIME = (8, 45)
EOD_SUMMARY_TIME = (15, 35)
NIGHTLY_DIGEST_TIME = (22, 0)
