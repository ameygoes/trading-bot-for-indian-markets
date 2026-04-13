"""
research/report.py — ResearchReport Pydantic model.

This is the single gating artefact: a stock cannot enter the watchlist or
trigger a notification unless a valid ResearchReport exists for it.

Saved as JSON to data/research_reports/<SYMBOL>_<date>.json
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from trading_bot import config


class Recommendation(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    WATCH = "WATCH"      # interesting but not ready yet
    SKIP = "SKIP"        # not worth tracking


class TradingStyle(str, Enum):
    SWING = "SWING"
    SWING_LIGHT = "SWING_LIGHT"
    DAY_TRADE = "DAY_TRADE"


class NewsItem(BaseModel):
    title: str
    source: str
    url: str = ""
    published_at: str = ""
    sentiment_score: float = 0.0   # VADER compound: -1 (bearish) to +1 (bullish)
    sentiment_label: str = "neutral"


class RedditPost(BaseModel):
    title: str
    subreddit: str
    upvotes: int = 0
    url: str = ""
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"


class Fundamentals(BaseModel):
    market_cap_cr: Optional[float] = None      # in Crores
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    eps_ttm: Optional[float] = None
    roe_pct: Optional[float] = None
    revenue_growth_yoy_pct: Optional[float] = None
    debt_to_equity: Optional[float] = None
    sector: str = ""
    industry: str = ""
    promoter_holding_pct: Optional[float] = None


class TechnicalSummary(BaseModel):
    price: float
    change_pct: float
    trading_bias: str                          # STRONG_BULLISH / BULLISH / NEUTRAL / etc.
    score: int                                 # -4 to +4
    trend: str
    rsi: Optional[float] = None
    macd_cross: Optional[str] = None
    supertrend: str = ""
    volume_label: str = ""
    is_breakout: bool = False
    breakout_strength: str = ""
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    atr: Optional[float] = None
    atr_pct: Optional[float] = None


class ResearchReport(BaseModel):
    # ── Identity ──────────────────────────────────────────────────────────────
    symbol: str
    exchange: str
    generated_at: str = Field(
        default_factory=lambda: datetime.now(config.IST).isoformat()
    )

    # ── Analysis layers ───────────────────────────────────────────────────────
    technical: TechnicalSummary
    fundamentals: Fundamentals = Field(default_factory=Fundamentals)
    news: list[NewsItem] = Field(default_factory=list)
    reddit: list[RedditPost] = Field(default_factory=list)

    # ── Sentiment aggregates ──────────────────────────────────────────────────
    news_sentiment_score: float = 0.0       # mean VADER compound across news items
    reddit_sentiment_score: float = 0.0     # mean VADER compound across reddit posts
    combined_sentiment_score: float = 0.0   # weighted blend

    # ── AI / rule-based recommendation ───────────────────────────────────────
    recommendation: Recommendation = Recommendation.SKIP
    trading_style: Optional[TradingStyle] = None
    confidence_score: float = 0.0           # 0–1

    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    position_size_pct: Optional[float] = None   # fraction of portfolio

    reasoning: str = ""                     # 3–5 sentence human-readable rationale
    ai_raw_response: str = ""               # full Claude response for audit trail

    # ── Metadata ──────────────────────────────────────────────────────────────
    data_sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> Path:
        """Write report to data/research_reports/<SYMBOL>_<date>.json"""
        config.RESEARCH_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(config.IST).strftime("%Y-%m-%d")
        path = config.RESEARCH_REPORTS_DIR / f"{self.symbol}_{date_str}.json"
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "ResearchReport":
        return cls.model_validate_json(path.read_text())

    @classmethod
    def latest_for(cls, symbol: str) -> Optional["ResearchReport"]:
        """Load the most recent report for *symbol*, or None if none exists."""
        reports = sorted(
            config.RESEARCH_REPORTS_DIR.glob(f"{symbol.upper()}_*.json"),
            reverse=True,
        )
        return cls.load(reports[0]) if reports else None
