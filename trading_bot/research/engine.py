"""
research/engine.py — Research Engine orchestrator.

Pipeline per stock:
  1. Technical analysis  (mcp_server tools — pure yfinance + pandas)
  2. Fundamentals        (yfinance info)
  3. News + sentiment    (yfinance news → VADER)
  4. Reddit sentiment    (PRAW, optional)
  5. Claude AI analysis  (anthropic SDK, optional — falls back to rule-based)
  6. Build + save ResearchReport

Usage (CLI):
    python -m trading_bot.research.engine DIXON NSE

Usage (programmatic):
    from trading_bot.research.engine import ResearchEngine
    engine = ResearchEngine()
    report = await engine.research("DIXON", "NSE")
    print(report.model_dump_json(indent=2))
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from typing import Optional

from loguru import logger

from trading_bot import config
from trading_bot.mcp_server import (
    analyze_stock,
    get_support_resistance,
)
from .fundamentals import fetch_fundamentals
from .news_fetcher import fetch_all_news
from .report import (
    Fundamentals,
    NewsItem,
    Recommendation,
    RedditPost,
    ResearchReport,
    TechnicalSummary,
    TradingStyle,
)
from .sentiment import aggregate_sentiment


# ── Claude prompt template ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior equity analyst specialising in Indian stock markets (NSE/BSE).
You receive structured data about a stock — technical analysis, fundamentals, and
recent news sentiment — and must produce a concise, actionable research opinion.

Your output must be valid JSON and nothing else. Schema:
{
  "recommendation": "BUY" | "SELL" | "WATCH" | "SKIP",
  "trading_style": "SWING" | "SWING_LIGHT" | "DAY_TRADE" | null,
  "confidence_score": <float 0.0–1.0>,
  "entry_price": <float or null>,
  "stop_loss": <float or null>,
  "target_1": <float or null>,
  "target_2": <float or null>,
  "reasoning": "<3–5 sentence plain-English rationale>"
}

Guidelines:
- BUY: clear uptrend, volume confirmation, positive fundamentals, ≥ 1:2 R:R
- WATCH: good setup forming but not ready; needs confirmation
- SKIP: no edge, mixed signals, or risky fundamentals
- SELL: only if specifically asked to evaluate an existing position
- Entry should be current price or limit below; SL below nearest support; T1/T2 at resistance
- trading_style: SWING (large-cap, 5–30 days), SWING_LIGHT (mid-cap), DAY_TRADE (small-cap / volatile)
- confidence_score: weight technical (50%) + fundamental (30%) + sentiment (20%)
"""

_USER_PROMPT_TPL = """\
Analyse {symbol} ({exchange}) and provide your recommendation.

## Technical Analysis
{technical_json}

## Fundamentals
{fundamentals_json}

## Recent News ({news_count} articles, avg sentiment: {news_sentiment:.2f})
{news_headlines}

## Reddit Sentiment ({reddit_count} posts, avg sentiment: {reddit_sentiment:.2f})
{reddit_titles}

Current price: ₹{price:,.2f}
Nearest support: ₹{support} | Nearest resistance: ₹{resistance}

Respond with JSON only.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════════════

class ResearchEngine:
    """
    Orchestrates the full research pipeline for a single stock.

    Claude API is used if ANTHROPIC_API_KEY is set; otherwise a rule-based
    fallback produces a recommendation from the technical bias score.
    """

    def __init__(self) -> None:
        self._client = None
        if config.ANTHROPIC_API_KEY:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
                logger.info("ResearchEngine: Claude API ready.")
            except ImportError:
                logger.warning("anthropic SDK not installed — using rule-based fallback.")
        else:
            logger.info("ANTHROPIC_API_KEY not set — using rule-based fallback.")

    # ── Public API ────────────────────────────────────────────────────────────

    async def research(self, symbol: str, exchange: str = "NSE") -> ResearchReport:
        """Run full research pipeline and return a saved ResearchReport."""
        symbol = symbol.upper()
        exchange = exchange.upper()
        logger.info("Research started: {} ({})", symbol, exchange)

        # 1. Technical (run in executor so async callers aren't blocked)
        loop = asyncio.get_event_loop()
        ta_data = await loop.run_in_executor(None, analyze_stock, symbol, exchange)
        sr_data = await loop.run_in_executor(None, get_support_resistance, symbol, exchange)

        if "error" in ta_data:
            logger.error("Technical analysis failed for {}: {}", symbol, ta_data["error"])
            raise ValueError(f"Cannot research {symbol}: {ta_data['error']}")

        tech = _build_technical_summary(ta_data, sr_data)
        data_sources = ["yfinance_ta"]

        # 2. Fundamentals
        fund = await loop.run_in_executor(None, fetch_fundamentals, symbol, exchange)
        data_sources.append("yfinance_fundamentals")

        # 3. News + Reddit
        news_items, reddit_posts = await loop.run_in_executor(
            None, fetch_all_news, symbol, exchange, ""
        )
        if news_items:
            data_sources.append("yfinance_news")
        if reddit_posts:
            data_sources.append("reddit_praw")

        news_scores = [n.sentiment_score for n in news_items]
        reddit_scores = [r.sentiment_score for r in reddit_posts]
        news_sentiment = aggregate_sentiment(news_scores)
        reddit_sentiment = aggregate_sentiment(reddit_scores)
        combined_sentiment = round(
            0.6 * news_sentiment + 0.4 * reddit_sentiment
            if reddit_scores
            else news_sentiment,
            4,
        )

        # 4. AI / rule-based recommendation
        if self._client:
            ai_result, raw = await loop.run_in_executor(
                None,
                self._claude_recommend,
                symbol, exchange, tech, fund, news_items, reddit_posts,
                news_sentiment, reddit_sentiment,
            )
            data_sources.append("claude_ai")
        else:
            ai_result, raw = _rule_based_recommend(tech, fund)

        # 5. Build report
        report = ResearchReport(
            symbol=symbol,
            exchange=exchange,
            technical=tech,
            fundamentals=fund,
            news=news_items[:20],
            reddit=reddit_posts[:15],
            news_sentiment_score=news_sentiment,
            reddit_sentiment_score=reddit_sentiment,
            combined_sentiment_score=combined_sentiment,
            recommendation=ai_result.get("recommendation", Recommendation.SKIP),
            trading_style=ai_result.get("trading_style"),
            confidence_score=float(ai_result.get("confidence_score", 0.0)),
            entry_price=ai_result.get("entry_price"),
            stop_loss=ai_result.get("stop_loss"),
            target_1=ai_result.get("target_1"),
            target_2=ai_result.get("target_2"),
            position_size_pct=_position_size(ai_result.get("trading_style"), fund),
            reasoning=ai_result.get("reasoning", ""),
            ai_raw_response=raw,
            data_sources=data_sources,
        )

        path = report.save()
        logger.success("Research complete for {}: {} (confidence {:.0f}%) → saved to {}",
                       symbol, report.recommendation, report.confidence_score * 100, path)
        return report

    # ── Claude API call ───────────────────────────────────────────────────────

    def _claude_recommend(
        self,
        symbol: str,
        exchange: str,
        tech: TechnicalSummary,
        fund: Fundamentals,
        news: list[NewsItem],
        reddit: list[RedditPost],
        news_sentiment: float,
        reddit_sentiment: float,
    ) -> tuple[dict, str]:
        sr = get_support_resistance(symbol, exchange)
        supports = sr.get("support_levels", [])
        resistances = sr.get("resistance_levels", [])
        nearest_s = supports[0]["price"] if supports else "N/A"
        nearest_r = resistances[0]["price"] if resistances else "N/A"

        news_headlines = "\n".join(
            f"  [{n.sentiment_label:7s}] {n.title[:80]} ({n.source})"
            for n in news[:10]
        ) or "  (no news available)"

        reddit_titles = "\n".join(
            f"  [{r.sentiment_label:7s}] {r.title[:80]} ({r.subreddit})"
            for r in reddit[:8]
        ) or "  (no reddit posts)"

        user_msg = _USER_PROMPT_TPL.format(
            symbol=symbol,
            exchange=exchange,
            technical_json=json.dumps({
                "bias": tech.trading_bias,
                "score": tech.score,
                "trend": tech.trend,
                "rsi": tech.rsi,
                "macd_cross": tech.macd_cross,
                "supertrend": tech.supertrend,
                "volume": tech.volume_label,
                "is_breakout": tech.is_breakout,
                "breakout_strength": tech.breakout_strength,
                "atr_pct": tech.atr_pct,
            }, indent=2),
            fundamentals_json=json.dumps({
                "market_cap_cr": fund.market_cap_cr,
                "pe_ratio": fund.pe_ratio,
                "pb_ratio": fund.pb_ratio,
                "eps_ttm": fund.eps_ttm,
                "roe_pct": fund.roe_pct,
                "revenue_growth_yoy_pct": fund.revenue_growth_yoy_pct,
                "debt_to_equity": fund.debt_to_equity,
                "sector": fund.sector,
                "industry": fund.industry,
            }, indent=2),
            news_count=len(news),
            news_sentiment=news_sentiment,
            news_headlines=news_headlines,
            reddit_count=len(reddit),
            reddit_sentiment=reddit_sentiment,
            reddit_titles=reddit_titles,
            price=tech.price,
            support=nearest_s,
            resistance=nearest_r,
        )

        try:
            import anthropic
            response = self._client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            return _normalise_ai_output(parsed), raw
        except Exception as exc:
            logger.warning("Claude API call failed ({}), falling back to rule-based.", exc)
            result, fallback_raw = _rule_based_recommend(tech, fund)
            return result, fallback_raw


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_technical_summary(ta: dict, sr: dict) -> TechnicalSummary:
    quote = ta.get("quote", {})
    indicators = ta.get("indicators", {})
    breakout = ta.get("breakout_scan", {})
    sigs = indicators.get("signals", {})

    supports = sr.get("support_levels", [])
    resistances = sr.get("resistance_levels", [])

    return TechnicalSummary(
        price=quote.get("price", ta.get("quote", {}).get("price", 0.0)),
        change_pct=quote.get("change_pct", 0.0),
        trading_bias=ta.get("trading_bias", "NEUTRAL"),
        score=ta.get("score", 0),
        trend=sigs.get("trend", "NEUTRAL"),
        rsi=indicators.get("rsi", {}).get("value"),
        macd_cross=sigs.get("macd_cross"),
        supertrend=sigs.get("supertrend", ""),
        volume_label=indicators.get("volume", {}).get("label", ""),
        is_breakout=breakout.get("is_breakout", False),
        breakout_strength=breakout.get("breakout_strength", ""),
        nearest_support=supports[0]["price"] if supports else None,
        nearest_resistance=resistances[0]["price"] if resistances else None,
        atr=indicators.get("atr", {}).get("value"),
        atr_pct=indicators.get("atr", {}).get("atr_pct"),
    )


def _normalise_ai_output(parsed: dict) -> dict:
    """Ensure Recommendation and TradingStyle enums are valid strings."""
    rec = parsed.get("recommendation", "SKIP").upper()
    if rec not in Recommendation._value2member_map_:
        rec = "SKIP"

    style = parsed.get("trading_style")
    if style and style.upper() not in TradingStyle._value2member_map_:
        style = None

    return {
        "recommendation": Recommendation(rec),
        "trading_style": TradingStyle(style.upper()) if style else None,
        "confidence_score": min(1.0, max(0.0, float(parsed.get("confidence_score", 0.5)))),
        "entry_price": parsed.get("entry_price"),
        "stop_loss": parsed.get("stop_loss"),
        "target_1": parsed.get("target_1"),
        "target_2": parsed.get("target_2"),
        "reasoning": parsed.get("reasoning", ""),
    }


def _rule_based_recommend(
    tech: TechnicalSummary,
    fund: Fundamentals,
) -> tuple[dict, str]:
    """
    Deterministic fallback when Claude API is unavailable.

    Scores:  technical bias score (-4..+4) + sentiment adjustments.
    """
    score = tech.score

    # Fundamental adjustments
    if fund.pe_ratio and fund.pe_ratio > 80:
        score -= 1
    if fund.roe_pct and fund.roe_pct > 15:
        score += 1
    if fund.debt_to_equity and fund.debt_to_equity > 2:
        score -= 1
    if tech.is_breakout:
        score += 1

    if score >= 3:
        rec = Recommendation.BUY
        confidence = 0.72
    elif score >= 1:
        rec = Recommendation.WATCH
        confidence = 0.55
    elif score <= -2:
        rec = Recommendation.SKIP
        confidence = 0.65
    else:
        rec = Recommendation.WATCH
        confidence = 0.40

    # Trading style from market cap
    cap = fund.market_cap_cr or 0
    if cap > 20_000:
        style = TradingStyle.SWING
    elif cap > 5_000:
        style = TradingStyle.SWING_LIGHT
    else:
        style = TradingStyle.DAY_TRADE

    price = tech.price
    atr = tech.atr or (price * 0.015)   # default 1.5% ATR

    entry = round(price, 2) if rec == Recommendation.BUY else None
    sl = round(price - 1.5 * atr, 2) if entry else None
    t1 = round(price + 2.0 * atr, 2) if entry else None
    t2 = round(price + 3.5 * atr, 2) if entry else None

    reasoning = (
        f"Rule-based analysis (Claude API unavailable). "
        f"Technical score {tech.score}/4 → bias {tech.trading_bias}. "
        f"Trend: {tech.trend}. RSI: {tech.rsi:.0f}. "
        f"Supertrend: {tech.supertrend}. "
        f"Volume: {tech.volume_label}. "
        + ("BREAKOUT DETECTED. " if tech.is_breakout else "")
        + f"Recommendation: {rec.value} with {confidence*100:.0f}% confidence."
    )

    raw = json.dumps({
        "source": "rule_based_fallback",
        "technical_score": tech.score,
        "adjusted_score": score,
        "recommendation": rec.value,
    })

    return {
        "recommendation": rec,
        "trading_style": style,
        "confidence_score": confidence,
        "entry_price": entry,
        "stop_loss": sl,
        "target_1": t1,
        "target_2": t2,
        "reasoning": reasoning,
    }, raw


def _position_size(style: Optional[TradingStyle], fund: Fundamentals) -> Optional[float]:
    if style is None:
        return None
    return config.POSITION_LIMITS.get(style.value)


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

async def _cli_main(symbol: str, exchange: str) -> None:
    from loguru import logger as log
    log.remove()
    log.add(sys.stderr, level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    engine = ResearchEngine()
    report = await engine.research(symbol, exchange)

    # Pretty-print to stdout
    print("\n" + "=" * 70)
    print(f"  RESEARCH REPORT: {report.symbol} ({report.exchange})")
    print("=" * 70)
    print(report.model_dump_json(indent=2))
    print("=" * 70)


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "DIXON"
    exc = sys.argv[2] if len(sys.argv) > 2 else "NSE"
    asyncio.run(_cli_main(sym, exc))
