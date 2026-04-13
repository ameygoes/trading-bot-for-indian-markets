"""
research/news_fetcher.py — Fetch recent news for an NSE/BSE stock.

Sources (tried in order, results merged):
  1. yfinance .news  — quick, no API key required
  2. SerpAPI Google News — richer results (optional, requires SERPAPI_KEY)
  3. Reddit via PRAW  — social sentiment (optional, requires REDDIT_* env vars)

Returns lists of NewsItem and RedditPost ready for sentiment scoring.
"""

from __future__ import annotations

from typing import Optional

import yfinance as yf
from loguru import logger

from trading_bot import config
from .report import NewsItem, RedditPost
from .sentiment import score_text


# ══════════════════════════════════════════════════════════════════════════════
# yfinance news
# ══════════════════════════════════════════════════════════════════════════════

def fetch_yfinance_news(symbol: str, exchange: str = "NSE", limit: int = 15) -> list[NewsItem]:
    """Pull news from yfinance ticker.news (no API key required)."""
    suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
    yf_sym = f"{symbol.upper()}{suffix}" if "." not in symbol else symbol.upper()

    items: list[NewsItem] = []
    try:
        ticker = yf.Ticker(yf_sym)
        raw = ticker.news or []
        for article in raw[:limit]:
            content = article.get("content", {})
            title = (
                content.get("title", "")
                or article.get("title", "")
            )
            if not title:
                continue

            # provider / source
            provider = content.get("provider", {})
            source = (
                provider.get("displayName", "")
                or article.get("publisher", "")
                or "yfinance"
            )

            # URL
            canonical = content.get("canonicalUrl", {})
            url = (
                canonical.get("url", "")
                if isinstance(canonical, dict)
                else article.get("link", "")
            )

            # published timestamp
            pub_date = content.get("pubDate", "") or str(article.get("providerPublishTime", ""))

            compound, label = score_text(title)
            items.append(NewsItem(
                title=title,
                source=source,
                url=url,
                published_at=pub_date,
                sentiment_score=compound,
                sentiment_label=label,
            ))
    except Exception as exc:
        logger.warning("yfinance news fetch failed for {}: {}", symbol, exc)

    logger.debug("yfinance news: {} items for {}", len(items), symbol)
    return items


# ══════════════════════════════════════════════════════════════════════════════
# SerpAPI Google News (optional)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_serpapi_news(symbol: str, company_name: str = "", limit: int = 10) -> list[NewsItem]:
    """
    Fetch Google News via SerpAPI.
    Returns [] immediately if SERPAPI_KEY is not configured.
    """
    if not config.SERPAPI_KEY:
        return []

    query = f"{symbol} NSE stock news" if not company_name else f"{company_name} stock NSE"
    try:
        import httpx
        resp = httpx.get(
            "https://serpapi.com/search",
            params={
                "engine": "google_news",
                "q": query,
                "api_key": config.SERPAPI_KEY,
                "num": limit,
                "gl": "in",
                "hl": "en",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("SerpAPI news fetch failed: {}", exc)
        return []

    items: list[NewsItem] = []
    for article in data.get("news_results", [])[:limit]:
        title = article.get("title", "")
        if not title:
            continue
        compound, label = score_text(title)
        items.append(NewsItem(
            title=title,
            source=article.get("source", {}).get("name", ""),
            url=article.get("link", ""),
            published_at=article.get("date", ""),
            sentiment_score=compound,
            sentiment_label=label,
        ))

    logger.debug("SerpAPI news: {} items for {}", len(items), symbol)
    return items


# ══════════════════════════════════════════════════════════════════════════════
# Reddit via PRAW (optional)
# ══════════════════════════════════════════════════════════════════════════════

_SUBREDDITS = [
    "IndiaInvestments",
    "IndianStockMarket",
    "DalalStreetTalks",
    "stocks",
    "investing",
]


def fetch_reddit_posts(symbol: str, limit: int = 20) -> list[RedditPost]:
    """
    Search Reddit for recent posts mentioning *symbol*.
    Returns [] if REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not configured.
    """
    if not config.REDDIT_CLIENT_ID or not config.REDDIT_CLIENT_SECRET:
        logger.debug("Reddit creds not set — skipping Reddit fetch for {}", symbol)
        return []

    try:
        import praw
        reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT,
        )
        subreddit = reddit.subreddit("+".join(_SUBREDDITS))
        results = subreddit.search(symbol, sort="new", time_filter="week", limit=limit)
    except Exception as exc:
        logger.warning("Reddit fetch failed for {}: {}", symbol, exc)
        return []

    posts: list[RedditPost] = []
    for submission in results:
        title = submission.title or ""
        compound, label = score_text(title)
        posts.append(RedditPost(
            title=title,
            subreddit=f"r/{submission.subreddit.display_name}",
            upvotes=submission.score,
            url=f"https://reddit.com{submission.permalink}",
            sentiment_score=compound,
            sentiment_label=label,
        ))

    logger.debug("Reddit: {} posts for {}", len(posts), symbol)
    return posts


# ══════════════════════════════════════════════════════════════════════════════
# Combined fetch
# ══════════════════════════════════════════════════════════════════════════════

def fetch_all_news(
    symbol: str,
    exchange: str = "NSE",
    company_name: str = "",
) -> tuple[list[NewsItem], list[RedditPost]]:
    """
    Run all news sources and return (news_items, reddit_posts).
    Deduplicates news by title similarity.
    """
    yf_news = fetch_yfinance_news(symbol, exchange)
    serp_news = fetch_serpapi_news(symbol, company_name)
    reddit = fetch_reddit_posts(symbol)

    # Merge yfinance + SerpAPI, deduplicate by title prefix (first 40 chars)
    seen: set[str] = set()
    merged: list[NewsItem] = []
    for item in yf_news + serp_news:
        key = item.title[:40].lower().strip()
        if key not in seen:
            seen.add(key)
            merged.append(item)

    sources_used = ["yfinance"]
    if serp_news:
        sources_used.append("serpapi")
    if reddit:
        sources_used.append("reddit")

    logger.info(
        "News for {}: {} articles, {} reddit posts (sources: {})",
        symbol,
        len(merged),
        len(reddit),
        ", ".join(sources_used),
    )
    return merged, reddit
