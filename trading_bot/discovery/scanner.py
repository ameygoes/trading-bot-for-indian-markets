"""
discovery/scanner.py — Stock discovery and candidate ranking.

Two discovery paths run in parallel and their results are merged:

  Path A — Social / News mining
    Scans Reddit posts (IndiaInvestments, IndianStockMarket, DalalStreetTalks)
    and yfinance news for the full watchlist to find symbols being actively
    discussed.  Mention count + sentiment score → social_score.

  Path B — Technical breakout scan
    Runs scan_breakout() from mcp_server for every symbol in the watchlist
    (or a supplied subset) and flags those with confirmed breakouts.

Both paths produce DiscoveryCandidate dataclass instances that are then merged
(by symbol), ranked by combined_score, and returned to the caller.

Usage:
    from trading_bot.discovery.scanner import DiscoveryScanner
    scanner = DiscoveryScanner()
    candidates = await scanner.run()
    for c in candidates[:10]:
        print(c.symbol, c.combined_score, c.triggers)

CLI:
    python -m trading_bot.discovery.scanner
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

from trading_bot import config
from trading_bot.mcp_server import scan_breakout
from trading_bot.research.news_fetcher import (
    fetch_reddit_posts,
    fetch_yfinance_news,
)
from trading_bot.research.sentiment import aggregate_sentiment
from .symbols import ALL_WATCHLIST, NIFTY50_SYMBOLS, extract_symbols_from_text


# ══════════════════════════════════════════════════════════════════════════════
# Data model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiscoveryCandidate:
    """A stock candidate produced by the discovery scanner."""

    symbol: str
    exchange: str = "NSE"

    # Social signals
    mention_count: int = 0          # total Reddit + news mentions
    social_sentiment: float = 0.0   # aggregate VADER compound (-1..+1)
    social_score: float = 0.0       # weighted: mentions × sentiment magnitude

    # Technical signals
    is_breakout: bool = False
    breakout_strength: str = ""
    technical_score: int = 0        # from analyze_stock (–4..+4)

    # Combined
    combined_score: float = 0.0     # final ranking score (higher = more interesting)
    triggers: list[str] = field(default_factory=list)   # human-readable reason list
    discovered_at: str = field(
        default_factory=lambda: datetime.now(config.IST).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "mention_count": self.mention_count,
            "social_sentiment": self.social_sentiment,
            "social_score": self.social_score,
            "is_breakout": self.is_breakout,
            "breakout_strength": self.breakout_strength,
            "technical_score": self.technical_score,
            "combined_score": self.combined_score,
            "triggers": self.triggers,
            "discovered_at": self.discovered_at,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Scanner
# ══════════════════════════════════════════════════════════════════════════════

class DiscoveryScanner:
    """
    Discovers NSE/BSE stock candidates via social mention mining and
    technical breakout detection.

    Args:
        watchlist:       Symbols to scan for breakouts. Defaults to ALL_WATCHLIST.
        reddit_subreddits: Reddit communities to mine. Defaults to India-focused subs.
        max_reddit_posts:  Posts to fetch per subreddit symbol query.
        min_combined_score: Minimum score to include a candidate in results.
        exchange:          Exchange for all symbols (NSE or BSE).
    """

    _DEFAULT_SUBREDDITS = [
        "IndiaInvestments",
        "IndianStockMarket",
        "DalalStreetTalks",
        "stocks",
    ]

    def __init__(
        self,
        watchlist: Optional[frozenset[str]] = None,
        reddit_subreddits: Optional[list[str]] = None,
        max_reddit_posts: int = 30,
        min_combined_score: float = 0.2,
        exchange: str = "NSE",
    ) -> None:
        self._watchlist = watchlist or ALL_WATCHLIST
        self._subreddits = reddit_subreddits or self._DEFAULT_SUBREDDITS
        self._max_reddit_posts = max_reddit_posts
        self._min_score = min_combined_score
        self._exchange = exchange.upper()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(
        self,
        run_social: bool = True,
        run_breakout: bool = True,
        breakout_symbols: Optional[list[str]] = None,
    ) -> list[DiscoveryCandidate]:
        """
        Run both discovery paths and return ranked DiscoveryCandidate list.

        Args:
            run_social:        Mine Reddit + news for mentions (network call).
            run_breakout:      Scan watchlist for technical breakouts (network).
            breakout_symbols:  Override which symbols to scan for breakouts.
                               Defaults to all NIFTY50_SYMBOLS (fast subset).

        Returns:
            Candidates sorted by combined_score descending, filtered to
            those above min_combined_score.
        """
        loop = asyncio.get_event_loop()
        candidates: dict[str, DiscoveryCandidate] = {}

        tasks = []
        if run_social:
            tasks.append(self._social_discovery(loop))
        if run_breakout:
            symbols_to_scan = breakout_symbols or list(NIFTY50_SYMBOLS)
            tasks.append(self._breakout_discovery(loop, symbols_to_scan))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning("Discovery sub-task failed: {}", result)
                continue
            for c in result:
                if c.symbol in candidates:
                    _merge(candidates[c.symbol], c)
                else:
                    candidates[c.symbol] = c

        # Final scoring + filtering
        ranked = []
        for c in candidates.values():
            c.combined_score = _compute_combined_score(c)
            if c.combined_score >= self._min_score:
                ranked.append(c)

        ranked.sort(key=lambda x: x.combined_score, reverse=True)
        logger.info(
            "Discovery complete: {} candidates above score {:.2f}",
            len(ranked), self._min_score,
        )
        return ranked

    # ── Social path ───────────────────────────────────────────────────────────

    async def _social_discovery(
        self, loop: asyncio.AbstractEventLoop
    ) -> list[DiscoveryCandidate]:
        """Mine Reddit posts and yfinance news headlines for symbol mentions."""
        mention_counts: dict[str, int] = {}
        sentiment_scores: dict[str, list[float]] = {}

        # A. Reddit scan (general market posts — not per-symbol queries)
        reddit_posts = await loop.run_in_executor(
            None, self._fetch_reddit_market_posts
        )
        for post in reddit_posts:
            syms = extract_symbols_from_text(post["text"])
            for sym in syms:
                mention_counts[sym] = mention_counts.get(sym, 0) + 1
                sentiment_scores.setdefault(sym, []).append(post["score"])

        # B. yfinance news — sample top active NSE symbols
        sample = list(NIFTY50_SYMBOLS[:20])
        news_results = await loop.run_in_executor(
            None, self._fetch_bulk_news, sample
        )
        for title, sent_score in news_results:
            syms = extract_symbols_from_text(title)
            for sym in syms:
                mention_counts[sym] = mention_counts.get(sym, 0) + 1
                sentiment_scores.setdefault(sym, []).append(sent_score)

        candidates = []
        for sym, count in mention_counts.items():
            scores = sentiment_scores.get(sym, [])
            avg_sent = aggregate_sentiment(scores)
            magnitude = abs(avg_sent)
            social_score = round(min(1.0, count * 0.1) * (0.5 + 0.5 * magnitude), 4)

            triggers = []
            if count >= 3:
                triggers.append(f"high_mentions:{count}")
            if avg_sent >= 0.2:
                triggers.append("positive_sentiment")
            elif avg_sent <= -0.2:
                triggers.append("negative_sentiment")

            c = DiscoveryCandidate(
                symbol=sym,
                exchange=self._exchange,
                mention_count=count,
                social_sentiment=round(avg_sent, 4),
                social_score=social_score,
                triggers=triggers,
            )
            candidates.append(c)

        logger.info("Social discovery: {} symbols mentioned", len(candidates))
        return candidates

    # ── Breakout path ─────────────────────────────────────────────────────────

    async def _breakout_discovery(
        self, loop: asyncio.AbstractEventLoop, symbols: list[str]
    ) -> list[DiscoveryCandidate]:
        """Scan symbols for technical breakouts using the MCP server."""
        candidates = []

        # Run breakout scans concurrently (but throttled to avoid yfinance rate limits)
        semaphore = asyncio.Semaphore(5)

        async def _scan_one(sym: str) -> Optional[DiscoveryCandidate]:
            async with semaphore:
                try:
                    result = await loop.run_in_executor(
                        None, scan_breakout, sym, self._exchange
                    )
                    if result.get("error"):
                        return None

                    is_bo = result.get("is_breakout", False)
                    strength = result.get("breakout_strength", "")
                    tech_score = result.get("score", 0)

                    if not is_bo and tech_score < 2:
                        return None  # skip unininteresting stocks early

                    triggers = []
                    if is_bo:
                        triggers.append(f"breakout:{strength}")
                    if tech_score >= 3:
                        triggers.append(f"tech_score:{tech_score}")
                    elif tech_score >= 2:
                        triggers.append(f"tech_score:{tech_score}")

                    return DiscoveryCandidate(
                        symbol=sym,
                        exchange=self._exchange,
                        is_breakout=is_bo,
                        breakout_strength=strength,
                        technical_score=tech_score,
                        triggers=triggers,
                    )
                except Exception as exc:
                    logger.debug("Breakout scan failed for {}: {}", sym, exc)
                    return None

        tasks = [_scan_one(sym) for sym in symbols]
        results = await asyncio.gather(*tasks)
        candidates = [r for r in results if r is not None]

        logger.info(
            "Breakout scan: {}/{} symbols have interesting technicals",
            len(candidates), len(symbols),
        )
        return candidates

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fetch_reddit_market_posts(self) -> list[dict]:
        """
        Fetch general market discussion posts from Indian investment subreddits.
        Returns list of {text, score} dicts.
        """
        if not config.REDDIT_CLIENT_ID or not config.REDDIT_CLIENT_SECRET:
            logger.debug("Reddit creds not configured — skipping Reddit discovery")
            return []

        posts = []
        try:
            import praw
            reddit = praw.Reddit(
                client_id=config.REDDIT_CLIENT_ID,
                client_secret=config.REDDIT_CLIENT_SECRET,
                user_agent=config.REDDIT_USER_AGENT,
            )
            sub = reddit.subreddit("+".join(self._subreddits))
            for submission in sub.hot(limit=self._max_reddit_posts):
                text = f"{submission.title} {submission.selftext or ''}"
                from trading_bot.research.sentiment import score_text
                compound, _ = score_text(submission.title)
                posts.append({"text": text, "score": compound})
        except Exception as exc:
            logger.warning("Reddit market post fetch failed: {}", exc)

        return posts

    def _fetch_bulk_news(self, symbols: list[str]) -> list[tuple[str, float]]:
        """
        Fetch yfinance news for a list of symbols.
        Returns list of (title, sentiment_score) tuples.
        """
        results = []
        for sym in symbols:
            try:
                items = fetch_yfinance_news(sym, self._exchange, limit=5)
                for item in items:
                    results.append((item.title, item.sentiment_score))
            except Exception as exc:
                logger.debug("News fetch failed for {}: {}", sym, exc)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Scoring helpers
# ══════════════════════════════════════════════════════════════════════════════

def _merge(base: DiscoveryCandidate, other: DiscoveryCandidate) -> None:
    """Merge *other* into *base* in-place (union triggers, sum mentions)."""
    base.mention_count += other.mention_count
    # weighted average of social sentiments
    total = base.mention_count + other.mention_count
    if total > 0:
        base.social_sentiment = round(
            (base.social_sentiment * base.mention_count +
             other.social_sentiment * other.mention_count) / total,
            4,
        )
    base.social_score = max(base.social_score, other.social_score)
    base.is_breakout = base.is_breakout or other.is_breakout
    if other.breakout_strength:
        base.breakout_strength = other.breakout_strength
    if other.technical_score != 0:
        base.technical_score = other.technical_score
    # deduplicate triggers
    existing = set(base.triggers)
    for t in other.triggers:
        if t not in existing:
            base.triggers.append(t)
            existing.add(t)


def _compute_combined_score(c: DiscoveryCandidate) -> float:
    """
    Combined ranking score in [0, 1].

    Weights:
      50% — technical (breakout + score)
      30% — social score (mentions × sentiment magnitude)
      20% — sentiment direction bonus
    """
    # Technical component (0–1)
    tech = 0.0
    if c.is_breakout:
        tech += 0.6
    tech += min(0.4, max(0.0, c.technical_score / 4.0) * 0.4)
    tech = min(1.0, tech)

    # Social component (already 0–1 from scanner)
    social = min(1.0, c.social_score)

    # Sentiment direction bonus (positive → +bonus)
    sentiment_bonus = max(0.0, c.social_sentiment) * 0.5

    combined = round(0.5 * tech + 0.3 * social + 0.2 * sentiment_bonus, 4)
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

async def _cli_main() -> None:
    import sys
    from loguru import logger as log
    log.remove()
    log.add(sys.stderr, level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    scanner = DiscoveryScanner(min_combined_score=0.1)
    # For CLI: quick scan — only top-20 Nifty50 breakouts + social if creds available
    candidates = await scanner.run(
        run_social=True,
        run_breakout=True,
        breakout_symbols=list(NIFTY50_SYMBOLS[:20]),
    )

    print(f"\n{'='*60}")
    print(f"  DISCOVERY RESULTS  ({len(candidates)} candidates)")
    print(f"{'='*60}")
    for i, c in enumerate(candidates[:20], 1):
        print(
            f"  {i:2}. {c.symbol:<16} score={c.combined_score:.3f}  "
            f"{'BREAKOUT ' if c.is_breakout else ''}"
            f"mentions={c.mention_count}  "
            f"sentiment={c.social_sentiment:+.2f}  "
            f"triggers={c.triggers}"
        )
    print(f"{'='*60}\n")

    # Full JSON dump
    print(json.dumps([c.to_dict() for c in candidates], indent=2))


if __name__ == "__main__":
    asyncio.run(_cli_main())
