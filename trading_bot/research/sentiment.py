"""
research/sentiment.py — Sentiment scoring for news headlines and Reddit posts.

Uses VADER (fast, no GPU needed) as the primary scorer.
FinBERT is used when available and the text is longer than 20 words — it is
more accurate for financial text but much slower. Falls back to VADER silently.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from loguru import logger

# ── VADER ─────────────────────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    _VADER_OK = True
except ImportError:
    _vader = None
    _VADER_OK = False
    logger.warning("vaderSentiment not installed — sentiment scoring disabled.")

# ── FinBERT (optional, lazy-loaded) ──────────────────────────────────────────
_finbert_pipe = None
_FINBERT_TRIED = False


def _load_finbert():
    global _finbert_pipe, _FINBERT_TRIED
    if _FINBERT_TRIED:
        return _finbert_pipe
    _FINBERT_TRIED = True
    try:
        from transformers import pipeline
        _finbert_pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded successfully.")
    except Exception as exc:
        logger.debug("FinBERT not available ({}). Using VADER only.", exc)
        _finbert_pipe = None
    return _finbert_pipe


# ── Helpers ────────────────────────────────────────────────────────────────────

_LABEL_MAP = {
    # FinBERT labels
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
    # VADER compound already -1..+1
}


def _finbert_score(text: str) -> Optional[float]:
    """Return compound-equivalent (-1..+1) from FinBERT, or None on failure."""
    pipe = _load_finbert()
    if pipe is None:
        return None
    try:
        result = pipe(text[:512])[0]
        label = result["label"].lower()
        score = result["score"]   # confidence 0..1
        direction = _LABEL_MAP.get(label, 0.0)
        return round(direction * score, 4)
    except Exception:
        return None


def _vader_score(text: str) -> float:
    """Return VADER compound score (-1..+1)."""
    if not _VADER_OK or _vader is None:
        return 0.0
    return round(_vader.polarity_scores(text)["compound"], 4)


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def score_text(text: str, use_finbert: bool = False) -> tuple[float, str]:
    """
    Score a piece of text for financial sentiment.

    Args:
        text:        Headline or body text to score.
        use_finbert: Try FinBERT first (slower, more accurate) if True.

    Returns:
        (compound_score, label) where compound ∈ [-1, +1] and
        label ∈ {"bullish", "bearish", "neutral"}.
    """
    if not text or not text.strip():
        return 0.0, "neutral"

    compound: float = 0.0

    if use_finbert and _word_count(text) > 8:
        fb = _finbert_score(text)
        if fb is not None:
            compound = fb
        else:
            compound = _vader_score(text)
    else:
        compound = _vader_score(text)

    if compound >= 0.05:
        label = "bullish"
    elif compound <= -0.05:
        label = "bearish"
    else:
        label = "neutral"

    return compound, label


def aggregate_sentiment(scores: list[float]) -> float:
    """Simple mean of compound scores; 0.0 for empty list."""
    return round(sum(scores) / len(scores), 4) if scores else 0.0
