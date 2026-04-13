"""
tests/test_discovery.py — Unit tests for the discovery layer.

Fast tests only (no network calls):
  - Symbol extraction from free text
  - DiscoveryCandidate scoring helpers
  - DiscoveryScanner with mocked sub-tasks

Live tests (marked @pytest.mark.live):
  - Full scan on a small subset of Nifty50 symbols
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading_bot.discovery.symbols import (
    ALL_WATCHLIST,
    MIDCAP_WATCHLIST,
    NIFTY50_SYMBOLS,
    extract_symbols_from_text,
)
from trading_bot.discovery.scanner import (
    DiscoveryCandidate,
    DiscoveryScanner,
    _compute_combined_score,
    _merge,
)


# ── Symbol extraction ─────────────────────────────────────────────────────────

def test_extract_uppercase_ticker() -> None:
    syms = extract_symbols_from_text("I bought RELIANCE and INFY today")
    assert "RELIANCE" in syms
    assert "INFY" in syms


def test_extract_alias_company_name() -> None:
    syms = extract_symbols_from_text("Dixon Technologies is rallying hard this week")
    assert "DIXON" in syms


def test_extract_dollar_ticker() -> None:
    syms = extract_symbols_from_text("Feeling bullish on $TCS and $WIPRO right now")
    assert "TCS" in syms
    assert "WIPRO" in syms


def test_extract_case_insensitive_alias() -> None:
    syms = extract_symbols_from_text("HDFC bank looks strong on technicals")
    assert "HDFCBANK" in syms


def test_extract_multiple_aliases() -> None:
    syms = extract_symbols_from_text("State bank and SBI are both mentioned — same thing")
    assert "SBIN" in syms


def test_extract_unknown_ticker_excluded() -> None:
    # FAKECORP is not in watchlist
    syms = extract_symbols_from_text("FAKECORP is the next big thing")
    assert "FAKECORP" not in syms


def test_extract_empty_text() -> None:
    assert extract_symbols_from_text("") == set()


def test_extract_no_symbols() -> None:
    assert extract_symbols_from_text("nothing relevant here") == set()


# ── Watchlist sanity ──────────────────────────────────────────────────────────

def test_nifty50_count() -> None:
    assert len(NIFTY50_SYMBOLS) == 50


def test_all_watchlist_contains_nifty50() -> None:
    for sym in NIFTY50_SYMBOLS:
        assert sym in ALL_WATCHLIST


def test_all_watchlist_contains_midcap() -> None:
    for sym in MIDCAP_WATCHLIST:
        assert sym in ALL_WATCHLIST


def test_no_duplicate_symbols() -> None:
    combined = list(NIFTY50_SYMBOLS) + list(MIDCAP_WATCHLIST)
    assert len(combined) == len(set(combined)), "Duplicate symbols found across watchlists"


# ── DiscoveryCandidate scoring ────────────────────────────────────────────────

def test_compute_score_breakout_dominates() -> None:
    c = DiscoveryCandidate(
        symbol="DIXON",
        is_breakout=True,
        technical_score=3,
        social_score=0.0,
        social_sentiment=0.0,
    )
    score = _compute_combined_score(c)
    assert score > 0.3   # breakout alone should push score up


def test_compute_score_no_signals_low() -> None:
    c = DiscoveryCandidate(
        symbol="WIPRO",
        is_breakout=False,
        technical_score=0,
        social_score=0.0,
        social_sentiment=0.0,
    )
    score = _compute_combined_score(c)
    assert score == pytest.approx(0.0)


def test_compute_score_social_signal() -> None:
    c = DiscoveryCandidate(
        symbol="INFY",
        is_breakout=False,
        technical_score=0,
        social_score=0.8,
        social_sentiment=0.5,
    )
    score = _compute_combined_score(c)
    assert score > 0.1


def test_compute_score_bounded() -> None:
    c = DiscoveryCandidate(
        symbol="TCS",
        is_breakout=True,
        technical_score=4,
        social_score=1.0,
        social_sentiment=1.0,
    )
    score = _compute_combined_score(c)
    assert 0.0 <= score <= 1.0


# ── Merge helper ──────────────────────────────────────────────────────────────

def test_merge_triggers_deduplicated() -> None:
    base = DiscoveryCandidate(symbol="RELIANCE", triggers=["breakout:strong"])
    other = DiscoveryCandidate(symbol="RELIANCE", triggers=["breakout:strong", "tech_score:3"])
    _merge(base, other)
    assert base.triggers.count("breakout:strong") == 1
    assert "tech_score:3" in base.triggers


def test_merge_breakout_propagates() -> None:
    base = DiscoveryCandidate(symbol="INFY", is_breakout=False)
    other = DiscoveryCandidate(symbol="INFY", is_breakout=True, breakout_strength="4/5")
    _merge(base, other)
    assert base.is_breakout is True
    assert base.breakout_strength == "4/5"


def test_merge_mention_counts_sum() -> None:
    base = DiscoveryCandidate(symbol="TCS", mention_count=3)
    other = DiscoveryCandidate(symbol="TCS", mention_count=5)
    _merge(base, other)
    assert base.mention_count == 8


# ── DiscoveryScanner (unit — mocked network) ──────────────────────────────────

@pytest.mark.asyncio
async def test_scanner_returns_ranked_list() -> None:
    """Scanner should return candidates sorted by combined_score descending."""
    scanner = DiscoveryScanner(min_combined_score=0.0)

    # Mock both sub-tasks so no network needed
    breakout_candidate = DiscoveryCandidate(
        symbol="DIXON",
        is_breakout=True,
        technical_score=3,
        triggers=["breakout:strong"],
    )
    social_candidate = DiscoveryCandidate(
        symbol="INFY",
        mention_count=5,
        social_score=0.5,
        social_sentiment=0.3,
        triggers=["high_mentions:5"],
    )

    async def _fake_social(_loop):
        return [social_candidate]

    async def _fake_breakout(_loop, _syms):
        return [breakout_candidate]

    scanner._social_discovery = _fake_social
    scanner._breakout_discovery = _fake_breakout

    candidates = await scanner.run()
    assert len(candidates) == 2
    # DIXON (breakout) should rank higher than INFY (social only)
    assert candidates[0].symbol == "DIXON"
    assert candidates[1].symbol == "INFY"
    # Sorted descending
    assert candidates[0].combined_score >= candidates[1].combined_score


@pytest.mark.asyncio
async def test_scanner_min_score_filter() -> None:
    """Candidates below min_combined_score should be excluded."""
    scanner = DiscoveryScanner(min_combined_score=0.5)

    low_candidate = DiscoveryCandidate(
        symbol="WIPRO",
        is_breakout=False,
        technical_score=0,
        social_score=0.0,
    )

    async def _fake_social(_loop):
        return [low_candidate]

    async def _fake_breakout(_loop, _syms):
        return []

    scanner._social_discovery = _fake_social
    scanner._breakout_discovery = _fake_breakout

    candidates = await scanner.run()
    assert len(candidates) == 0   # WIPRO scores 0.0 → filtered out


@pytest.mark.asyncio
async def test_scanner_merges_same_symbol() -> None:
    """Same symbol from both paths should be merged, not duplicated."""
    scanner = DiscoveryScanner(min_combined_score=0.0)

    social = DiscoveryCandidate(
        symbol="RELIANCE",
        mention_count=4,
        social_score=0.4,
        social_sentiment=0.2,
        triggers=["high_mentions:4"],
    )
    breakout = DiscoveryCandidate(
        symbol="RELIANCE",
        is_breakout=True,
        technical_score=3,
        triggers=["breakout:moderate"],
    )

    async def _fake_social(_loop):
        return [social]

    async def _fake_breakout(_loop, _syms):
        return [breakout]

    scanner._social_discovery = _fake_social
    scanner._breakout_discovery = _fake_breakout

    candidates = await scanner.run()
    assert len(candidates) == 1   # merged, not duplicated
    c = candidates[0]
    assert c.symbol == "RELIANCE"
    assert c.is_breakout is True
    assert c.mention_count == 4
    assert "high_mentions:4" in c.triggers
    assert "breakout:moderate" in c.triggers


@pytest.mark.asyncio
async def test_scanner_handles_subtask_exception() -> None:
    """If one path raises, the other path's results should still be returned."""
    scanner = DiscoveryScanner(min_combined_score=0.0)

    async def _broken(_loop, *args):
        raise RuntimeError("network down")

    async def _ok(_loop):
        return [DiscoveryCandidate(symbol="TCS", is_breakout=True, technical_score=3)]

    scanner._social_discovery = _ok
    scanner._breakout_discovery = _broken

    candidates = await scanner.run()
    # Social path succeeded — should still get TCS
    assert any(c.symbol == "TCS" for c in candidates)


# ── Live tests ────────────────────────────────────────────────────────────────

@pytest.mark.live
@pytest.mark.asyncio
async def test_scanner_live_small_subset() -> None:
    """
    Real scan on 5 Nifty50 symbols — verifies end-to-end without Reddit creds.
    Breakout path only (social is skipped when no Reddit creds).
    """
    scanner = DiscoveryScanner(min_combined_score=0.0)
    candidates = await scanner.run(
        run_social=False,
        run_breakout=True,
        breakout_symbols=["RELIANCE", "TCS", "INFY", "HDFCBANK", "DIXON"],
    )

    # Basic sanity: candidates is a list, each has required fields
    assert isinstance(candidates, list)
    for c in candidates:
        assert c.symbol in {"RELIANCE", "TCS", "INFY", "HDFCBANK", "DIXON"}
        assert 0.0 <= c.combined_score <= 1.0
        assert isinstance(c.triggers, list)
        assert isinstance(c.is_breakout, bool)

    print(f"\nLive scan results ({len(candidates)} candidates):")
    for c in candidates:
        print(f"  {c.symbol}: score={c.combined_score:.3f}  breakout={c.is_breakout}  triggers={c.triggers}")
