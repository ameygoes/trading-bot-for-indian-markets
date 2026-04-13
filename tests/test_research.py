"""
tests/test_research.py — Unit and integration tests for the research engine.

Fast tests: model validation, sentiment math, rule-based fallback (no network).
Live tests: full pipeline on DIXON.NS (marked @pytest.mark.live).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trading_bot.research.report import (
    Fundamentals,
    NewsItem,
    Recommendation,
    ResearchReport,
    TechnicalSummary,
    TradingStyle,
)
from trading_bot.research.sentiment import aggregate_sentiment, score_text
from trading_bot.research.engine import _rule_based_recommend, _normalise_ai_output


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_tech(
    price: float = 14_250.0,
    score: int = 3,
    bias: str = "BULLISH",
    is_breakout: bool = True,
    atr: float = 220.0,
) -> TechnicalSummary:
    return TechnicalSummary(
        price=price,
        change_pct=1.8,
        trading_bias=bias,
        score=score,
        trend="UPTREND",
        rsi=62.5,
        macd_cross="BULLISH_CROSS",
        supertrend="BULLISH",
        volume_label="HIGH_VOLUME",
        is_breakout=is_breakout,
        breakout_strength="4/5 signals bullish",
        nearest_support=13_600.0,
        nearest_resistance=15_500.0,
        atr=atr,
        atr_pct=1.54,
    )


def _make_fund(
    market_cap_cr: float = 25_000.0,
    pe: float = 45.0,
    roe: float = 22.0,
    d2e: float = 0.3,
) -> Fundamentals:
    return Fundamentals(
        market_cap_cr=market_cap_cr,
        pe_ratio=pe,
        pb_ratio=8.0,
        eps_ttm=316.0,
        roe_pct=roe,
        revenue_growth_yoy_pct=28.5,
        debt_to_equity=d2e,
        sector="Technology",
        industry="Electronic Manufacturing Services",
    )


# ── Sentiment scoring ─────────────────────────────────────────────────────────

def test_score_text_bullish() -> None:
    score, label = score_text("Reliance hits record high on strong earnings beat")
    assert label == "bullish"
    assert score > 0.05


def test_score_text_bearish() -> None:
    score, label = score_text("Stock crashes after disappointing revenue miss")
    assert label == "bearish"
    assert score < -0.05


def test_score_text_empty() -> None:
    score, label = score_text("")
    assert score == 0.0
    assert label == "neutral"


def test_aggregate_sentiment_mean() -> None:
    assert aggregate_sentiment([0.5, -0.5, 0.0]) == pytest.approx(0.0, abs=1e-4)


def test_aggregate_sentiment_empty() -> None:
    assert aggregate_sentiment([]) == 0.0


# ── Rule-based recommendation ─────────────────────────────────────────────────

def test_rule_based_buy_signal() -> None:
    tech = _make_tech(score=3, is_breakout=True)
    fund = _make_fund()
    result, raw = _rule_based_recommend(tech, fund)
    assert result["recommendation"] == Recommendation.BUY
    assert result["confidence_score"] > 0.5


def test_rule_based_watch_signal() -> None:
    tech = _make_tech(score=1, is_breakout=False)
    fund = _make_fund()
    result, _ = _rule_based_recommend(tech, fund)
    assert result["recommendation"] in (Recommendation.WATCH, Recommendation.BUY)


def test_rule_based_skip_signal() -> None:
    tech = _make_tech(score=-3, bias="STRONG_BEARISH", is_breakout=False)
    fund = _make_fund()
    result, _ = _rule_based_recommend(tech, fund)
    assert result["recommendation"] == Recommendation.SKIP


def test_rule_based_entry_and_sl() -> None:
    tech = _make_tech(score=3, price=14_250.0, atr=220.0)
    fund = _make_fund()
    result, _ = _rule_based_recommend(tech, fund)
    if result["recommendation"] == Recommendation.BUY:
        assert result["entry_price"] == pytest.approx(14_250.0)
        assert result["stop_loss"] < result["entry_price"]
        assert result["target_1"] > result["entry_price"]


def test_rule_based_trading_style_large_cap() -> None:
    tech = _make_tech()
    fund = _make_fund(market_cap_cr=25_000)
    result, _ = _rule_based_recommend(tech, fund)
    assert result["trading_style"] == TradingStyle.SWING


def test_rule_based_trading_style_small_cap() -> None:
    tech = _make_tech()
    fund = _make_fund(market_cap_cr=2_000)
    result, _ = _rule_based_recommend(tech, fund)
    assert result["trading_style"] == TradingStyle.DAY_TRADE


# ── _normalise_ai_output ──────────────────────────────────────────────────────

def test_normalise_valid_output() -> None:
    raw = {
        "recommendation": "BUY",
        "trading_style": "SWING",
        "confidence_score": 0.78,
        "entry_price": 14250.0,
        "stop_loss": 13600.0,
        "target_1": 15500.0,
        "target_2": 16800.0,
        "reasoning": "Strong breakout with volume.",
    }
    result = _normalise_ai_output(raw)
    assert result["recommendation"] == Recommendation.BUY
    assert result["trading_style"] == TradingStyle.SWING
    assert result["confidence_score"] == pytest.approx(0.78)


def test_normalise_invalid_recommendation_defaults_to_skip() -> None:
    result = _normalise_ai_output({"recommendation": "MAYBE"})
    assert result["recommendation"] == Recommendation.SKIP


# ── ResearchReport Pydantic model ─────────────────────────────────────────────

def test_report_serialisation_roundtrip(tmp_path: Path) -> None:
    tech = _make_tech()
    fund = _make_fund()
    report = ResearchReport(
        symbol="DIXON",
        exchange="NSE",
        technical=tech,
        fundamentals=fund,
        recommendation=Recommendation.BUY,
        trading_style=TradingStyle.SWING,
        confidence_score=0.78,
        entry_price=14_250.0,
        stop_loss=13_600.0,
        target_1=15_500.0,
        target_2=16_800.0,
        reasoning="Strong EMS sector tailwinds with volume breakout.",
    )

    json_str = report.model_dump_json(indent=2)
    loaded = ResearchReport.model_validate_json(json_str)
    assert loaded.symbol == "DIXON"
    assert loaded.recommendation == Recommendation.BUY
    assert loaded.confidence_score == pytest.approx(0.78)


def test_report_save_and_load(tmp_path: Path, monkeypatch) -> None:
    from trading_bot import config as cfg
    monkeypatch.setattr(cfg, "RESEARCH_REPORTS_DIR", tmp_path)

    tech = _make_tech()
    fund = _make_fund()
    report = ResearchReport(
        symbol="TESTSTOCK",
        exchange="NSE",
        technical=tech,
        fundamentals=fund,
        recommendation=Recommendation.WATCH,
    )
    path = report.save()
    assert path.exists()

    loaded = ResearchReport.load(path)
    assert loaded.symbol == "TESTSTOCK"
    assert loaded.recommendation == Recommendation.WATCH


# ── Full pipeline — live ──────────────────────────────────────────────────────

@pytest.mark.live
@pytest.mark.asyncio
async def test_research_engine_dixon_live(tmp_path: Path, monkeypatch) -> None:
    """Full end-to-end pipeline on DIXON.NS (live yfinance + rule-based fallback)."""
    from trading_bot import config as cfg
    # Redirect report save to tmp_path so we don't write to real data/ dir
    monkeypatch.setattr(cfg, "RESEARCH_REPORTS_DIR", tmp_path)
    # Force rule-based by clearing API key
    monkeypatch.setattr(cfg, "ANTHROPIC_API_KEY", "")

    from trading_bot.research.engine import ResearchEngine
    engine = ResearchEngine()
    report = await engine.research("DIXON", "NSE")

    assert report.symbol == "DIXON"
    assert report.exchange == "NSE"
    assert report.technical.price > 0
    assert report.recommendation in list(Recommendation)
    assert 0.0 <= report.confidence_score <= 1.0
    assert report.reasoning != ""

    # Should have saved the file
    files = list(tmp_path.glob("DIXON_*.json"))
    assert len(files) == 1

    # Load back and validate
    loaded = ResearchReport.load(files[0])
    assert loaded.symbol == "DIXON"

    print(f"\nDIXON Research Report:\n{report.model_dump_json(indent=2)}")
