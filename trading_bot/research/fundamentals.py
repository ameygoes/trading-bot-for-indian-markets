"""
research/fundamentals.py — Fetch company fundamentals from yfinance.

Returns a Fundamentals object. All fields are optional; missing data is
silently skipped rather than raising — partial data is better than none.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import yfinance as yf
from loguru import logger

from .report import Fundamentals


def _safe(val: Any) -> Optional[float]:
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def fetch_fundamentals(symbol: str, exchange: str = "NSE") -> Fundamentals:
    """
    Pull key fundamental metrics for *symbol* from yfinance.

    Args:
        symbol:   Ticker without suffix, e.g. "DIXON"
        exchange: "NSE" or "BSE"

    Returns:
        Fundamentals — all fields are Optional; never raises.
    """
    suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
    yf_sym = f"{symbol.upper()}{suffix}" if "." not in symbol else symbol.upper()

    try:
        ticker = yf.Ticker(yf_sym)
        info: dict = ticker.info or {}
    except Exception as exc:
        logger.warning("fundamentals fetch failed for {}: {}", yf_sym, exc)
        return Fundamentals()

    # Market cap → Crores (1 Cr = 10M INR, 1 USD ≈ 84 INR)
    market_cap_raw = _safe(info.get("marketCap"))
    currency = info.get("currency", "INR")
    if market_cap_raw is not None:
        if currency == "USD":
            market_cap_raw = market_cap_raw * 84   # rough USD→INR
        market_cap_cr = round(market_cap_raw / 1e7, 2)
    else:
        market_cap_cr = None

    # Revenue growth (yfinance gives it as a fraction, e.g. 0.23 = 23%)
    rev_growth = _safe(info.get("revenueGrowth"))
    if rev_growth is not None:
        rev_growth = round(rev_growth * 100, 2)

    # ROE (also a fraction)
    roe = _safe(info.get("returnOnEquity"))
    if roe is not None:
        roe = round(roe * 100, 2)

    fund = Fundamentals(
        market_cap_cr=market_cap_cr,
        pe_ratio=_safe(info.get("trailingPE") or info.get("forwardPE")),
        pb_ratio=_safe(info.get("priceToBook")),
        eps_ttm=_safe(info.get("trailingEps")),
        roe_pct=roe,
        revenue_growth_yoy_pct=rev_growth,
        debt_to_equity=_safe(info.get("debtToEquity")),
        sector=info.get("sector", ""),
        industry=info.get("industry", ""),
        promoter_holding_pct=None,   # not available via yfinance for NSE
    )

    logger.debug(
        "Fundamentals for {}: market_cap_cr={} pe={} sector={}",
        symbol,
        fund.market_cap_cr,
        fund.pe_ratio,
        fund.sector,
    )
    return fund
