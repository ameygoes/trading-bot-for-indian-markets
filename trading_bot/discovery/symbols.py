"""
discovery/symbols.py — Curated NSE/BSE watchlist and symbol extraction utilities.

Provides:
  - NIFTY50_SYMBOLS    : the 50 Nifty index constituents (NSE)
  - MIDCAP_WATCHLIST   : hand-picked mid/small-cap NSE names worth watching
  - ALL_WATCHLIST      : union of both lists
  - extract_symbols_from_text(text) → set[str]
      Regex-based parser that finds stock tickers and company aliases mentioned
      in free text (Reddit posts, news headlines, etc.)

Design notes:
  - No network calls here — purely in-memory lookups.
  - extract_symbols_from_text returns only symbols that appear in ALL_WATCHLIST
    so callers always get valid, tradeable tickers.
  - Company → symbol alias map lets the scanner resolve "Dixon" → "DIXON".
"""

from __future__ import annotations

import re
from typing import FrozenSet

# ── Nifty 50 constituents (as of April 2026) ──────────────────────────────────
NIFTY50_SYMBOLS: tuple[str, ...] = (
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
    "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
    "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
)

# ── Mid / small-cap watchlist ─────────────────────────────────────────────────
MIDCAP_WATCHLIST: tuple[str, ...] = (
    # EMS / Electronics
    "DIXON", "AMBER", "KAYNES", "SYRMA", "PGEL",
    # Defence
    "HAL", "BDL", "DATAPATTNS", "MAZDOCK", "COCHINSHIP",
    # Pharma / Specialty
    "LAURUS", "GRANULES", "LAURUSLABS", "AAVAS", "ALKEM",
    # Financials
    "CHOLAFIN", "MUTHOOTFIN", "MANAPPURAM", "IIFL", "PNBHOUSING",
    # IT / SaaS
    "LTIM", "PERSISTENT", "COFORGE", "MPHASIS", "SONATSOFTW",
    # Infra / Realty
    "OBEROIRLTY", "GODREJPROP", "DLF", "PRESTIGE", "BRIGADE",
    # Consumer
    "DMART", "TATACOMM", "ZOMATO", "NYKAA", "MARICO",
    # Chemicals
    "PIIND", "DEEPAKNTR", "AARTI", "CLEAN", "FINEORG",
    # Capital Goods
    "THERMAX", "CUMMINSIND", "GRINDWELL", "AIA", "ELGIEQUIP",
    # Auto Ancillary
    "MOTHERSON", "BOSCHLTD", "SUNDRMFAST", "BALKRISIND", "APOLLOTYRE",
)

# ── Combined (deduplicated, order: Nifty50 first) ─────────────────────────────
ALL_WATCHLIST: FrozenSet[str] = frozenset(NIFTY50_SYMBOLS + MIDCAP_WATCHLIST)


# ── Company → symbol alias table ──────────────────────────────────────────────
# Lower-cased aliases so matching is case-insensitive.
_ALIAS_MAP: dict[str, str] = {
    # Nifty 50
    "reliance": "RELIANCE", "ril": "RELIANCE",
    "tcs": "TCS", "tata consultancy": "TCS",
    "infosys": "INFY", "infy": "INFY",
    "wipro": "WIPRO",
    "hcltech": "HCLTECH", "hcl": "HCLTECH",
    "techm": "TECHM", "tech mahindra": "TECHM",
    "ltimindtree": "LTIM", "ltim": "LTIM",
    "hdfcbank": "HDFCBANK", "hdfc bank": "HDFCBANK",
    "icicibank": "ICICIBANK", "icici": "ICICIBANK",
    "axisbank": "AXISBANK", "axis bank": "AXISBANK",
    "kotakbank": "KOTAKBANK", "kotak": "KOTAKBANK",
    "sbi": "SBIN", "state bank": "SBIN",
    "bajfinance": "BAJFINANCE", "bajaj finance": "BAJFINANCE",
    "tatamotors": "TATAMOTORS", "tata motors": "TATAMOTORS",
    "maruti": "MARUTI", "maruti suzuki": "MARUTI",
    "sunpharma": "SUNPHARMA", "sun pharma": "SUNPHARMA",
    "drreddy": "DRREDDY", "dr reddy": "DRREDDY",
    "cipla": "CIPLA",
    "hindustan unilever": "HINDUNILVR", "hul": "HINDUNILVR",
    "itc": "ITC",
    "nestle": "NESTLEIND",
    "asian paint": "ASIANPAINT", "asian paints": "ASIANPAINT",
    "titan": "TITAN",
    "trent": "TRENT",
    "britannia": "BRITANNIA",
    "adani ports": "ADANIPORTS", "adaniports": "ADANIPORTS",
    "adani ent": "ADANIENT", "adani enterprises": "ADANIENT",
    "hindalco": "HINDALCO",
    "jswsteel": "JSWSTEEL", "jsw steel": "JSWSTEEL",
    "tatasteel": "TATASTEEL", "tata steel": "TATASTEEL",
    "ongc": "ONGC",
    "bpcl": "BPCL",
    "ntpc": "NTPC",
    "powergrid": "POWERGRID", "power grid": "POWERGRID",
    "coalindia": "COALINDIA", "coal india": "COALINDIA",
    "bharti airtel": "BHARTIARTL", "airtel": "BHARTIARTL",
    "lt": "LT", "larsen": "LT", "l&t": "LT",
    "grasim": "GRASIM",
    "ultratech": "ULTRACEMCO", "ultracemco": "ULTRACEMCO",
    "apollo hospital": "APOLLOHOSP", "apollo hospitals": "APOLLOHOSP",
    "eicher": "EICHERMOT",
    "hero motocorp": "HEROMOTOCO", "hero moto": "HEROMOTOCO",
    "bajaj auto": "BAJAJ-AUTO",
    "m&m": "M&M", "mahindra": "M&M",
    "shriram finance": "SHRIRAMFIN",
    "indusind": "INDUSINDBK",
    "sbilife": "SBILIFE", "sbi life": "SBILIFE",
    "hdfclife": "HDFCLIFE", "hdfc life": "HDFCLIFE",
    "bel": "BEL",
    # Mid / small cap
    "dixon": "DIXON", "dixon technologies": "DIXON",
    "amber": "AMBER", "amber enterprises": "AMBER",
    "kaynes": "KAYNES",
    "syrma": "SYRMA",
    "hal": "HAL", "hindustan aeronautics": "HAL",
    "bdl": "BDL",
    "data patterns": "DATAPATTNS",
    "mazagon": "MAZDOCK",
    "cochin shipyard": "COCHINSHIP",
    "laurus": "LAURUS",
    "granules": "GRANULES",
    "cholamandalam": "CHOLAFIN", "chola": "CHOLAFIN",
    "muthoot": "MUTHOOTFIN", "muthoot finance": "MUTHOOTFIN",
    "manappuram": "MANAPPURAM",
    "persistent": "PERSISTENT",
    "coforge": "COFORGE",
    "mphasis": "MPHASIS",
    "oberoi realty": "OBEROIRLTY",
    "godrej properties": "GODREJPROP",
    "dlf": "DLF",
    "prestige": "PRESTIGE",
    "dmart": "DMART", "avenue supermarts": "DMART",
    "zomato": "ZOMATO",
    "nykaa": "NYKAA",
    "marico": "MARICO",
    "pi industries": "PIIND",
    "deepak nitrite": "DEEPAKNTR",
    "aarti": "AARTI",
    "motherson": "MOTHERSON",
    "bosch": "BOSCHLTD",
    "apollo tyre": "APOLLOTYRE", "apollo tyres": "APOLLOTYRE",
    "balkrishna": "BALKRISIND",
}

# ── Regex patterns ─────────────────────────────────────────────────────────────
# Matches bare tickers like DIXON, TCS, BAJAJ-AUTO, M&M (2-15 uppercase chars + optional hyphen/ampersand)
_TICKER_RE = re.compile(r"\b([A-Z][A-Z0-9&-]{1,14})\b")

# Matches "$TICK" style (common on Twitter/Reddit)
_DOLLAR_TICKER_RE = re.compile(r"\$([A-Z][A-Z0-9&-]{1,14})\b")


def extract_symbols_from_text(text: str) -> set[str]:
    """
    Extract NSE watchlist symbols mentioned in *text*.

    Strategy:
      1. Check alias map (case-insensitive) for company names / abbreviations.
      2. Regex-scan for uppercase tickers that are in ALL_WATCHLIST.
      3. Regex-scan for $TICKER style mentions.

    Returns only symbols present in ALL_WATCHLIST so callers always get valid tickers.

    Args:
        text: Raw text (Reddit post title/body, news headline, etc.)

    Returns:
        Set of NSE ticker strings (upper-case, no exchange suffix).
    """
    if not text:
        return set()

    found: set[str] = set()
    lower = text.lower()

    # 1. Alias scan (longest first to prefer "tata motors" over "tata")
    for alias in sorted(_ALIAS_MAP, key=len, reverse=True):
        if alias in lower:
            sym = _ALIAS_MAP[alias]
            if sym in ALL_WATCHLIST:
                found.add(sym)

    # 2. Bare uppercase ticker scan
    for m in _TICKER_RE.finditer(text):
        candidate = m.group(1)
        if candidate in ALL_WATCHLIST:
            found.add(candidate)

    # 3. $TICKER scan
    for m in _DOLLAR_TICKER_RE.finditer(text):
        candidate = m.group(1)
        if candidate in ALL_WATCHLIST:
            found.add(candidate)

    return found
