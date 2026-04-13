"""
mcp_server.py — MCP Technical Analysis Server for Indian Markets (NSE/BSE).

Exposes these tools over the Model Context Protocol so Claude (or any MCP
client) can call them directly:

  get_quote             — real-time price snapshot for any NSE/BSE symbol
  get_ohlcv             — historical OHLCV bars (daily / intraday)
  get_indicators        — EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, Supertrend
  get_support_resistance— pivot-based support & resistance levels
  scan_breakout         — volume + price breakout detector
  get_market_status     — is NSE currently open?
  analyze_stock         — full composite analysis (all tools combined)

Run standalone:
    python -m trading_bot.mcp_server

Or via MCP config:
    {
      "mcpServers": {
        "indian-ta": {
          "command": "python",
          "args": ["-m", "trading_bot.mcp_server"]
        }
      }
    }
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

try:
    import pandas_ta as ta  # type: ignore
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False

from mcp.server.fastmcp import FastMCP

from trading_bot import config

# ── Server instance ────────────────────────────────────────────────────────────
mcp = FastMCP(
    "Indian Markets Technical Analysis",
    instructions="NSE/BSE OHLCV data, indicators, S/R levels and breakout scanning.",
)

IST = config.IST
_MARKET_OPEN = (config.MARKET_OPEN_HOUR, config.MARKET_OPEN_MINUTE)
_MARKET_CLOSE = (config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_yf_symbol(symbol: str, exchange: str = "NSE") -> str:
    """Append Yahoo Finance exchange suffix if not already present."""
    symbol = symbol.upper().strip()
    if "." in symbol:
        return symbol
    suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
    return f"{symbol}{suffix}"


def _fetch_ohlcv(
    yf_symbol: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV from yfinance. Returns empty DataFrame on failure."""
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.DatetimeIndex(df.index)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as exc:
        return pd.DataFrame()


def _safe_float(val: Any) -> Optional[float]:
    """Convert to float, returning None for NaN / inf."""
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _ema_pandas(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _rsi_pandas(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _macd_pandas(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bbands_pandas(
    series: pd.Series, length: int = 20, std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(length).mean()
    stddev = series.rolling(length).std(ddof=0)
    return mid + std * stddev, mid, mid - std * stddev


def _atr_pandas(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP — rolling from start of available data."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tpv = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tpv / cum_vol


def _supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.Series:
    """Returns Supertrend line (NaN where insufficient data)."""
    atr = _atr_pandas(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i - 1]

        ub = upper_band.iloc[i]
        lb = lower_band.iloc[i]
        prev_ub = upper_band.iloc[i - 1]
        prev_lb = lower_band.iloc[i - 1]
        prev_st = supertrend.iloc[i - 1] if i > 1 else ub

        # Adjust bands
        if lb < prev_lb or prev_close < prev_lb:
            lb = lb
        else:
            lb = max(lb, prev_lb)

        if ub > prev_ub or prev_close > prev_ub:
            ub = ub
        else:
            ub = min(ub, prev_ub)

        upper_band.iloc[i] = ub
        lower_band.iloc[i] = lb

        if i == 1:
            direction.iloc[i] = 1
        elif prev_st == prev_ub:
            direction.iloc[i] = -1 if close > ub else 1
        else:
            direction.iloc[i] = 1 if close < lb else -1

        supertrend.iloc[i] = lb if direction.iloc[i] == -1 else ub

    return supertrend


# ══════════════════════════════════════════════════════════════════════════════
# MCP Tools
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_market_status() -> dict:
    """
    Check if NSE is currently open for trading.

    Returns current IST time, market status (OPEN/CLOSED/PRE-OPEN),
    and next open/close time.
    """
    now = datetime.now(IST)
    weekday = now.weekday()  # 0=Mon, 6=Sun

    is_weekday = weekday < 5
    open_h, open_m = _MARKET_OPEN
    close_h, close_m = _MARKET_CLOSE

    market_open_dt = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    market_close_dt = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    premarket_dt = now.replace(hour=9, minute=0, second=0, microsecond=0)

    if not is_weekday:
        status = "CLOSED"
        reason = "Weekend"
    elif now < premarket_dt:
        status = "CLOSED"
        reason = "Before pre-open"
    elif now < market_open_dt:
        status = "PRE-OPEN"
        reason = "Pre-open session (9:00–9:15 IST)"
    elif now <= market_close_dt:
        status = "OPEN"
        reason = "Regular trading session"
    else:
        status = "CLOSED"
        reason = "After market close"

    minutes_to_close = (
        int((market_close_dt - now).total_seconds() / 60) if status == "OPEN" else None
    )

    return {
        "status": status,
        "reason": reason,
        "current_time_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "market_open": f"{open_h:02d}:{open_m:02d} IST",
        "market_close": f"{close_h:02d}:{close_m:02d} IST",
        "minutes_to_close": minutes_to_close,
        "is_trading_day": is_weekday,
    }


@mcp.tool()
def get_quote(symbol: str, exchange: str = "NSE") -> dict:
    """
    Real-time price quote for an NSE or BSE stock.

    Args:
        symbol:   Stock symbol without suffix, e.g. "DIXON", "RELIANCE", "INFY"
        exchange: "NSE" (default) or "BSE"

    Returns price, change, volume, 52-week high/low and circuit limits.
    """
    yf_sym = _to_yf_symbol(symbol, exchange)
    try:
        ticker = yf.Ticker(yf_sym)
        info = ticker.fast_info
        hist = ticker.history(period="2d", interval="1d", auto_adjust=True)

        if hist.empty:
            return {"error": f"No data found for {yf_sym}"}

        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]

        price = float(latest["Close"])
        prev_close = float(prev["Close"])
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0

        return {
            "symbol": symbol.upper(),
            "exchange": exchange.upper(),
            "yf_symbol": yf_sym,
            "price": round(price, 2),
            "prev_close": round(prev_close, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "open": round(float(latest["Open"]), 2),
            "high": round(float(latest["High"]), 2),
            "low": round(float(latest["Low"]), 2),
            "volume": int(latest["Volume"]),
            "52w_high": _safe_float(getattr(info, "year_high", None)),
            "52w_low": _safe_float(getattr(info, "year_low", None)),
            "market_cap_cr": _safe_float(
                getattr(info, "market_cap", None) and info.market_cap / 1e7
            ),
            "timestamp_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        }
    except Exception as exc:
        return {"error": str(exc), "symbol": symbol, "exchange": exchange}


@mcp.tool()
def get_ohlcv(
    symbol: str,
    exchange: str = "NSE",
    period: str = "6mo",
    interval: str = "1d",
    limit: int = 60,
) -> dict:
    """
    Historical OHLCV bars for an NSE/BSE stock.

    Args:
        symbol:   Stock symbol, e.g. "DIXON"
        exchange: "NSE" or "BSE"
        period:   yfinance period string: "1mo", "3mo", "6mo", "1y", "2y"
        interval: Bar size: "1d" (daily), "1wk" (weekly), "60m" (hourly)
        limit:    Max number of bars to return (most recent)

    Returns list of OHLCV bars sorted oldest-first.
    """
    yf_sym = _to_yf_symbol(symbol, exchange)
    df = _fetch_ohlcv(yf_sym, period=period, interval=interval)

    if df.empty:
        return {"error": f"No OHLCV data for {yf_sym}", "symbol": symbol}

    df = df.tail(limit)
    bars = [
        {
            "date": str(idx.date() if hasattr(idx, "date") else idx),
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
            "volume": int(row["volume"]),
        }
        for idx, row in df.iterrows()
    ]

    return {
        "symbol": symbol.upper(),
        "exchange": exchange.upper(),
        "interval": interval,
        "bars_returned": len(bars),
        "from_date": bars[0]["date"] if bars else None,
        "to_date": bars[-1]["date"] if bars else None,
        "bars": bars,
    }


@mcp.tool()
def get_indicators(
    symbol: str,
    exchange: str = "NSE",
    period: str = "1y",
) -> dict:
    """
    Full suite of technical indicators for an NSE/BSE stock (daily timeframe).

    Indicators returned (latest bar values):
      - EMA 9, 21, 50, 200
      - RSI 14
      - MACD (12,26,9): line, signal, histogram
      - Bollinger Bands (20,2): upper, middle, lower, %B, bandwidth
      - ATR 14
      - VWAP (rolling session)
      - Supertrend (10,3)
      - Volume ratio (today vs 20-day avg)

    Also returns signal summary: trend direction, momentum, volatility.

    Args:
        symbol:   Stock symbol, e.g. "DIXON"
        exchange: "NSE" or "BSE"
        period:   Data lookback for indicator calculation, e.g. "1y"
    """
    yf_sym = _to_yf_symbol(symbol, exchange)
    df = _fetch_ohlcv(yf_sym, period=period, interval="1d")

    if df.empty or len(df) < 30:
        return {"error": f"Insufficient data for {yf_sym} (got {len(df)} bars)"}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── EMAs ──────────────────────────────────────────────────────────────────
    ema9 = _ema_pandas(close, 9)
    ema21 = _ema_pandas(close, 21)
    ema50 = _ema_pandas(close, 50) if len(df) >= 50 else pd.Series(dtype=float)
    ema200 = _ema_pandas(close, 200) if len(df) >= 200 else pd.Series(dtype=float)

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi = _rsi_pandas(close, 14)

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_line, macd_signal, macd_hist = _macd_pandas(close)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_upper, bb_mid, bb_lower = _bbands_pandas(close, 20, 2.0)
    bb_pct_b = (close - bb_lower) / (bb_upper - bb_lower)
    bb_bandwidth = (bb_upper - bb_lower) / bb_mid * 100

    # ── ATR ───────────────────────────────────────────────────────────────────
    atr = _atr_pandas(df, 14)

    # ── VWAP ─────────────────────────────────────────────────────────────────
    vwap_series = _vwap(df)

    # ── Supertrend ────────────────────────────────────────────────────────────
    supertrend = _supertrend(df, 10, 3.0)

    # ── Volume ratio ──────────────────────────────────────────────────────────
    avg_vol_20 = volume.rolling(20).mean()
    vol_ratio = volume / avg_vol_20

    # ── Latest values ─────────────────────────────────────────────────────────
    price = _safe_float(close.iloc[-1])
    st_val = _safe_float(supertrend.iloc[-1])
    st_trend = "BULLISH" if (price and st_val and price > st_val) else "BEARISH"

    rsi_val = _safe_float(rsi.iloc[-1])
    macd_h_val = _safe_float(macd_hist.iloc[-1])
    macd_h_prev = _safe_float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else None

    # ── Signal summary ────────────────────────────────────────────────────────
    trend_score = 0
    if price and _safe_float(ema9.iloc[-1]) and price > _safe_float(ema9.iloc[-1]):
        trend_score += 1
    if price and _safe_float(ema21.iloc[-1]) and price > _safe_float(ema21.iloc[-1]):
        trend_score += 1
    if not ema50.empty and price and _safe_float(ema50.iloc[-1]) and price > _safe_float(ema50.iloc[-1]):
        trend_score += 1
    if not ema200.empty and price and _safe_float(ema200.iloc[-1]) and price > _safe_float(ema200.iloc[-1]):
        trend_score += 1

    trend_label = (
        "STRONG_UPTREND" if trend_score >= 3
        else "UPTREND" if trend_score == 2
        else "DOWNTREND" if trend_score <= 1
        else "NEUTRAL"
    )

    rsi_label = (
        "OVERBOUGHT" if rsi_val and rsi_val > 70
        else "OVERSOLD" if rsi_val and rsi_val < 30
        else "NEUTRAL"
    )

    macd_cross = None
    if macd_h_val is not None and macd_h_prev is not None:
        if macd_h_prev < 0 < macd_h_val:
            macd_cross = "BULLISH_CROSS"
        elif macd_h_prev > 0 > macd_h_val:
            macd_cross = "BEARISH_CROSS"

    return {
        "symbol": symbol.upper(),
        "exchange": exchange.upper(),
        "price": price,
        "as_of": str(df.index[-1].date()),
        "ema": {
            "ema9": _safe_float(ema9.iloc[-1]),
            "ema21": _safe_float(ema21.iloc[-1]),
            "ema50": _safe_float(ema50.iloc[-1]) if not ema50.empty else None,
            "ema200": _safe_float(ema200.iloc[-1]) if not ema200.empty else None,
        },
        "rsi": {
            "value": rsi_val,
            "label": rsi_label,
        },
        "macd": {
            "line": _safe_float(macd_line.iloc[-1]),
            "signal": _safe_float(macd_signal.iloc[-1]),
            "histogram": macd_h_val,
            "cross": macd_cross,
        },
        "bollinger_bands": {
            "upper": _safe_float(bb_upper.iloc[-1]),
            "middle": _safe_float(bb_mid.iloc[-1]),
            "lower": _safe_float(bb_lower.iloc[-1]),
            "pct_b": _safe_float(bb_pct_b.iloc[-1]),
            "bandwidth": _safe_float(bb_bandwidth.iloc[-1]),
        },
        "atr": {
            "value": _safe_float(atr.iloc[-1]),
            "atr_pct": _safe_float(atr.iloc[-1] / price * 100) if price else None,
        },
        "vwap": _safe_float(vwap_series.iloc[-1]),
        "supertrend": {
            "value": st_val,
            "trend": st_trend,
        },
        "volume": {
            "today": int(volume.iloc[-1]),
            "avg_20d": int(avg_vol_20.iloc[-1]) if not math.isnan(avg_vol_20.iloc[-1]) else None,
            "ratio": _safe_float(vol_ratio.iloc[-1]),
            "label": (
                "HIGH_VOLUME" if vol_ratio.iloc[-1] > 1.5
                else "LOW_VOLUME" if vol_ratio.iloc[-1] < 0.5
                else "NORMAL"
            ),
        },
        "signals": {
            "trend": trend_label,
            "trend_score": f"{trend_score}/4 EMAs above price",
            "momentum": rsi_label,
            "macd_cross": macd_cross,
            "supertrend": st_trend,
        },
    }


@mcp.tool()
def get_support_resistance(
    symbol: str,
    exchange: str = "NSE",
    period: str = "1y",
    levels: int = 5,
) -> dict:
    """
    Pivot-based support and resistance levels for an NSE/BSE stock.

    Finds the *levels* most significant S/R zones using:
      - Classic pivot points (daily and weekly)
      - Recent swing highs / swing lows (5-bar fractals)
      - 52-week high/low

    Args:
        symbol:   Stock symbol, e.g. "DIXON"
        exchange: "NSE" or "BSE"
        period:   Data lookback: "6mo", "1y"
        levels:   Number of S and R levels to return each
    """
    yf_sym = _to_yf_symbol(symbol, exchange)
    df = _fetch_ohlcv(yf_sym, period=period, interval="1d")

    if df.empty or len(df) < 20:
        return {"error": f"Insufficient data for {yf_sym}"}

    price = float(df["close"].iloc[-1])

    # ── Classic daily pivot ────────────────────────────────────────────────────
    prev = df.iloc[-2]
    H, L, C = float(prev["high"]), float(prev["low"]), float(prev["close"])
    PP = (H + L + C) / 3
    R1 = 2 * PP - L
    R2 = PP + (H - L)
    R3 = H + 2 * (PP - L)
    S1 = 2 * PP - H
    S2 = PP - (H - L)
    S3 = L - 2 * (H - PP)

    pivot_levels = {
        "PP": round(PP, 2),
        "R1": round(R1, 2), "R2": round(R2, 2), "R3": round(R3, 2),
        "S1": round(S1, 2), "S2": round(S2, 2), "S3": round(S3, 2),
    }

    # ── Swing highs / lows (5-bar fractal) ────────────────────────────────────
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    highs = df["high"].values
    lows = df["low"].values

    for i in range(2, len(df) - 2):
        if highs[i] == max(highs[i - 2 : i + 3]):
            swing_highs.append(round(float(highs[i]), 2))
        if lows[i] == min(lows[i - 2 : i + 3]):
            swing_lows.append(round(float(lows[i]), 2))

    # Deduplicate within 0.5% of each other, keep most recent
    def _dedup(vals: list[float], pct: float = 0.005) -> list[float]:
        out: list[float] = []
        for v in reversed(vals):
            if not any(abs(v - x) / x < pct for x in out):
                out.append(v)
        return sorted(out, reverse=True)

    swing_highs = _dedup(swing_highs)
    swing_lows = _dedup(swing_lows)

    # ── 52-week high / low ────────────────────────────────────────────────────
    yr_high = round(float(df["high"].max()), 2)
    yr_low = round(float(df["low"].min()), 2)

    # ── Split into resistance (above price) and support (below price) ─────────
    all_resistance = sorted(
        set(
            [v for v in [R1, R2, R3] if v > price]
            + [v for v in swing_highs if v > price]
            + ([yr_high] if yr_high > price else [])
        )
    )[:levels]

    all_support = sorted(
        set(
            [v for v in [S1, S2, S3] if v < price]
            + [v for v in swing_lows if v < price]
            + ([yr_low] if yr_low < price else [])
        ),
        reverse=True,
    )[:levels]

    def _pct_from_price(lvl: float) -> float:
        return round((lvl - price) / price * 100, 2)

    return {
        "symbol": symbol.upper(),
        "exchange": exchange.upper(),
        "current_price": round(price, 2),
        "pivot_points": pivot_levels,
        "resistance_levels": [
            {"price": r, "pct_away": _pct_from_price(r)} for r in all_resistance
        ],
        "support_levels": [
            {"price": s, "pct_away": _pct_from_price(s)} for s in all_support
        ],
        "52w_high": yr_high,
        "52w_low": yr_low,
        "near_52w_high": (yr_high - price) / yr_high < 0.05,
        "near_52w_low": (price - yr_low) / yr_low < 0.05,
    }


@mcp.tool()
def scan_breakout(
    symbol: str,
    exchange: str = "NSE",
    volume_threshold: float = 1.5,
    price_threshold_pct: float = 1.0,
) -> dict:
    """
    Detect if a stock is breaking out (price + volume confirmation).

    A breakout is flagged when:
      - Today's close is above the 20-day high (price breakout), AND
      - Today's volume is ≥ volume_threshold × 20-day average (volume confirmation)

    Also checks for:
      - EMA 9/21 golden cross (recent)
      - RSI entering bullish momentum zone (50–70)
      - Price crossing above Supertrend line

    Args:
        symbol:              Stock symbol, e.g. "DIXON"
        exchange:            "NSE" or "BSE"
        volume_threshold:    Min volume ratio to confirm breakout (default 1.5×)
        price_threshold_pct: Min % above 20-day high to qualify (default 1.0%)
    """
    yf_sym = _to_yf_symbol(symbol, exchange)
    df = _fetch_ohlcv(yf_sym, period="6mo", interval="1d")

    if df.empty or len(df) < 25:
        return {"error": f"Insufficient data for {yf_sym}"}

    close = df["close"]
    volume = df["volume"]
    price = float(close.iloc[-1])

    # 20-day lookback (exclude today)
    lookback = df.iloc[-21:-1]
    high_20d = float(lookback["high"].max())
    avg_vol_20 = float(volume.iloc[-21:-1].mean())
    today_vol = float(volume.iloc[-1])

    price_breakout = price > high_20d * (1 + price_threshold_pct / 100)
    volume_confirm = today_vol >= avg_vol_20 * volume_threshold
    vol_ratio = today_vol / avg_vol_20 if avg_vol_20 else 0.0

    # EMA cross
    ema9 = _ema_pandas(close, 9)
    ema21 = _ema_pandas(close, 21)
    golden_cross = (
        ema9.iloc[-1] > ema21.iloc[-1]
        and ema9.iloc[-2] <= ema21.iloc[-2]
    )
    ema_aligned = ema9.iloc[-1] > ema21.iloc[-1]

    # RSI zone
    rsi = _rsi_pandas(close, 14)
    rsi_val = float(rsi.iloc[-1])
    rsi_bullish = 50 <= rsi_val <= 75

    # Supertrend
    st = _supertrend(df)
    st_bullish = price > float(st.iloc[-1]) if not math.isnan(st.iloc[-1]) else False
    st_just_crossed = (
        st_bullish
        and df["close"].iloc[-2] <= st.iloc[-2]
        if not math.isnan(st.iloc[-2]) else False
    )

    # Composite signal
    bull_signals = sum([price_breakout, volume_confirm, ema_aligned, rsi_bullish, st_bullish])
    is_breakout = price_breakout and volume_confirm  # hard requirement

    return {
        "symbol": symbol.upper(),
        "exchange": exchange.upper(),
        "current_price": round(price, 2),
        "is_breakout": is_breakout,
        "breakout_strength": f"{bull_signals}/5 signals bullish",
        "checks": {
            "price_above_20d_high": {
                "result": price_breakout,
                "price": round(price, 2),
                "20d_high": round(high_20d, 2),
                "pct_above": round((price - high_20d) / high_20d * 100, 2),
            },
            "volume_confirmation": {
                "result": volume_confirm,
                "today_volume": int(today_vol),
                "avg_20d_volume": int(avg_vol_20),
                "ratio": round(vol_ratio, 2),
                "threshold": volume_threshold,
            },
            "ema_alignment": {
                "result": ema_aligned,
                "golden_cross_today": golden_cross,
                "ema9": round(float(ema9.iloc[-1]), 2),
                "ema21": round(float(ema21.iloc[-1]), 2),
            },
            "rsi_momentum": {
                "result": rsi_bullish,
                "value": round(rsi_val, 2),
                "zone": "BULLISH" if rsi_bullish else ("OVERBOUGHT" if rsi_val > 75 else "WEAK"),
            },
            "supertrend": {
                "result": st_bullish,
                "just_crossed": st_just_crossed,
                "st_level": round(float(st.iloc[-1]), 2) if not math.isnan(st.iloc[-1]) else None,
            },
        },
    }


@mcp.tool()
def analyze_stock(
    symbol: str,
    exchange: str = "NSE",
) -> dict:
    """
    Full composite technical analysis — combines all tools in one call.

    Aggregates: quote, indicators, S/R levels, breakout scan.
    Returns a single JSON with a trading_bias (BULLISH / BEARISH / NEUTRAL)
    and a plain-English summary suitable for the research engine.

    Args:
        symbol:   Stock symbol, e.g. "DIXON", "RELIANCE", "INFY"
        exchange: "NSE" or "BSE"
    """
    # Run all analyses
    quote = get_quote(symbol, exchange)
    if "error" in quote:
        return {"error": quote["error"], "symbol": symbol}

    indicators = get_indicators(symbol, exchange)
    sr = get_support_resistance(symbol, exchange)
    breakout = scan_breakout(symbol, exchange)

    price = quote["price"]
    sigs = indicators.get("signals", {})

    # ── Scoring ───────────────────────────────────────────────────────────────
    score = 0  # -4 to +4

    trend = sigs.get("trend", "NEUTRAL")
    if "STRONG_UPTREND" in trend:
        score += 2
    elif "UPTREND" in trend:
        score += 1
    elif "DOWNTREND" in trend:
        score -= 1

    momentum = sigs.get("momentum", "NEUTRAL")
    if momentum == "NEUTRAL":
        score += 1
    elif momentum == "OVERBOUGHT":
        score -= 1
    elif momentum == "OVERSOLD":
        score -= 1

    if sigs.get("macd_cross") == "BULLISH_CROSS":
        score += 1
    elif sigs.get("macd_cross") == "BEARISH_CROSS":
        score -= 1

    if sigs.get("supertrend") == "BULLISH":
        score += 1
    else:
        score -= 1

    if breakout.get("is_breakout"):
        score += 1

    # Nearest S/R
    resistances = sr.get("resistance_levels", [])
    supports = sr.get("support_levels", [])
    nearest_r = resistances[0]["price"] if resistances else None
    nearest_s = supports[0]["price"] if supports else None

    # Risk / reward to nearest levels
    rr_ratio = None
    if nearest_r and nearest_s and nearest_s < price < nearest_r:
        upside = nearest_r - price
        downside = price - nearest_s
        rr_ratio = round(upside / downside, 2) if downside else None

    # Trading bias
    if score >= 3:
        bias = "STRONG_BULLISH"
    elif score >= 1:
        bias = "BULLISH"
    elif score <= -3:
        bias = "STRONG_BEARISH"
    elif score <= -1:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    # ── Plain-English summary ─────────────────────────────────────────────────
    vol_label = indicators.get("volume", {}).get("label", "NORMAL")
    rsi_val = indicators.get("rsi", {}).get("value")
    macd_cross = sigs.get("macd_cross")
    st_trend = sigs.get("supertrend")
    is_breakout = breakout.get("is_breakout", False)

    summary_parts = [
        f"{symbol.upper()} ({exchange}) is trading at ₹{price:,.2f}",
        f"({quote['change_pct']:+.2f}% today).",
        f"Trend: {trend}.",
        f"RSI {rsi_val:.0f} — {momentum}." if rsi_val else "",
        f"MACD: {macd_cross}." if macd_cross else "",
        f"Supertrend: {st_trend}.",
        f"Volume: {vol_label}.",
        f"BREAKOUT DETECTED — {breakout.get('breakout_strength', '')}." if is_breakout else "",
        f"Nearest support ₹{nearest_s:,.2f} / resistance ₹{nearest_r:,.2f}." if nearest_s and nearest_r else "",
        f"R:R to nearest levels: {rr_ratio}." if rr_ratio else "",
    ]
    summary = " ".join(p for p in summary_parts if p)

    return {
        "symbol": symbol.upper(),
        "exchange": exchange.upper(),
        "trading_bias": bias,
        "score": score,
        "summary": summary,
        "quote": quote,
        "indicators": indicators,
        "support_resistance": sr,
        "breakout_scan": breakout,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.info("Starting Indian Markets MCP server on stdio…")
    mcp.run(transport="stdio")
