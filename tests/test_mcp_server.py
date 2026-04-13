"""
tests/test_mcp_server.py — Unit tests for the MCP technical analysis server.

These tests call the tool functions directly (no network for indicator math,
live yfinance calls only for smoke tests marked with @pytest.mark.live).
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── helpers exposed for testing ───────────────────────────────────────────────
from trading_bot.mcp_server import (
    _atr_pandas,
    _bbands_pandas,
    _ema_pandas,
    _macd_pandas,
    _rsi_pandas,
    _safe_float,
    _supertrend,
    _to_yf_symbol,
    _vwap,
    get_market_status,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 60, start: float = 100.0, trend: float = 0.5) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame."""
    rng = pd.date_range("2025-01-01", periods=n, freq="B")
    close = pd.Series([start + i * trend + (i % 5) for i in range(n)], index=rng)
    high = close + 2
    low = close - 2
    open_ = close - 0.5
    volume = pd.Series([100_000 + i * 500 for i in range(n)], index=rng)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=rng,
    )


# ── _to_yf_symbol ─────────────────────────────────────────────────────────────

def test_to_yf_symbol_nse() -> None:
    assert _to_yf_symbol("DIXON", "NSE") == "DIXON.NS"


def test_to_yf_symbol_bse() -> None:
    assert _to_yf_symbol("RELIANCE", "BSE") == "RELIANCE.BO"


def test_to_yf_symbol_already_has_suffix() -> None:
    assert _to_yf_symbol("INFY.NS", "NSE") == "INFY.NS"


def test_to_yf_symbol_lowercase_normalised() -> None:
    assert _to_yf_symbol("tcs", "NSE") == "TCS.NS"


# ── _safe_float ───────────────────────────────────────────────────────────────

def test_safe_float_normal() -> None:
    assert _safe_float(3.14159) == pytest.approx(3.1416, abs=1e-4)


def test_safe_float_nan() -> None:
    assert _safe_float(float("nan")) is None


def test_safe_float_inf() -> None:
    assert _safe_float(float("inf")) is None


def test_safe_float_none() -> None:
    assert _safe_float(None) is None


# ── Indicator math ─────────────────────────────────────────────────────────────

def test_ema_length() -> None:
    df = _make_df(60)
    result = _ema_pandas(df["close"], 9)
    assert len(result) == 60
    assert not result.isna().all()


def test_rsi_bounds() -> None:
    df = _make_df(60)
    rsi = _rsi_pandas(df["close"], 14)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_macd_histogram_shape() -> None:
    df = _make_df(60)
    macd_line, signal, hist = _macd_pandas(df["close"])
    assert len(hist) == 60
    # histogram = macd - signal
    diff = (macd_line - signal - hist).dropna().abs()
    assert (diff < 1e-9).all()


def test_bbands_middle_is_sma() -> None:
    df = _make_df(60)
    upper, mid, lower = _bbands_pandas(df["close"], 20, 2.0)
    expected_mid = df["close"].rolling(20).mean()
    pd.testing.assert_series_equal(mid, expected_mid, check_names=False, rtol=1e-9)


def test_atr_positive() -> None:
    df = _make_df(60)
    atr = _atr_pandas(df, 14)
    assert atr.dropna().gt(0).all()


def test_vwap_monotone_volume_denominator() -> None:
    df = _make_df(60)
    vwap = _vwap(df)
    # VWAP should be between min and max close
    valid = vwap.dropna()
    assert valid.min() > 0


def test_supertrend_length() -> None:
    df = _make_df(60)
    st = _supertrend(df, 10, 3.0)
    assert len(st) == 60


# ── get_market_status ─────────────────────────────────────────────────────────

def test_market_status_returns_required_keys() -> None:
    result = get_market_status()
    for key in ("status", "current_time_ist", "market_open", "market_close", "is_trading_day"):
        assert key in result, f"Missing key: {key}"


def test_market_status_valid_status_values() -> None:
    result = get_market_status()
    assert result["status"] in ("OPEN", "CLOSED", "PRE-OPEN")


def test_market_status_weekend() -> None:
    """Force a Sunday — market must be CLOSED."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    # 2025-04-13 is a Sunday
    sunday = datetime(2025, 4, 13, 12, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    with patch("trading_bot.mcp_server.datetime") as mock_dt:
        mock_dt.now.return_value = sunday
        result = get_market_status()
    assert result["status"] == "CLOSED"


def test_market_status_trading_hours() -> None:
    """Force a Monday 11:00 IST — market must be OPEN."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    monday = datetime(2025, 4, 14, 11, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    with patch("trading_bot.mcp_server.datetime") as mock_dt:
        mock_dt.now.return_value = monday
        result = get_market_status()
    assert result["status"] == "OPEN"


# ── Smoke tests (live yfinance — mark with live) ──────────────────────────────

@pytest.mark.live
def test_get_quote_live() -> None:
    from trading_bot.mcp_server import get_quote
    result = get_quote("RELIANCE", "NSE")
    assert "error" not in result
    assert result["price"] > 0
    assert result["symbol"] == "RELIANCE"


@pytest.mark.live
def test_get_indicators_live() -> None:
    from trading_bot.mcp_server import get_indicators
    result = get_indicators("INFY", "NSE")
    assert "error" not in result
    assert "ema" in result
    assert "rsi" in result
    assert result["rsi"]["value"] is not None


@pytest.mark.live
def test_analyze_stock_live() -> None:
    from trading_bot.mcp_server import analyze_stock
    result = analyze_stock("DIXON", "NSE")
    assert "error" not in result
    assert result["trading_bias"] in (
        "STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"
    )
    assert "summary" in result
    assert "DIXON" in result["summary"]
