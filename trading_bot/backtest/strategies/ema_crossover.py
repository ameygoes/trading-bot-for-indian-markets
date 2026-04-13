"""
backtest/strategies/ema_crossover.py — EMA crossover signal generator.

Entry: EMA(fast) crosses above EMA(slow) AND RSI is in [rsi_entry_min, rsi_entry_max].
Exit:  EMA(fast) crosses below EMA(slow) (or SL/TP hit, handled by the engine).

The RSI filter prevents entering into overbought conditions after a fast move.
"""

from __future__ import annotations

import pandas as pd

from trading_bot.backtest._indicators import ema, rsi, atr
from trading_bot.backtest.engine import BacktestParams


def add_signals(df: pd.DataFrame, params: BacktestParams) -> pd.DataFrame:
    """
    Compute indicators and attach entry/exit signal columns to *df* (copy).

    Added columns:
      ema_fast      — fast EMA of close
      ema_slow      — slow EMA of close
      rsi           — RSI of close
      atr           — ATR (14-period, used by engine for SL/TP)
      entry_signal  — bool: long entry on bar close
      exit_signal   — bool: close long on bar close
    """
    close = df["close"]

    df = df.copy()
    df["ema_fast"] = ema(close, params.ema_fast)
    df["ema_slow"] = ema(close, params.ema_slow)
    df["rsi"]      = rsi(close, params.rsi_period)
    df["atr"]      = atr(df, 14)

    prev_fast = df["ema_fast"].shift(1)
    prev_slow = df["ema_slow"].shift(1)

    cross_above = (df["ema_fast"] > df["ema_slow"]) & (prev_fast <= prev_slow)
    cross_below = (df["ema_fast"] < df["ema_slow"]) & (prev_fast >= prev_slow)

    rsi_ok = (df["rsi"] >= params.rsi_entry_min) & (df["rsi"] <= params.rsi_entry_max)

    df["entry_signal"] = cross_above & rsi_ok
    df["exit_signal"]  = cross_below

    return df
