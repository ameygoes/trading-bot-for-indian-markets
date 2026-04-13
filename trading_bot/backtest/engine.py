"""
backtest/engine.py — Vectorized event-driven backtester.

Execution model:
  - Signals generated at bar close (i-1).
  - Entry / exit executed at next bar's open (bar i).
  - SL and TP checked against bar i's high/low (intrabar).
  - One position at a time (long only).

Usage:
    from trading_bot.backtest.engine import BacktestParams, run_backtest
    import yfinance as yf

    df = yf.Ticker("DIXON.NS").history(period="2y")
    df.columns = [c.lower() for c in df.columns]

    params = BacktestParams(ema_fast=9, ema_slow=21)
    result = run_backtest(df, params, symbol="DIXON")
    print(result.summary())
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, Field

from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BacktestParams:
    """Hyperparameters for the EMA-crossover strategy + position sizing."""
    ema_fast: int   = 9
    ema_slow: int   = 21
    rsi_period: int = 14
    rsi_entry_min: float = 45.0   # only enter when RSI ≥ this (avoids oversold reversals)
    rsi_entry_max: float = 75.0   # only enter when RSI ≤ this (avoids overbought chasing)
    atr_sl_mult: float   = 1.5    # stop-loss  = entry − atr_sl_mult × ATR
    atr_tp_mult: float   = 3.0    # take-profit = entry + atr_tp_mult × ATR
    position_size_pct: float = 0.10  # fraction of current equity per trade


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════

class Trade(BaseModel):
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_pct: float          # % gain/loss on the trade (not the full equity)
    exit_reason: str        # "tp" | "sl" | "signal"
    holding_days: int


class BacktestResult(BaseModel):
    symbol: str = ""
    params: dict = Field(default_factory=dict)

    # Trade statistics
    total_trades: int  = 0
    win_count: int     = 0
    loss_count: int    = 0
    win_rate_pct: float      = 0.0
    profit_factor: float     = 0.0
    avg_holding_days: float  = 0.0

    # Return / risk metrics
    total_return_pct: float  = 0.0
    cagr_pct: float          = 0.0
    sharpe: float            = 0.0
    max_drawdown_pct: float  = 0.0   # negative (e.g. -12.3)

    # Raw series
    trades: list[Trade]      = Field(default_factory=list)
    equity_curve: list[float] = Field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Trades={self.total_trades}  WinRate={self.win_rate_pct:.1f}%  "
            f"Sharpe={self.sharpe:.2f}  MaxDD={self.max_drawdown_pct:.1f}%  "
            f"CAGR={self.cagr_pct:.1f}%"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Core backtester
# ══════════════════════════════════════════════════════════════════════════════

_INITIAL_EQUITY = 100_000.0   # ₹1 lakh paper capital


def run_backtest(
    df: pd.DataFrame,
    params: BacktestParams,
    symbol: str = "",
    initial_equity: float = _INITIAL_EQUITY,
) -> BacktestResult:
    """
    Run a backtest of the EMA-crossover strategy on *df*.

    Args:
        df:             OHLCV DataFrame with lowercase column names
                        (open, high, low, close, volume). DatetimeIndex preferred.
        params:         Strategy and sizing parameters.
        symbol:         Ticker name stored in the result (cosmetic).
        initial_equity: Starting paper capital in ₹.

    Returns:
        BacktestResult with full trade log and performance metrics.
    """
    min_bars = params.ema_slow + params.rsi_period + 10
    if len(df) < min_bars:
        logger.debug("Not enough bars for backtest ({} < {})", len(df), min_bars)
        return BacktestResult(symbol=symbol, params=dataclasses.asdict(params))

    # Import here to avoid circular imports at module level
    from trading_bot.backtest.strategies.ema_crossover import add_signals

    df = add_signals(df, params)
    df = df.dropna(subset=["ema_fast", "ema_slow", "rsi", "atr"]).copy()
    df = df.reset_index(drop=True)

    if len(df) < 5:
        return BacktestResult(symbol=symbol, params=dataclasses.asdict(params))

    trades: list[Trade] = []
    equity = initial_equity
    equity_curve: list[float] = [equity]

    in_position = False
    entry_price = 0.0
    entry_date  = ""
    sl = tp = 0.0
    entry_bar = 0

    def _date_str(idx: int) -> str:
        val = df.index[idx]
        return str(val)[:10]

    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        if in_position:
            exit_price: float | None = None
            exit_reason: str | None  = None

            # Intrabar SL/TP check (priority: SL > TP > signal)
            if float(row["low"]) <= sl:
                exit_price  = sl
                exit_reason = "sl"
            elif float(row["high"]) >= tp:
                exit_price  = tp
                exit_reason = "tp"
            elif bool(prev["exit_signal"]):
                exit_price  = float(row["open"])
                exit_reason = "signal"

            if exit_price is not None:
                pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                equity  += equity * params.position_size_pct * (pnl_pct / 100.0)

                trades.append(Trade(
                    entry_date   = entry_date,
                    exit_date    = _date_str(i),
                    entry_price  = round(entry_price, 4),
                    exit_price   = round(exit_price, 4),
                    pnl_pct      = round(pnl_pct, 4),
                    exit_reason  = exit_reason,
                    holding_days = i - entry_bar,
                ))
                in_position = False

        else:
            if bool(prev["entry_signal"]):
                entry_price = float(row["open"])
                entry_date  = _date_str(i)
                atr_val     = float(prev["atr"])
                sl          = entry_price - params.atr_sl_mult * atr_val
                tp          = entry_price + params.atr_tp_mult * atr_val
                in_position = True
                entry_bar   = i

        equity_curve.append(equity)

    return _compute_metrics(symbol, params, trades, equity_curve, initial_equity)


# ══════════════════════════════════════════════════════════════════════════════
# Metric computation
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(
    symbol: str,
    params: BacktestParams,
    trades: list[Trade],
    equity_curve: list[float],
    initial_equity: float,
) -> BacktestResult:
    base = dict(
        symbol       = symbol,
        params       = dataclasses.asdict(params),
        equity_curve = equity_curve,
        trades       = trades,
    )

    if not trades:
        return BacktestResult(**base)

    wins   = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    gross_profit = sum(t.pnl_pct for t in wins)
    gross_loss   = abs(sum(t.pnl_pct for t in losses))
    profit_factor = (
        round(gross_profit / gross_loss, 4) if gross_loss > 0 else 999.0
    )

    win_rate     = round(len(wins) / len(trades) * 100, 2)
    avg_holding  = round(sum(t.holding_days for t in trades) / len(trades), 1)

    # Equity metrics
    final_equity  = equity_curve[-1]
    total_return  = (final_equity - initial_equity) / initial_equity * 100
    years         = max(len(equity_curve) / 252, 1 / 252)
    cagr          = ((final_equity / initial_equity) ** (1.0 / years) - 1.0) * 100

    # Sharpe (annualised, risk-free rate = 0)
    eq_series     = pd.Series(equity_curve, dtype=float)
    daily_returns = eq_series.pct_change().dropna()
    sharpe        = 0.0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = round(
            float(daily_returns.mean() / daily_returns.std() * math.sqrt(252)), 4
        )

    # Max drawdown
    rolling_max = eq_series.cummax()
    drawdowns   = (eq_series - rolling_max) / rolling_max * 100
    max_dd      = round(float(drawdowns.min()), 4)

    return BacktestResult(
        **base,
        total_trades     = len(trades),
        win_count        = len(wins),
        loss_count       = len(losses),
        win_rate_pct     = win_rate,
        profit_factor    = profit_factor,
        avg_holding_days = avg_holding,
        total_return_pct = round(float(total_return), 4),
        cagr_pct         = round(float(cagr), 4),
        sharpe           = sharpe,
        max_drawdown_pct = max_dd,
    )
