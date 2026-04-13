"""
backtest/walk_forward.py — Walk-forward validation.

Splits the price history into n_splits equal time windows and runs the
backtester independently on each.  This reveals whether strategy performance
is consistent across different market regimes rather than curve-fitted to one
period.

Usage:
    from trading_bot.backtest.walk_forward import run_walk_forward
    from trading_bot.backtest.engine import BacktestParams

    wf = run_walk_forward(df, BacktestParams(), n_splits=3, symbol="DIXON")
    print(wf.summary())
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, Field

from loguru import logger

from trading_bot.backtest.engine import BacktestParams, BacktestResult, run_backtest


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WalkForwardWindow:
    fold: int
    start_date: str
    end_date: str
    n_bars: int
    result: BacktestResult


class WalkForwardResult(BaseModel):
    symbol: str       = ""
    params: dict      = Field(default_factory=dict)
    n_splits: int     = 0
    windows: list[dict] = Field(default_factory=list)   # serialised WalkForwardWindow

    # Aggregate cross-window metrics
    avg_sharpe: float            = 0.0
    avg_max_drawdown_pct: float  = 0.0
    avg_win_rate_pct: float      = 0.0
    avg_cagr_pct: float          = 0.0
    avg_profit_factor: float     = 0.0
    total_trades: int            = 0
    consistent_windows: int      = 0   # windows where sharpe > 0

    def summary(self) -> str:
        return (
            f"WalkForward({self.n_splits} windows) — "
            f"AvgSharpe={self.avg_sharpe:.2f}  "
            f"AvgMaxDD={self.avg_max_drawdown_pct:.1f}%  "
            f"AvgWinRate={self.avg_win_rate_pct:.1f}%  "
            f"AvgCAGR={self.avg_cagr_pct:.1f}%  "
            f"Consistent={self.consistent_windows}/{self.n_splits}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward runner
# ══════════════════════════════════════════════════════════════════════════════

def run_walk_forward(
    df: pd.DataFrame,
    params: BacktestParams,
    n_splits: int = 3,
    symbol: str = "",
) -> WalkForwardResult:
    """
    Split *df* into *n_splits* equal time windows and backtest each independently.

    Args:
        df:       OHLCV DataFrame (lowercase columns, DatetimeIndex preferred).
        params:   Strategy parameters (same for all windows).
        n_splits: Number of equal-length windows to evaluate.
        symbol:   Ticker stored in the result.

    Returns:
        WalkForwardResult with per-window results and aggregate metrics.
    """
    import dataclasses

    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    min_window = params.ema_slow + params.rsi_period + 20
    total_bars = len(df)

    if total_bars < n_splits * min_window:
        logger.warning(
            "Not enough bars ({}) for {} splits (need {} each). Reducing n_splits.",
            total_bars, n_splits, min_window,
        )
        n_splits = max(1, total_bars // min_window)

    window_size = total_bars // n_splits
    windows: list[WalkForwardWindow] = []

    for fold in range(n_splits):
        start = fold * window_size
        end   = start + window_size if fold < n_splits - 1 else total_bars
        chunk = df.iloc[start:end].copy()

        start_str = str(chunk.index[0])[:10] if len(chunk) else "?"
        end_str   = str(chunk.index[-1])[:10] if len(chunk) else "?"

        result = run_backtest(chunk, params, symbol=symbol)
        windows.append(WalkForwardWindow(
            fold       = fold + 1,
            start_date = start_str,
            end_date   = end_str,
            n_bars     = len(chunk),
            result     = result,
        ))
        logger.debug(
            "WF fold {}/{} [{} → {}]: {}",
            fold + 1, n_splits, start_str, end_str, result.summary(),
        )

    # Aggregate
    valid = [w.result for w in windows if w.result.total_trades > 0]

    def _avg(attr: str) -> float:
        vals = [getattr(r, attr) for r in valid]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return WalkForwardResult(
        symbol              = symbol,
        params              = dataclasses.asdict(params),
        n_splits            = len(windows),
        windows             = [
            {
                "fold":       w.fold,
                "start_date": w.start_date,
                "end_date":   w.end_date,
                "n_bars":     w.n_bars,
                "trades":     w.result.total_trades,
                "sharpe":     w.result.sharpe,
                "cagr_pct":   w.result.cagr_pct,
                "max_dd_pct": w.result.max_drawdown_pct,
                "win_rate":   w.result.win_rate_pct,
            }
            for w in windows
        ],
        avg_sharpe           = _avg("sharpe"),
        avg_max_drawdown_pct = _avg("max_drawdown_pct"),
        avg_win_rate_pct     = _avg("win_rate_pct"),
        avg_cagr_pct         = _avg("cagr_pct"),
        avg_profit_factor    = _avg("profit_factor"),
        total_trades         = sum(r.total_trades for r in valid),
        consistent_windows   = sum(1 for r in valid if r.sharpe > 0),
    )
