"""
CLI: python -m trading_bot.backtest [SYMBOL] [EXCHANGE] [--trials N] [--splits N]

Examples:
    python -m trading_bot.backtest DIXON NSE
    python -m trading_bot.backtest RELIANCE NSE --trials 30 --splits 3
"""

from __future__ import annotations

import argparse
import json
import sys

import yfinance as yf
from loguru import logger

from trading_bot.backtest.engine import BacktestParams, run_backtest
from trading_bot.backtest.walk_forward import run_walk_forward
from trading_bot.backtest.optimizer import optimize


def _fetch(symbol: str, exchange: str, period: str = "3y") -> "pd.DataFrame":
    import pandas as pd
    suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
    ticker = yf.Ticker(f"{symbol}{suffix}")
    df = ticker.history(period=period, auto_adjust=True)
    if df.empty:
        logger.error("No data returned for {}{}", symbol, suffix)
        sys.exit(1)
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description="EMA-crossover backtest + optimiser")
    parser.add_argument("symbol",   nargs="?", default="DIXON")
    parser.add_argument("exchange", nargs="?", default="NSE")
    parser.add_argument("--trials", type=int, default=40,
                        help="Optuna trials for optimisation (default 40)")
    parser.add_argument("--splits", type=int, default=3,
                        help="Walk-forward windows (default 3)")
    parser.add_argument("--metric", default="sharpe",
                        choices=["sharpe", "cagr_pct", "win_rate_pct"],
                        help="Optimisation objective (default sharpe)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    symbol, exchange = args.symbol.upper(), args.exchange.upper()
    logger.info("Fetching 3y of data for {} ({})", symbol, exchange)
    df = _fetch(symbol, exchange)
    logger.info("  {} bars loaded ({} → {})", len(df),
                str(df.index[0])[:10], str(df.index[-1])[:10])

    # ── 1. Quick baseline with default params ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BASELINE  ({symbol}, default params)")
    print(f"{'='*60}")
    baseline = run_backtest(df, BacktestParams(), symbol=symbol)
    print(f"  {baseline.summary()}")

    # ── 2. Walk-forward validation of default params ──────────────────────────
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD  ({args.splits} windows, default params)")
    print(f"{'='*60}")
    wf_default = run_walk_forward(df, BacktestParams(),
                                  n_splits=args.splits, symbol=symbol)
    print(f"  {wf_default.summary()}")
    for w in wf_default.windows:
        print(f"    Fold {w['fold']}: {w['start_date']} → {w['end_date']}  "
              f"trades={w['trades']}  sharpe={w['sharpe']:.2f}  "
              f"cagr={w['cagr_pct']:.1f}%")

    # ── 3. Optimisation ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  OPTIMISING  ({args.trials} trials, metric={args.metric})")
    print(f"{'='*60}")
    opt = optimize(df, symbol=symbol, n_trials=args.trials,
                   metric=args.metric, n_wf_splits=args.splits)
    print(f"  {opt.summary()}")

    # ── 4. Walk-forward with optimised params ─────────────────────────────────
    from trading_bot.backtest.engine import BacktestParams as BP
    best_p = BP(**opt.best_params)
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD  ({args.splits} windows, OPTIMISED params)")
    print(f"{'='*60}")
    wf_opt = run_walk_forward(df, best_p, n_splits=args.splits, symbol=symbol)
    print(f"  {wf_opt.summary()}")
    for w in wf_opt.windows:
        print(f"    Fold {w['fold']}: {w['start_date']} → {w['end_date']}  "
              f"trades={w['trades']}  sharpe={w['sharpe']:.2f}  "
              f"cagr={w['cagr_pct']:.1f}%")

    # ── 5. JSON dump ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    output = {
        "symbol":           symbol,
        "baseline":         baseline.model_dump(exclude={"trades", "equity_curve"}),
        "walk_forward_default": wf_default.model_dump(),
        "optimization":     opt.model_dump(exclude={"all_trials"}),
        "walk_forward_optimised": wf_opt.model_dump(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
