"""
backtest/optimizer.py — Optuna hyperparameter optimizer for EMA-crossover strategy.

Objective: maximise Sharpe ratio (or win_rate / cagr) on the full dataset using
walk-forward validation as the objective function.  This means each trial's
params are evaluated on multiple OOS windows — not just the full in-sample
period — reducing overfitting risk.

Usage:
    from trading_bot.backtest.optimizer import optimize

    result = optimize(df, symbol="DIXON", n_trials=50, metric="sharpe")
    print(result.best_params)
    print(result.best_value)
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from loguru import logger

from trading_bot.backtest.engine import BacktestParams
from trading_bot.backtest.walk_forward import run_walk_forward


# ══════════════════════════════════════════════════════════════════════════════
# Result model
# ══════════════════════════════════════════════════════════════════════════════

class TrialSummary(BaseModel):
    trial_number: int
    params: dict
    value: float
    state: str   # "complete" | "pruned" | "failed"


class OptimizationResult(BaseModel):
    symbol: str    = ""
    n_trials: int  = 0
    metric: str    = "sharpe"

    best_params: dict  = Field(default_factory=dict)
    best_value: float  = 0.0

    all_trials: list[TrialSummary] = Field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Optimization({self.n_trials} trials, metric={self.metric}) — "
            f"best_{self.metric}={self.best_value:.4f}  "
            f"best_params={self.best_params}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Optimizer
# ══════════════════════════════════════════════════════════════════════════════

OptimizeMetric = Literal["sharpe", "cagr_pct", "win_rate_pct"]


def optimize(
    df: pd.DataFrame,
    symbol: str = "",
    n_trials: int = 50,
    metric: OptimizeMetric = "sharpe",
    n_wf_splits: int = 3,
    n_jobs: int = 1,
) -> OptimizationResult:
    """
    Search for optimal EMA-crossover parameters using Optuna + walk-forward.

    The objective for each trial is the walk-forward aggregate of *metric*
    (avg_sharpe / avg_cagr_pct / avg_win_rate_pct).  Using walk-forward inside
    the objective penalises parameters that only work in one market regime.

    Args:
        df:         OHLCV DataFrame (lowercase columns).
        symbol:     Ticker stored in the result.
        n_trials:   Number of Optuna trials.
        metric:     Objective to maximise — "sharpe" | "cagr_pct" | "win_rate_pct".
        n_wf_splits: Walk-forward windows per trial evaluation.
        n_jobs:     Parallel Optuna workers (1 = serial, safe default).

    Returns:
        OptimizationResult with best params and full trial log.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is required for optimisation — pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    _metric_attr = {
        "sharpe":       "avg_sharpe",
        "cagr_pct":     "avg_cagr_pct",
        "win_rate_pct": "avg_win_rate_pct",
    }
    attr = _metric_attr[metric]

    def objective(trial: "optuna.Trial") -> float:  # noqa: F821
        ema_fast = trial.suggest_int("ema_fast", 5, 30)
        ema_slow = trial.suggest_int("ema_slow", 20, 100)

        if ema_fast >= ema_slow:
            raise optuna.TrialPruned()

        params = BacktestParams(
            ema_fast         = ema_fast,
            ema_slow         = ema_slow,
            rsi_period       = trial.suggest_int("rsi_period", 10, 21),
            rsi_entry_min    = trial.suggest_float("rsi_entry_min", 30.0, 55.0),
            rsi_entry_max    = trial.suggest_float("rsi_entry_max", 55.0, 80.0),
            atr_sl_mult      = trial.suggest_float("atr_sl_mult", 1.0, 3.0),
            atr_tp_mult      = trial.suggest_float("atr_tp_mult", 1.5, 5.0),
        )

        wf = run_walk_forward(df, params, n_splits=n_wf_splits, symbol=symbol)
        return float(getattr(wf, attr))

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=42),
        pruner     = optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)

    # Collect trial summaries
    trial_summaries = []
    for t in study.trials:
        trial_summaries.append(TrialSummary(
            trial_number = t.number,
            params       = t.params,
            value        = t.value if t.value is not None else float("nan"),
            state        = str(t.state).split(".")[-1].lower(),
        ))

    best = study.best_trial
    best_params = BacktestParams(
        ema_fast         = best.params["ema_fast"],
        ema_slow         = best.params["ema_slow"],
        rsi_period       = best.params["rsi_period"],
        rsi_entry_min    = best.params["rsi_entry_min"],
        rsi_entry_max    = best.params["rsi_entry_max"],
        atr_sl_mult      = best.params["atr_sl_mult"],
        atr_tp_mult      = best.params["atr_tp_mult"],
    )

    logger.success(
        "Optimisation complete — best {}={:.4f}  params={}",
        metric, best.value, dataclasses.asdict(best_params),
    )

    return OptimizationResult(
        symbol      = symbol,
        n_trials    = len(study.trials),
        metric      = metric,
        best_params = dataclasses.asdict(best_params),
        best_value  = round(float(best.value), 6),
        all_trials  = trial_summaries,
    )
