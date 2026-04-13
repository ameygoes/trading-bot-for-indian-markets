"""
Backtesting + hyperparameter optimisation for NSE/BSE strategies.

Quick start:
    from trading_bot.backtest import BacktestParams, run_backtest, run_walk_forward, optimize
"""

from .engine import BacktestParams, BacktestResult, Trade, run_backtest
from .walk_forward import WalkForwardResult, run_walk_forward
from .optimizer import OptimizationResult, optimize

__all__ = [
    "BacktestParams",
    "BacktestResult",
    "Trade",
    "run_backtest",
    "WalkForwardResult",
    "run_walk_forward",
    "OptimizationResult",
    "optimize",
]
