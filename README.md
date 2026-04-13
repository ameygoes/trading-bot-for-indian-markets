# Trading Bot for Indian Markets

An incremental, production-grade trading bot targeting NSE/BSE equities.
Built in Python 3.14 with async-first design, Pydantic v2 models, and a Telegram-based human-in-the-loop approval flow.

## Features built so far

| Step | Module | What it does |
|------|--------|--------------|
| 1 | `config.py`, `main.py` | Environment config, entry-point scaffold |
| 2 | `notifications/` | Notification gateway — console, Telegram, OpenClaw providers |
| 3 | `mcp_server.py` | FastMCP server exposing NSE/BSE TA tools (breakout scan, stock analysis) |
| 4 | `research/` | Per-symbol deep research — fundamentals, news, sentiment, AI recommendation |
| 5 | `discovery/` | Stock screener — NIFTY50 + midcap watchlist, social + breakout scoring |
| 6 | `backtest/` | EMA-crossover backtester, walk-forward validator, Optuna hyperparameter optimizer |
| 7 | `notifications/telegram_provider.py` | Full approval flow — YES/NO/WAIT/DETAILS inline buttons, /status command |

## Quick start

```bash
git clone <repo>
cd trading-bot-for-indian-markets
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env   # fill in your keys
```

## Environment variables

| Variable | Required for | Notes |
|----------|-------------|-------|
| `TELEGRAM_BOT_TOKEN` | Telegram alerts | From @BotFather |
| `TELEGRAM_CHAT_ID` | Telegram alerts | Your chat or group ID |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Discovery social scoring | Falls back gracefully if absent |
| `ANTHROPIC_API_KEY` | AI-powered research | Falls back to rule-based logic if absent |
| `KITE_API_KEY` / `KITE_API_SECRET` | Live trading (Step 10) | Not needed until Step 10 |

Set `NOTIFICATION_PROVIDER=telegram` (default: `console`) to enable Telegram.  
Set `LIVE_TRADING=true` only when a real broker connector is wired (Step 10+).

## Running

```bash
# Run the full bot (scaffold until Steps 9-11 are complete)
python trading_bot/main.py

# Run the discovery scanner
python -m trading_bot.discovery.scanner

# Run a backtest + optimiser on any NSE symbol
python -m trading_bot.backtest DIXON NSE --trials 40 --splits 3

# Start the MCP technical-analysis server standalone
python -m trading_bot.mcp_server
```

## Backtesting

The backtester uses an EMA-crossover strategy with an RSI filter.
Walk-forward validation is used as the Optuna objective to reduce curve-fitting.

```python
from trading_bot.backtest import BacktestParams, run_backtest, optimize
import yfinance as yf

df = yf.Ticker("DIXON.NS").history(period="3y", auto_adjust=True)
df.columns = [c.lower() for c in df.columns]
df = df[["open", "high", "low", "close", "volume"]].dropna()

# Baseline run
result = run_backtest(df, BacktestParams(), symbol="DIXON")
print(result.summary())

# Optimised params
opt = optimize(df, symbol="DIXON", n_trials=50, metric="sharpe")
print(opt.summary())
```

## Telegram approval flow

When a trade candidate is found, the bot sends a structured alert to Telegram:

```
BUY DIXON [NSE]
Style: SWING

Entry      ₹14,250.00
Stop Loss  ₹13,600.00
Target 1   ₹15,500.00
Target 2   ₹16,800.00
Size       4.0% of portfolio
Confidence 78%
R:R        1:1.9

[✅ YES]  [❌ NO]
[⏳ WAIT] [🔍 DETAILS]
```

- **YES** — confirm the trade; message updates to "✅ APPROVED"
- **NO** — reject; message updates to "❌ REJECTED"
- **WAIT** — snooze; message updates to "⏳ SNOOZED"; bot re-alerts at the next candle
- **DETAILS** — sends the full research report (or reasoning text); keyboard switches to YES / NO / WAIT
- **/status** — lists all currently-pending alerts

## Testing

```bash
# Fast unit tests (no network)
pytest -m "not live"

# Single test file
pytest tests/test_backtest.py -v

# Single test by name
pytest tests/test_telegram_provider.py::test_button_details_does_not_resolve_event -v

# Live tests (requires yfinance / Reddit network access)
pytest -m live -v
```

## Architecture overview

```
trading_bot/
├── config.py              # All env vars — every module imports from here
├── main.py                # Entry point (scaffold until Step 11)
├── mcp_server.py          # FastMCP server — scan_breakout, analyze_stock, etc.
├── discovery/             # Screener: social + breakout scoring → DiscoveryCandidate
├── research/              # Deep research: news, sentiment, AI recommendation
├── notifications/         # Alert gateway: console / Telegram / OpenClaw
├── backtest/              # Backtester + walk-forward + Optuna optimizer
│   ├── _indicators.py     #   Pure-pandas EMA / RSI / ATR (no external deps)
│   ├── engine.py          #   BacktestParams, run_backtest(), _compute_metrics()
│   ├── walk_forward.py    #   run_walk_forward() — N equal OOS windows
│   ├── optimizer.py       #   optimize() — Optuna TPE, WF as objective
│   └── strategies/
│       └── ema_crossover.py  # add_signals() — EMA cross + RSI filter
├── analysis/              # Placeholder
└── trading/               # Placeholder — broker connector (Step 10)
```

**Key design rules:**

- **yfinance suffix** — always `.NS` for NSE, `.BO` for BSE. Use `_to_yf_ticker()` in `mcp_server.py`.
- **Config** — all env vars from `trading_bot.config`, never `os.environ` directly.
- **MCP tools** — `scan_breakout`, `analyze_stock`, etc. are importable callables; no MCP server process needed.
- **Trading guard** — `config.LIVE_TRADING` defaults to `False`; every order path must check it.
- **Test marking** — network tests use `@pytest.mark.live`; CI runs `pytest -m "not live"`.

## Planned next steps

| Step | Module | What |
|------|--------|------|
| 8 | `trading_bot/watcher/` | Per-symbol polling threads (market hours only) |
| 9 | `trading_bot/portfolio/` | Portfolio manager — 24/7 discovery + research refresh loop |
| 10 | `trading_bot/trading/` | Zerodha Kite broker connector (paper + live) |
| 11 | `trading_bot/main.py` | Wire all layers into full end-to-end integration |
