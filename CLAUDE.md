# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate venv (always do this first)
source .venv/bin/activate

# Run all fast unit tests (no network calls)
pytest -m "not live"

# Run a single test file
pytest tests/test_discovery.py -v

# Run a single test by name
pytest tests/test_discovery.py::test_extract_uppercase_ticker -v

# Run live tests (require yfinance / Reddit network access)
pytest -m live -v

# Start the MCP server standalone
python -m trading_bot.mcp_server

# Run the discovery scanner CLI
python -m trading_bot.discovery.scanner

# Run backtest CLI (downloads 3y of data, runs optimizer, prints results)
python -m trading_bot.backtest DIXON NSE --trials 40 --splits 3

# Run the full bot (mostly a scaffold until Steps 7–11 are wired)
python trading_bot/main.py
```

## Architecture

This is an incremental build — each step adds a layer. Steps 1–5 are complete.

```
trading_bot/
├── config.py           # All env vars loaded here; every other module imports from here
├── main.py             # Entry point — orchestrates background tasks (scaffold until ~Step 9)
├── mcp_server.py       # FastMCP server exposing NSE/BSE TA tools (Step 3)
├── discovery/          # Stock screener / candidate ranking (Step 5)
│   ├── symbols.py      #   NIFTY50_SYMBOLS, MIDCAP_WATCHLIST, ALL_WATCHLIST, extract_symbols_from_text()
│   └── scanner.py      #   DiscoveryScanner (social + breakout paths), DiscoveryCandidate, _compute_combined_score()
├── research/           # Per-symbol deep research (Step 4)
│   ├── report.py       #   ResearchReport (Pydantic), TechnicalSummary, Fundamentals, Recommendation enum
│   ├── engine.py       #   ResearchEngine — calls MCP tools, Claude AI (or rule-based fallback)
│   ├── news_fetcher.py #   fetch_reddit_posts(), fetch_yfinance_news()
│   └── sentiment.py    #   VADER-based score_text(), aggregate_sentiment()
├── notifications/      # Alert delivery layer (Step 2)
│   ├── base.py         #   NotificationProvider ABC, StockAlert, ApprovalResponse
│   ├── gateway.py      #   get_gateway() singleton factory; NOTIFICATION_PROVIDER selects provider
│   ├── telegram_provider.py
│   ├── console_provider.py
│   └── openclaw_provider.py
├── backtest/           # Backtester + Optuna optimizer (Step 6)
│   ├── _indicators.py  #   Pure-pandas EMA/RSI/ATR (self-contained, no mcp_server dep)
│   ├── engine.py       #   BacktestParams, Trade, BacktestResult, run_backtest()
│   ├── walk_forward.py #   run_walk_forward() — time-window consistency validation
│   ├── optimizer.py    #   optimize() — Optuna TPE, WF objective, OptimizationResult
│   └── strategies/
│       └── ema_crossover.py  # add_signals() — EMA cross + RSI filter
├── analysis/           # Placeholder (unused)
└── trading/            # Placeholder for Step 10+ (broker connector)
```

## Key Design Rules

**yfinance ticker suffix**: Always append `.NS` for NSE or `.BO` for BSE when calling yfinance. The helper `_to_yf_ticker(sym, exchange)` in `mcp_server.py` does this.

**Config import**: All env vars come from `trading_bot.config`. Never read `os.environ` directly in other modules.

**MCP tools are reusable callables**: `mcp_server.py` functions (`scan_breakout`, `analyze_stock`, etc.) are importable and called directly by the research engine — they don't require the MCP server to be running.

**Notification flow**: `get_gateway()` → `StockAlert` → `send_alert()` → `poll_reply()`. The provider is swapped by `NOTIFICATION_PROVIDER` env var (`console` / `telegram` / `openclaw`).

**Discovery scoring**: `_compute_combined_score()` — 50% technical (breakout + score) + 30% social score + 20% sentiment direction bonus. Score is 0..1.

**Test marking**: Tests that hit real APIs must be marked `@pytest.mark.live`. The CI-safe suite is `pytest -m "not live"`. `asyncio_mode = auto` is set globally so no `@pytest.mark.asyncio` needed.

**Trading mode guard**: `config.LIVE_TRADING` defaults to `False`. Every order execution path must check this before placing real orders.

## Planned Steps (not yet built)

| Step | Module | What |
|------|--------|------|
| ~~6~~ | ~~`trading_bot/backtest/`~~ | ~~Backtester + Optuna optimizer~~ ✅ |
| ~~7~~ | ~~`trading_bot/notifications/telegram_provider.py`~~ | ~~Full approval flow with YES/NO/WAIT/DETAILS inline buttons~~ ✅ |
| 8 | `trading_bot/watcher/` | Per-symbol polling threads (market hours only) |
| 9 | `trading_bot/portfolio/` | Portfolio manager — 24/7 discovery + research refresh loop |
| 10 | `trading_bot/trading/` | Zerodha Kite broker connector (paper + live) |
| 11 | `trading_bot/main.py` | Wire all layers into full end-to-end integration |

## Environment Setup

```bash
cp .env.template .env   # fill in real values
```

Required for live features: `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`, `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET`, `ANTHROPIC_API_KEY`, `KITE_API_KEY` + `KITE_API_SECRET`.

Optional but tested without: Reddit creds (discovery falls back gracefully), Anthropic key (research engine falls back to rule-based logic).
