"""
main.py — Entry point for the Indian Markets Trading Bot.

Starts all background threads/tasks in order:
  1. Portfolio manager (24/7 discovery + research refresh)
  2. Symbol watcher thread pool (market hours only)
  3. Notification gateway health check
  4. Graceful shutdown on SIGINT / SIGTERM
"""

from __future__ import annotations

import asyncio
import signal
import sys

from loguru import logger

from trading_bot import config  # noqa: F401 — ensures .env is loaded


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )
    logger.add(
        "logs/trading_bot.log",
        rotation="100 MB",
        retention="30 days",
        serialize=True,  # structured JSON
        level="DEBUG",
    )


async def _main() -> None:
    _configure_logging()
    logger.info("Indian Markets Trading Bot starting up…")
    logger.info(f"Mode: {'LIVE' if config.LIVE_TRADING else 'PAPER'}")
    logger.info(f"Notification provider: {config.NOTIFICATION_PROVIDER}")

    # TODO (Step 2+): wire in gateway, portfolio_manager, symbol_watcher
    logger.info("Scaffold ready — awaiting Step 2 implementation.")

    try:
        # Keep alive until cancelled
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        logger.info("Shutdown signal received — saving state…")


def _handle_signal(loop: asyncio.AbstractEventLoop) -> None:
    logger.info("SIGINT/SIGTERM received — initiating graceful shutdown.")
    for task in asyncio.all_tasks(loop):
        task.cancel()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal, loop)

    try:
        loop.run_until_complete(_main())
    finally:
        loop.close()
        logger.info("Bot shut down cleanly.")
