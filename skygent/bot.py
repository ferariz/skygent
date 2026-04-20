"""
skygent/bot.py — Telegram bot polling entry point
==================================================

Runs the inbound Telegram bot as a standalone process alongside the
FastAPI server. Uses long-polling (getUpdates) — no public URL required.

Design decisions
----------------
1. Separate process from uvicorn: the bot polling loop is a blocking
   synchronous loop that does not belong inside the FastAPI/asyncio process.
   Running it separately keeps both processes simple and independently
   restartable.

2. Long-polling with timeout=30: Telegram holds the connection open for up
   to 30 seconds waiting for new updates. This means the bot responds within
   1 second of a user message with minimal API calls (one per 30s when idle).

3. Offset tracking: after processing a batch of updates, we send the highest
   update_id + 1 as the offset on the next getUpdates call. Telegram then
   only returns newer updates — preventing re-processing of old messages.

4. Shared SQLite DB: the bot reads/writes the same skygent.db as the API
   server. SQLite's WAL mode handles concurrent reads safely. The bot writes
   conversation_states; the API writes profiles, snapshots, alerts.

5. Graceful shutdown on Ctrl+C: KeyboardInterrupt exits the loop cleanly.

Run:
    python -m skygent.bot
    # or
    python skygent/bot.py

Requires:
    TELEGRAM_BOT_TOKEN  — bot token from @BotFather
    OPENAI_API_KEY      — for welcome forecast narrative generation
    SKYGENT_API_URL     — FastAPI base URL (default: http://localhost:8000)
"""

from __future__ import annotations

import logging
import time

from skygent.api.database import create_db_and_tables
from skygent.integrations.telegram_bot import dispatch, get_updates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def run_bot() -> None:
    """
    Main polling loop. Runs indefinitely until interrupted.

    Flow per iteration:
    1. Call getUpdates with current offset (long-poll, timeout=30s)
    2. Dispatch each update to the appropriate handler
    3. Advance offset to last_update_id + 1
    4. Repeat

    On exception: log the error and wait 5 seconds before retrying.
    This prevents a single bad update or transient network error from
    crashing the bot permanently.
    """
    # Ensure DB tables exist (including conversation_states)
    create_db_and_tables()

    logger.info("bot: starting polling loop")
    logger.info("bot: send /start in Telegram to begin")

    offset: int | None = None

    while True:
        try:
            updates = get_updates(offset)

            for update in updates:
                update_id = update.get("update_id", 0)
                try:
                    dispatch(update)
                except Exception as exc:
                    logger.error(
                        "bot: unhandled error dispatching update %d: %s",
                        update_id, exc,
                    )
                # Always advance offset even if dispatch failed
                offset = update_id + 1

        except KeyboardInterrupt:
            logger.info("bot: shutting down")
            break
        except Exception as exc:
            logger.error("bot: polling error: %s — retrying in 5s", exc)
            time.sleep(5)


if __name__ == "__main__":
    run_bot()