"""
skygent/launcher.py — Single-process launcher for API + Telegram bot
=====================================================================

Runs the FastAPI server (uvicorn) and the Telegram bot polling loop
concurrently in the same OS process:

  - uvicorn.run() is synchronous and blocking — it runs in a thread pool
    executor so it does not block the asyncio event loop
  - run_bot() is also a synchronous blocking loop — it runs in a second
    thread pool executor slot for the same reason
  - asyncio.gather() keeps main() alive until either service exits

Usage:
    python -m skygent.launcher
    # or as a Railway start command:
    python skygent/launcher.py

Environment:
    PORT  — TCP port for uvicorn (default: 8080)
"""

from __future__ import annotations

import asyncio
import os

import uvicorn

from skygent.bot import run_bot


def _run_uvicorn(port: int) -> None:
    uvicorn.run(
        "skygent.api.main:app",
        host="0.0.0.0",
        port=port,
    )


async def main() -> None:
    port = int(os.getenv("PORT", 8080))
    loop = asyncio.get_event_loop()

    uvicorn_future = loop.run_in_executor(None, _run_uvicorn, port)
    bot_future = loop.run_in_executor(None, run_bot)

    await asyncio.gather(uvicorn_future, bot_future)


if __name__ == "__main__":
    asyncio.run(main())
