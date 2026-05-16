"""
skygent/api/main.py — FastAPI application entry point
======================================================

Design decisions
----------------
1. Lifespan context manager over on_event: FastAPI deprecated @app.on_event
   in favor of the lifespan context manager. Startup code runs before yield,
   teardown after — making the contract explicit and symmetric.

2. Scheduler started in lifespan: AsyncIOScheduler must start after the
   asyncio event loop is running. The lifespan function runs inside the loop.

3. DB-backed SnapshotStore injected via set_snapshot_store(): jobs.py exposes
   a public setter rather than exposing the private _snapshot_store attribute.
   This decouples main.py from the scheduler's internal naming.

4. Active profiles re-registered on startup: if the application restarts,
   profiles registered in a previous run need to be re-scheduled. The DB is
   the source of truth; the scheduler is ephemeral state rebuilt from it.

5. CORS disabled for MVP: the Streamlit dashboard runs on the same machine.
   Add CORSMiddleware before any public deployment.

Fixes applied after Cursor review (v2)
---------------------------------------
A. Accurate startup log count: we now count successful registrations
   separately from profiles loaded. The final log says "N of M profile(s)
   successfully scheduled" so a mismatch is immediately visible.

B. Per-profile error isolation: register_profile() is wrapped in try/except
   per profile so a single bad row (e.g. corrupt data field) does not abort
   the entire startup sequence. Errors are logged at ERROR level and startup
   continues with remaining profiles.

C. set_snapshot_store() replaces direct _snapshot_store mutation: jobs.py
   now exposes a public setter so this module does not depend on the private
   attribute name remaining stable.

D. logging.basicConfig acknowledged: under uvicorn the server configures
   logging and basicConfig may be ignored or interact oddly. Kept for
   standalone/script use where uvicorn is not managing the log config.
   Safe to remove if uvicorn's --log-config is used in production.

E. shutdown(wait=True) acknowledged as a tradeoff: wait=True gives running
   jobs time to finish cleanly (data integrity). wait=False is faster for
   rapid redeploys. Kept as True for the MVP where correctness > speed.

Run locally:
    uvicorn skygent.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import structlog

from fastapi import FastAPI
from sqlalchemy import text

from skygent.api.database import (
    DBSnapshotStore,
    create_db_and_tables,
    engine,
    get_session_sync,
    list_profiles,
)
from skygent.api.routes import router
from skygent.scheduler.jobs import register_profile, set_snapshot_store, shutdown, start

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    env = os.getenv("ENV", "development")

    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if env == "production"
        else structlog.dev.ConsoleRenderer()
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


logger = logging.getLogger(__name__)
configure_logging()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: create DB tables, inject DB snapshot store, start scheduler,
             re-register all active profiles from DB.
    Shutdown: gracefully stop the scheduler.
    """
    logger.info("startup: initializing database")
    create_db_and_tables()

    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA journal_mode=WAL"))
        logger.info("startup: WAL mode active (journal_mode=%s)", result.scalar())
        conn.execute(text("PRAGMA synchronous=NORMAL"))

    logger.info("startup: injecting DB-backed snapshot store")
    set_snapshot_store(DBSnapshotStore())  # fix C: public setter, not _attr mutation

    logger.info("startup: starting scheduler")
    start()

    logger.info("startup: re-registering active profiles from database")
    with get_session_sync() as session:
        active_profiles = list_profiles(session, active_only=True)

    # fix A + B: accurate count, per-profile error isolation
    registered_count = 0
    for profile in active_profiles:
        try:
            registered = register_profile(profile)
            if registered:
                registered_count += 1
                logger.info("startup: re-registered '%s'", profile.name)
            else:
                logger.warning(
                    "startup: skipped expired profile '%s' (id=%s)",
                    profile.name, profile.id,
                )
        except Exception as exc:
            logger.error(
                "startup: failed to register profile '%s' (id=%s): %s — continuing",
                profile.name, profile.id, exc,
            )

    logger.info(
        "startup: complete — %d of %d profile(s) successfully scheduled",
        registered_count, len(active_profiles),
    )

    yield  # application runs here

    logger.info("shutdown: stopping scheduler (wait=True for clean job completion)")
    shutdown(wait=True)
    logger.info("shutdown: complete")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Skygent",
    description="AI weather monitoring agent — proactive forecast change alerts",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Health check (no prefix — for load balancers / Railway health checks)
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}