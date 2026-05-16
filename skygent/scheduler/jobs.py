"""
skygent/scheduler/jobs.py — APScheduler-based polling scheduler
================================================================

Design decisions
----------------
1. AsyncIOScheduler over BackgroundScheduler: the agent graph is fully
   async (LangGraph, httpx). AsyncIOScheduler runs jobs on the existing
   event loop so we avoid the overhead of thread-pool executors and the
   complexity of bridging sync/async boundaries. BackgroundScheduler would
   require asyncio.run() inside each job, which creates a new event loop
   per invocation — wasteful and error-prone.

2. One job per active MonitoringProfile: each profile gets its own
   IntervalTrigger keyed to profile.check_interval_hours. Jobs are
   registered by profile ID so they can be added, paused, and removed
   individually without touching other profiles.

3. In-memory snapshot store (SnapshotStore): the scheduler needs to
   pass the previous snapshot to run_agent() on every poll. For the MVP
   we store the last snapshot per profile in a simple dict. The interface
   (get/set) is designed so it can be swapped for a SQLite/PostgreSQL
   implementation in Step 5 (FastAPI + SQLModel) without changing the
   scheduler logic.

4. Jobs never raise: job_for_profile() catches all exceptions and logs
   them. APScheduler typically keeps the schedule running after a job
   exception, but explicit catching ensures structured logging and
   consistent behavior regardless of backend or version.

5. Scheduler lifecycle is managed by start()/shutdown(): the module exposes
   a single scheduler instance. The FastAPI lifespan handler (Step 5) will
   call start() on startup and shutdown() on teardown. For standalone use
   (scripts, tests) the same functions work directly.

6. Only active profiles are scheduled: register_profile() checks
   profile.is_active before adding a job. The job itself also checks
   is_active at runtime and removes itself if the event has passed — this
   handles the case where a profile expires between scheduler restarts.
"""

from __future__ import annotations

import structlog
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from skygent.agent.graph import run_agent
from skygent.core.models import ForecastSnapshot, MonitoringProfile

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# In-memory snapshot store (MVP — replace with DB in Step 5)
# ---------------------------------------------------------------------------

class SnapshotStore:
    """
    Stores the most recent ForecastSnapshot per profile.

    This is the scheduler's only persistent state for the MVP. It is
    intentionally simple — a dict with get/set — so the FastAPI layer
    can replace it with a SQLModel-backed store without changing the
    scheduler's job logic.

    Thread-safety: AsyncIOScheduler runs all jobs on the event loop thread,
    so this store does not need locking for the MVP. A multi-worker
    deployment would need an async-safe shared store.
    """

    def __init__(self) -> None:
        self._store: dict[str, ForecastSnapshot] = {}

    def get(self, profile_id: str) -> ForecastSnapshot | None:
        """Return the last stored snapshot for a profile, or None."""
        return self._store.get(profile_id)

    def set(self, snapshot: ForecastSnapshot) -> None:
        """Store or replace the snapshot for its profile."""
        self._store[snapshot.profile_id] = snapshot
        logger.debug(
            "SnapshotStore: saved snapshot",
            snapshot_id=str(snapshot.id),
            profile_id=str(snapshot.profile_id),
        )

    def clear(self, profile_id: str) -> None:
        """Remove the stored snapshot for a profile (e.g. when deregistered)."""
        self._store.pop(profile_id, None)

    @property
    def profile_ids(self) -> list[str]:
        """Return all profile IDs with stored snapshots."""
        return list(self._store.keys())


# Module-level store — shared across all jobs.
# Use set_snapshot_store() to replace it (e.g. with DBSnapshotStore at startup)
# rather than mutating _snapshot_store directly from outside this module.
_snapshot_store = SnapshotStore()


def set_snapshot_store(store: SnapshotStore) -> None:
    """
    Replace the module-level snapshot store.

    Called at application startup to swap the in-memory SnapshotStore for
    a DBSnapshotStore. Using an explicit setter decouples callers from the
    private attribute name and makes the injection point visible in the
    module's public API.

    Parameters
    ----------
    store: any object implementing the SnapshotStore get/set/clear interface
    """
    global _snapshot_store
    _snapshot_store = store
    logger.info("scheduler: snapshot store replaced", store_type=type(store).__name__)


# ---------------------------------------------------------------------------
# Scheduler instance
# ---------------------------------------------------------------------------

_scheduler = AsyncIOScheduler()


# ---------------------------------------------------------------------------
# Job implementation
# ---------------------------------------------------------------------------

async def _job_for_profile(profile: MonitoringProfile) -> None:
    """
    One scheduler tick for a single MonitoringProfile.

    Steps:
    1. Check if the profile is still active — remove the job if not.
    2. Load the previous snapshot from the store.
    3. Run the agent graph.
    4. Persist the new snapshot and any generated alert.
    5. Log errors without raising: APScheduler logs job exceptions and
       generally keeps the schedule running, but explicit catching ensures
       structured logging and predictable behavior across all backends.
    """
    log = logger.bind(profile_name=profile.name, profile_id=str(profile.id))
    profile_id = profile.id

    # Guard: remove expired profiles rather than running forever.
    # Also clear the snapshot store so stale data does not persist in memory
    # until the next restart or manual deregistration.
    if not profile.is_active:
        log.info(
            "scheduler: profile has passed its event date — removing job and clearing snapshot",
            job_id=_job_id(str(profile_id)),
        )
        _remove_job(profile_id)
        _snapshot_store.clear(profile_id)
        return

    run_start = datetime.now(timezone.utc)
    previous_snapshot = _snapshot_store.get(profile_id)

    log.info(
        "scheduler: starting run",
        previous_snapshot_id=str(previous_snapshot.id) if previous_snapshot is not None else "none",
    )

    try:
        final_state = await run_agent(profile, previous_snapshot=previous_snapshot)
    except Exception as exc:
        # run_agent should never raise — belt-and-suspenders catch for safety
        log.error(
            "scheduler: unexpected exception in run_agent",
            error=str(exc),
            exc_info=True,
        )
        return

    # Persist the current snapshot regardless of whether an alert fired
    current_snapshot = final_state.get("current_snapshot")
    if current_snapshot is not None:
        _snapshot_store.set(current_snapshot)

    # Log the outcome
    if final_state.get("error"):
        log.error(
            "scheduler: run completed with error",
            error=final_state["error"],
        )
    elif final_state.get("significant"):
        alert = final_state.get("alert")
        log.info(
            "scheduler: alert generated",
            alert_id=str(alert.id) if alert else "unknown",
            confidence=alert.confidence if alert else "unknown",
            horizon_days=alert.horizon_days if alert else 0.0,
            sent=alert.sent if alert else False,
        )
    else:
        elapsed = (datetime.now(timezone.utc) - run_start).total_seconds()
        log.info(
            "scheduler: no significant change",
            duration_ms=int(elapsed * 1000),
        )


def _job_id(profile_id: str) -> str:
    """Deterministic job ID for a profile — used to update/remove jobs."""
    return f"skygent_profile_{profile_id}"


def _remove_job(profile_id: str) -> None:
    """Remove a job from the scheduler if it exists."""
    job_id = _job_id(profile_id)
    if _scheduler.get_job(job_id):
        _scheduler.remove_job(job_id)
        logger.info("scheduler: removed job", profile_id=profile_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_profile(profile: MonitoringProfile) -> bool:
    """
    Register a MonitoringProfile with the scheduler.

    Creates an IntervalTrigger job that calls _job_for_profile() every
    profile.check_interval_hours hours. If a job already exists for this
    profile, it is replaced (allows re-registration after threshold changes).

    Parameters
    ----------
    profile: the profile to schedule

    Returns
    -------
    True if the job was registered, False if the profile is already expired.
    """
    if not profile.is_active:
        logger.warning(
            "scheduler: profile is already past its event date — not scheduling",
            profile_name=profile.name,
            profile_id=str(profile.id),
        )
        return False

    job_id = _job_id(profile.id)

    # Replace existing job if present (idempotent re-registration)
    if _scheduler.get_job(job_id):
        _scheduler.remove_job(job_id)
        logger.info("scheduler: replacing existing job", profile_name=profile.name)

    _scheduler.add_job(
        func=_job_for_profile,
        trigger=IntervalTrigger(hours=profile.check_interval_hours),
        args=[profile],
        id=job_id,
        name=f"Skygent: {profile.name}",
        # Run immediately on registration so the first snapshot is fetched
        # without waiting a full interval. Passing next_run_time=datetime.now()
        # tells APScheduler to fire the job immediately, then follow the interval.
        next_run_time=datetime.now(timezone.utc),
        max_instances=1,          # prevent overlapping runs for the same profile
        misfire_grace_time=300,   # allow up to 5 min late start before skipping
    )

    logger.info(
        "scheduler: registered profile — first run immediate",
        profile_name=profile.name,
        profile_id=str(profile.id),
        interval_hours=profile.check_interval_hours,
    )
    return True


def deregister_profile(profile_id: str) -> None:
    """
    Remove a profile's job from the scheduler and clear its stored snapshot.

    Safe to call even if the profile was never registered.
    """
    _remove_job(profile_id)
    _snapshot_store.clear(profile_id)
    logger.info("scheduler: deregistered profile", profile_id=profile_id)


def list_jobs() -> list[dict]:
    """
    Return a summary of all currently scheduled jobs.

    Used by the FastAPI status endpoint (Step 5).
    """
    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append({
            "id":           job.id,
            "name":         job.name,
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger":      str(job.trigger),
        })
    return jobs


def start() -> None:
    """
    Start the scheduler. Called once at application startup.

    Safe to call multiple times — no-op if already running.
    """
    if _scheduler.running:
        logger.warning("scheduler: already running — start() called again")
        return
    _scheduler.start()
    logger.info("scheduler: started")


def shutdown(wait: bool = True) -> None:
    """
    Shut down the scheduler gracefully.

    Parameters
    ----------
    wait: if True (default), wait for running jobs to finish before returning.
          Pass False for fast shutdown in tests.
    """
    if not _scheduler.running:
        return
    _scheduler.shutdown(wait=wait)
    logger.info("scheduler: shut down")


# ---------------------------------------------------------------------------
# Convenience accessor (for tests and FastAPI)
# ---------------------------------------------------------------------------

def get_snapshot_store() -> SnapshotStore:
    """Return the module-level snapshot store (injectable for tests)."""
    return _snapshot_store
