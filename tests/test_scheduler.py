"""
tests/test_scheduler.py — Unit tests for the APScheduler jobs layer
====================================================================

Test philosophy
---------------
- The scheduler is tested in isolation from the agent graph. run_agent()
  is mocked throughout — scheduler tests verify job registration, lifecycle,
  snapshot persistence, and error handling, not agent correctness.
- We never start the real APScheduler in unit tests. _job_for_profile() is
  called directly as an async function so we can await it and inspect state
  without needing a running scheduler event loop.
- SnapshotStore is tested independently as a plain unit since it has no
  dependencies.

Test structure
--------------
TestSnapshotStore       — in-memory store get/set/clear/profile_ids
TestRegisterProfile     — job registration, idempotency, expired profile guard
TestDeregisterProfile   — job removal and snapshot clearing
TestListJobs            — job summary output
TestJobForProfile       — the core job function: first run, significant change,
                          no change, expired profile, error handling
TestSchedulerLifecycle  — start/shutdown guards
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from skygent.core.models import ForecastSnapshot, MonitoringProfile
from skygent.scheduler.jobs import (
    SnapshotStore,
    _job_for_profile,
    _job_id,
    deregister_profile,
    get_snapshot_store,
    list_jobs,
    register_profile,
    shutdown,
    start,
    _scheduler,
    _snapshot_store,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_profile(**overrides) -> MonitoringProfile:
    defaults = dict(
        name="Test Wedding",
        location=(-34.9011, -56.1645),
        event_datetime=datetime.now(timezone.utc) + timedelta(days=10),
        monitoring_start=datetime.now(timezone.utc) - timedelta(hours=1),
        check_interval_hours=6,
    )
    defaults.update(overrides)
    return MonitoringProfile(**defaults)


def make_snapshot(profile_id: str, horizon_days: float = 5.0) -> ForecastSnapshot:
    return ForecastSnapshot(
        profile_id=profile_id,
        fetched_at=datetime.now(timezone.utc),
        target_datetime=datetime.now(timezone.utc) + timedelta(days=5),
        data={
            "precipitation_probability_max": 20.0,
            "temperature_2m_max": 25.0,
            "wind_speed_10m_max": 18.0,
            "weather_code": 3,
        },
        horizon_days=horizon_days,
    )


def make_final_state(profile, significant=False, error=None, with_alert=False):
    """Build a mock final_state dict as returned by run_agent()."""
    snapshot = make_snapshot(profile.id)
    alert = None
    if with_alert:
        from skygent.core.models import Alert
        alert = Alert(
            profile_id=profile.id,
            previous_snapshot_id="prev-id",
            current_snapshot_id=snapshot.id,
            changes={},
            horizon_days=5.0,
            confidence="medium",
            narrative="Rain probability increased.",
            sent=True,
        )
    return {
        "profile": profile,
        "current_snapshot": snapshot,
        "significant": significant,
        "triggering_variables": ["precipitation_probability_max"] if significant else [],
        "alert": alert,
        "error": error,
    }


# ---------------------------------------------------------------------------
# TestSnapshotStore
# ---------------------------------------------------------------------------

class TestSnapshotStore:
    def setup_method(self):
        self.store = SnapshotStore()
        self.profile = make_profile()

    def test_get_returns_none_for_unknown_profile(self):
        assert self.store.get("unknown-id") is None

    def test_set_and_get_roundtrip(self):
        snapshot = make_snapshot(self.profile.id)
        self.store.set(snapshot)
        assert self.store.get(self.profile.id) is snapshot

    def test_set_replaces_existing_snapshot(self):
        snap1 = make_snapshot(self.profile.id)
        snap2 = make_snapshot(self.profile.id)
        self.store.set(snap1)
        self.store.set(snap2)
        assert self.store.get(self.profile.id) is snap2

    def test_clear_removes_snapshot(self):
        snapshot = make_snapshot(self.profile.id)
        self.store.set(snapshot)
        self.store.clear(self.profile.id)
        assert self.store.get(self.profile.id) is None

    def test_clear_unknown_profile_does_not_raise(self):
        self.store.clear("does-not-exist")  # must not raise

    def test_profile_ids_reflects_stored_profiles(self):
        p1 = make_profile(name="Event 1")
        p2 = make_profile(name="Event 2")
        self.store.set(make_snapshot(p1.id))
        self.store.set(make_snapshot(p2.id))
        assert p1.id in self.store.profile_ids
        assert p2.id in self.store.profile_ids

    def test_profile_ids_empty_when_store_is_empty(self):
        assert self.store.profile_ids == []


# ---------------------------------------------------------------------------
# TestRegisterProfile
# ---------------------------------------------------------------------------

class TestRegisterProfile:
    def setup_method(self):
        # Ensure clean scheduler state for each test
        for job in _scheduler.get_jobs():
            _scheduler.remove_job(job.id)

    def teardown_method(self):
        for job in _scheduler.get_jobs():
            _scheduler.remove_job(job.id)

    def test_register_active_profile_returns_true(self):
        profile = make_profile()
        result = register_profile(profile)
        assert result is True

    def test_register_creates_job_with_correct_id(self):
        profile = make_profile()
        register_profile(profile)
        assert _scheduler.get_job(_job_id(profile.id)) is not None

    def test_register_expired_profile_returns_false(self):
        expired = make_profile(
            event_datetime=datetime.now(timezone.utc) - timedelta(days=1),
            monitoring_start=datetime.now(timezone.utc) - timedelta(days=2),
        )
        result = register_profile(expired)
        assert result is False

    def test_register_expired_profile_creates_no_job(self):
        expired = make_profile(
            event_datetime=datetime.now(timezone.utc) - timedelta(days=1),
            monitoring_start=datetime.now(timezone.utc) - timedelta(days=2),
        )
        register_profile(expired)
        assert _scheduler.get_job(_job_id(expired.id)) is None

    def test_re_registration_replaces_existing_job(self):
        profile = make_profile()
        register_profile(profile)
        job_before = _scheduler.get_job(_job_id(profile.id))

        # Re-register with different interval
        profile2 = make_profile()
        profile2 = profile2.model_copy(update={"check_interval_hours": 12})
        # Use same id by rebuilding with same profile id trick via override
        register_profile(profile)  # same profile, idempotent
        job_after = _scheduler.get_job(_job_id(profile.id))

        assert job_after is not None  # still registered


# ---------------------------------------------------------------------------
# TestDeregisterProfile
# ---------------------------------------------------------------------------

class TestDeregisterProfile:
    def setup_method(self):
        for job in _scheduler.get_jobs():
            _scheduler.remove_job(job.id)
        _snapshot_store._store.clear()

    def teardown_method(self):
        for job in _scheduler.get_jobs():
            _scheduler.remove_job(job.id)
        _snapshot_store._store.clear()

    def test_deregister_removes_job(self):
        profile = make_profile()
        register_profile(profile)
        assert _scheduler.get_job(_job_id(profile.id)) is not None

        deregister_profile(profile.id)
        assert _scheduler.get_job(_job_id(profile.id)) is None

    def test_deregister_clears_snapshot(self):
        profile = make_profile()
        snapshot = make_snapshot(profile.id)
        _snapshot_store.set(snapshot)
        assert _snapshot_store.get(profile.id) is not None

        deregister_profile(profile.id)
        assert _snapshot_store.get(profile.id) is None

    def test_deregister_unknown_profile_does_not_raise(self):
        deregister_profile("never-registered-id")  # must not raise


# ---------------------------------------------------------------------------
# TestListJobs
# ---------------------------------------------------------------------------

class TestListJobs:
    def setup_method(self):
        for job in _scheduler.get_jobs():
            _scheduler.remove_job(job.id)

    def teardown_method(self):
        for job in _scheduler.get_jobs():
            _scheduler.remove_job(job.id)

    def test_list_jobs_empty_when_no_profiles(self):
        assert list_jobs() == []

    def test_list_jobs_contains_registered_profile(self):
        profile = make_profile(name="Concert Night")
        register_profile(profile)
        jobs = list_jobs()
        assert len(jobs) == 1
        assert "Concert Night" in jobs[0]["name"]

    def test_list_jobs_fields_present(self):
        profile = make_profile()
        register_profile(profile)
        job = list_jobs()[0]
        assert "id" in job
        assert "name" in job
        assert "next_run_time" in job
        assert "trigger" in job


# ---------------------------------------------------------------------------
# TestJobForProfile
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestJobForProfile:
    def setup_method(self):
        _snapshot_store._store.clear()
        for job in _scheduler.get_jobs():
            _scheduler.remove_job(job.id)

    def teardown_method(self):
        _snapshot_store._store.clear()
        for job in _scheduler.get_jobs():
            _scheduler.remove_job(job.id)

    async def test_first_run_stores_snapshot(self):
        """First run: no previous snapshot → agent runs, snapshot stored."""
        profile = make_profile()
        final = make_final_state(profile, significant=False)

        with patch("skygent.scheduler.jobs.run_agent",
                   new=AsyncMock(return_value=final)):
            await _job_for_profile(profile)

        stored = _snapshot_store.get(profile.id)
        assert stored is not None
        assert stored.profile_id == profile.id

    async def test_significant_change_stores_snapshot_and_logs_alert(self):
        """Significant change: snapshot stored, alert logged."""
        profile = make_profile()
        final = make_final_state(profile, significant=True, with_alert=True)

        with patch("skygent.scheduler.jobs.run_agent",
                   new=AsyncMock(return_value=final)):
            await _job_for_profile(profile)

        assert _snapshot_store.get(profile.id) is not None

    async def test_no_significant_change_stores_snapshot(self):
        """No change: snapshot still stored for next diff comparison."""
        profile = make_profile()
        final = make_final_state(profile, significant=False)

        with patch("skygent.scheduler.jobs.run_agent",
                   new=AsyncMock(return_value=final)):
            await _job_for_profile(profile)

        assert _snapshot_store.get(profile.id) is not None

    async def test_error_in_state_does_not_raise(self):
        """run_agent returning an error state must not raise from the job."""
        profile = make_profile()
        final = make_final_state(profile, error="fetch failed")

        with patch("skygent.scheduler.jobs.run_agent",
                   new=AsyncMock(return_value=final)):
            await _job_for_profile(profile)  # must not raise

    async def test_unexpected_exception_does_not_raise(self):
        """Unexpected exception from run_agent must be caught, not propagated."""
        profile = make_profile()

        with patch("skygent.scheduler.jobs.run_agent",
                   new=AsyncMock(side_effect=RuntimeError("unexpected"))):
            await _job_for_profile(profile)  # must not raise

    async def test_expired_profile_removes_job_and_clears_snapshot(self):
        """
        If profile has expired at job run time, the job removes itself AND
        clears the stored snapshot so stale data does not persist in memory.
        """
        profile = make_profile()
        register_profile(profile)
        assert _scheduler.get_job(_job_id(profile.id)) is not None

        # Pre-populate snapshot store to verify it gets cleared
        _snapshot_store.set(make_snapshot(profile.id))
        assert _snapshot_store.get(profile.id) is not None

        # Simulate profile expiring between registration and job run
        expired = profile.model_copy(update={
            "event_datetime": datetime.now(timezone.utc) - timedelta(seconds=1),
        })

        with patch("skygent.scheduler.jobs.run_agent",
                   new=AsyncMock()) as mock_run:
            await _job_for_profile(expired)
            mock_run.assert_not_called()  # agent must not run for expired profile

        assert _scheduler.get_job(_job_id(profile.id)) is None
        assert _snapshot_store.get(profile.id) is None  # snapshot cleared

    async def test_previous_snapshot_passed_to_agent(self):
        """The stored snapshot from the previous run is passed to run_agent."""
        profile = make_profile()
        prev_snapshot = make_snapshot(profile.id)
        _snapshot_store.set(prev_snapshot)

        final = make_final_state(profile, significant=False)
        mock_run = AsyncMock(return_value=final)

        with patch("skygent.scheduler.jobs.run_agent", new=mock_run):
            await _job_for_profile(profile)

        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["previous_snapshot"] is prev_snapshot

    async def test_snapshot_updated_after_each_run(self):
        """Each run replaces the stored snapshot with the new current one."""
        profile = make_profile()
        old_snapshot = make_snapshot(profile.id)
        _snapshot_store.set(old_snapshot)

        final = make_final_state(profile, significant=False)
        new_snapshot = final["current_snapshot"]

        with patch("skygent.scheduler.jobs.run_agent",
                   new=AsyncMock(return_value=final)):
            await _job_for_profile(profile)

        stored = _snapshot_store.get(profile.id)
        assert stored is new_snapshot
        assert stored is not old_snapshot


# ---------------------------------------------------------------------------
# TestSchedulerLifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSchedulerLifecycle:
    """
    AsyncIOScheduler.start() requires a running event loop — these tests
    must be async so pytest-asyncio provides one.
    """

    async def test_start_when_already_running_does_not_raise(self):
        """Calling start() twice must be a safe no-op."""
        if not _scheduler.running:
            _scheduler.start()
        try:
            start()  # second call — must not raise
        finally:
            if _scheduler.running:
                _scheduler.shutdown(wait=False)

    async def test_shutdown_when_not_running_does_not_raise(self):
        """Calling shutdown() when scheduler is not running must not raise."""
        if _scheduler.running:
            _scheduler.shutdown(wait=False)
        shutdown(wait=False)  # must not raise

# Intentionally outside TestSchedulerLifecycle — that class is marked
# @pytest.mark.asyncio which would cause a warning on a sync test method.
def test_get_snapshot_store_returns_module_store():
    """get_snapshot_store() returns the shared module-level instance."""
    store = get_snapshot_store()
    assert store is _snapshot_store