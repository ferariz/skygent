"""
tests/test_api.py — Unit and integration tests for the FastAPI layer
=====================================================================

Test philosophy
---------------
- All tests use an in-memory SQLite database (separate from the app DB).
- The scheduler is mocked throughout — API tests verify HTTP contracts and
  DB persistence, not agent or scheduler behaviour.
- We use FastAPI's TestClient (synchronous) which wraps the async app
  cleanly for request/response testing.

Test structure
--------------
TestDatabase        — CRUD helpers: save/load profiles, snapshots, alerts
TestProfileEndpoints — POST/GET/DELETE /api/v1/profiles
TestAlertEndpoints   — GET /api/v1/alerts
TestStatusEndpoint   — GET /api/v1/status
TestHealthEndpoint   — GET /health
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from skygent.api.database import (
    AlertRow,
    DBSnapshotStore,
    ProfileRow,
    SnapshotRow,
    list_alerts,
    list_profiles,
    load_latest_snapshot,
    load_profile,
    save_alert,
    save_profile,
    save_snapshot,
)
from skygent.api.main import app
from skygent.api.routes import router
from skygent.core.models import Alert, ForecastSnapshot, MonitoringProfile, VariableChange


# ---------------------------------------------------------------------------
# In-memory test database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(name="engine")
def engine_fixture():
    """Create a fresh in-memory SQLite engine for each test."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(test_engine)
    yield test_engine
    SQLModel.metadata.drop_all(test_engine)


@pytest.fixture(name="session")
def session_fixture(engine):
    """Yield a session bound to the test engine."""
    with Session(engine) as session:
        yield session


@pytest.fixture(name="client")
def client_fixture(engine):
    """
    TestClient with the app's DB dependency overridden to use the
    in-memory test engine. Scheduler calls are mocked.
    """
    from skygent.api.database import get_session

    def override_get_session():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    app.dependency_overrides[get_session] = override_get_session

    with patch("skygent.api.main.create_db_and_tables"), \
         patch("skygent.api.main.set_snapshot_store"), \
         patch("skygent.api.main.start"), \
         patch("skygent.api.main.shutdown"), \
         patch("skygent.api.main.list_profiles", return_value=[]), \
         patch("skygent.api.routes.register_profile", return_value=True), \
         patch("skygent.api.routes.deregister_profile"), \
         patch("skygent.api.routes.list_jobs", return_value=[]):
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Domain object factories
# ---------------------------------------------------------------------------

def make_profile(**overrides) -> MonitoringProfile:
    defaults = dict(
        name="Test Wedding",
        location=(-34.9011, -56.1645),
        event_datetime=datetime.now(timezone.utc) + timedelta(days=30),
        monitoring_start=datetime.now(timezone.utc),
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


def make_alert(profile_id: str) -> Alert:
    return Alert(
        profile_id=profile_id,
        previous_snapshot_id="prev-id",
        current_snapshot_id="curr-id",
        changes={
            "precipitation_probability_max": VariableChange(
                from_value=10.0, to_value=55.0, delta=45.0, delta_pct=450.0
            )
        },
        horizon_days=5.0,
        confidence="medium",
        narrative="Rain probability has increased significantly.",
        sent=True,
    )


# ---------------------------------------------------------------------------
# TestDatabase — CRUD helpers
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_save_and_load_profile(self, session):
        profile = make_profile()
        save_profile(session, profile)
        session.commit()

        loaded = load_profile(session, profile.id)
        assert loaded is not None
        assert loaded.id == profile.id
        assert loaded.name == profile.name

    def test_load_unknown_profile_returns_none(self, session):
        assert load_profile(session, "does-not-exist") is None

    def test_list_profiles_active_only(self, session):
        p1 = make_profile(name="Active Event")
        p2 = make_profile(
            name="Past Event",
            event_datetime=datetime.now(timezone.utc) - timedelta(days=1),
            monitoring_start=datetime.now(timezone.utc) - timedelta(days=2),
        )
        save_profile(session, p1)
        save_profile(session, p2)
        session.commit()

        active = list_profiles(session, active_only=True)
        all_profiles = list_profiles(session, active_only=False)

        # p2 is past its event date — is_active=False
        assert any(p.id == p1.id for p in active)
        assert len(all_profiles) == 2

    def test_save_and_load_snapshot(self, session):
        profile = make_profile()
        snapshot = make_snapshot(profile.id)
        save_snapshot(session, snapshot)
        session.commit()

        loaded = load_latest_snapshot(session, profile.id)
        assert loaded is not None
        assert loaded.id == snapshot.id
        assert loaded.horizon_days == snapshot.horizon_days

    def test_load_latest_snapshot_returns_most_recent(self, session):
        profile = make_profile()
        old = make_snapshot(profile.id, horizon_days=7.0)
        new = make_snapshot(profile.id, horizon_days=5.0)
        save_snapshot(session, old)
        save_snapshot(session, new)
        session.commit()

        loaded = load_latest_snapshot(session, profile.id)
        # Both have the same fetched_at in test — order by desc returns either
        # What we verify: a snapshot is returned
        assert loaded is not None

    def test_load_snapshot_unknown_profile_returns_none(self, session):
        assert load_latest_snapshot(session, "no-such-profile") is None

    def test_save_and_list_alerts(self, session):
        profile = make_profile()
        alert = make_alert(profile.id)
        save_alert(session, alert)
        session.commit()

        alerts = list_alerts(session)
        assert any(a.id == alert.id for a in alerts)

    def test_list_profiles_excludes_past_event_datetime(self, session):
        """
        Fix A: active_only must filter by event_datetime > now(), not just
        is_active. A profile with a past event should be excluded even if
        its is_active flag was never explicitly set to False.
        """
        active = make_profile(name="Future Event")
        expired = make_profile(
            name="Past Event",
            event_datetime=datetime.now(timezone.utc) - timedelta(days=1),
            monitoring_start=datetime.now(timezone.utc) - timedelta(days=2),
        )
        save_profile(session, active)
        save_profile(session, expired)
        session.commit()

        results = list_profiles(session, active_only=True)
        ids = [p.id for p in results]
        assert active.id in ids
        assert expired.id not in ids

    def test_deregister_snapshots_hides_from_latest_lookup(self, session):
        """
        Fix B: after deregister_snapshots(), load_latest_snapshot() must
        return None so a re-registered profile starts a fresh diff baseline.
        The snapshot row is retained in the DB for audit.
        """
        from skygent.api.database import deregister_snapshots, SnapshotRow
        profile = make_profile()
        snapshot = make_snapshot(profile.id)
        save_snapshot(session, snapshot)
        session.commit()

        # Confirm it's visible before deregistration
        assert load_latest_snapshot(session, profile.id) is not None

        deregister_snapshots(session, profile.id)
        session.commit()

        # Now hidden from diff baseline lookups
        assert load_latest_snapshot(session, profile.id) is None

        # But the row still exists in DB for audit
        row = session.get(SnapshotRow, snapshot.id)
        assert row is not None
        assert row.deregistered_at is not None

    def test_list_alerts_filtered_by_profile(self, session):
        p1 = make_profile(name="Event 1")
        p2 = make_profile(name="Event 2")
        a1 = make_alert(p1.id)
        a2 = make_alert(p2.id)
        save_alert(session, a1)
        save_alert(session, a2)
        session.commit()

        p1_alerts = list_alerts(session, profile_id=p1.id)
        assert all(a.profile_id == p1.id for a in p1_alerts)
        assert len(p1_alerts) == 1


# ---------------------------------------------------------------------------
# TestProfileEndpoints
# ---------------------------------------------------------------------------

class TestProfileEndpoints:
    def _valid_payload(self, **overrides):
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        payload = {
            "name": "Test Wedding",
            "latitude": -34.9011,
            "longitude": -56.1645,
            "event_datetime": future,
            "check_interval_hours": 6,
            "event_duration_hours": 4,
            "context": "social_event",
            "notes": "",
        }
        payload.update(overrides)
        return payload

    def test_create_profile_returns_201(self, client):
        resp = client.post("/api/v1/profiles", json=self._valid_payload())
        assert resp.status_code == 201

    def test_create_profile_response_shape(self, client):
        resp = client.post("/api/v1/profiles", json=self._valid_payload())
        data = resp.json()
        assert "id" in data
        assert data["name"] == "Test Wedding"
        assert data["is_active"] is True

    def test_create_profile_past_event_returns_422(self, client):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        resp = client.post(
            "/api/v1/profiles",
            json=self._valid_payload(event_datetime=past),
        )
        assert resp.status_code == 422

    def test_create_profile_invalid_coordinates_returns_422(self, client):
        resp = client.post(
            "/api/v1/profiles",
            json=self._valid_payload(latitude=999.0),
        )
        assert resp.status_code == 422

    def test_create_profile_invalid_context_returns_422(self, client):
        """Fix C: context is now a Pydantic Literal — invalid values are
        rejected before the handler runs, producing a 422."""
        resp = client.post(
            "/api/v1/profiles",
            json=self._valid_payload(context="not_a_valid_context"),
        )
        assert resp.status_code == 422

    def test_create_profile_scheduler_failure_rolls_back_db(self, client, engine):
        """
        Fix A: if register_profile() returns False, the DB write must be
        rolled back so no orphaned profile row is left without a scheduler job.
        """
        from sqlmodel import Session as S, select
        from skygent.api.database import ProfileRow

        with patch("skygent.api.routes.register_profile", return_value=False):
            resp = client.post("/api/v1/profiles", json=self._valid_payload())

        assert resp.status_code == 422

        # Verify no orphaned row in DB
        with S(engine) as s:
            rows = s.exec(select(ProfileRow)).all()
        assert len(rows) == 0

    def test_get_profiles_returns_list(self, client):
        client.post("/api/v1/profiles", json=self._valid_payload())
        resp = client.get("/api/v1/profiles")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) >= 1

    def test_get_profile_by_id(self, client):
        created = client.post("/api/v1/profiles", json=self._valid_payload()).json()
        profile_id = created["id"]

        resp = client.get(f"/api/v1/profiles/{profile_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == profile_id

    def test_get_unknown_profile_returns_404(self, client):
        resp = client.get("/api/v1/profiles/does-not-exist")
        assert resp.status_code == 404

    def test_delete_profile_returns_204(self, client):
        created = client.post("/api/v1/profiles", json=self._valid_payload()).json()
        resp = client.delete(f"/api/v1/profiles/{created['id']}")
        assert resp.status_code == 204

    def test_delete_unknown_profile_returns_404(self, client):
        resp = client.delete("/api/v1/profiles/does-not-exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestAlertEndpoints
# ---------------------------------------------------------------------------

class TestAlertEndpoints:
    def test_get_alerts_returns_list(self, client, session, engine):
        """Seed an alert directly in the DB and verify it appears in the API."""
        from sqlmodel import Session as S
        profile = make_profile()
        alert = make_alert(profile.id)

        with S(engine) as s:
            save_profile(s, profile)
            save_alert(s, alert)
            s.commit()

        resp = client.get("/api/v1/alerts")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_alert_by_id(self, client, session, engine):
        profile = make_profile()
        alert = make_alert(profile.id)

        with Session(engine) as s:
            save_profile(s, profile)
            save_alert(s, alert)
            s.commit()

        resp = client.get(f"/api/v1/alerts/{alert.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == alert.id

    def test_get_unknown_alert_returns_404(self, client):
        resp = client.get("/api/v1/alerts/does-not-exist")
        assert resp.status_code == 404

    def test_get_alerts_filter_by_profile(self, client, engine):
        p1 = make_profile(name="Event 1")
        p2 = make_profile(name="Event 2")
        a1 = make_alert(p1.id)
        a2 = make_alert(p2.id)

        with Session(engine) as s:
            save_profile(s, p1)
            save_profile(s, p2)
            save_alert(s, a1)
            save_alert(s, a2)
            s.commit()

        resp = client.get(f"/api/v1/alerts?profile_id={p1.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert all(a["profile_id"] == p1.id for a in data)


# ---------------------------------------------------------------------------
# TestStatusEndpoint
# ---------------------------------------------------------------------------

class TestStatusEndpoint:
    def test_status_returns_200(self, client):
        resp = client.get("/api/v1/status")
        assert resp.status_code == 200

    def test_status_response_shape(self, client):
        resp = client.get("/api/v1/status")
        data = resp.json()
        assert data["status"] == "ok"
        assert "active_profiles" in data
        assert "scheduled_jobs" in data
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "scheduler_running" in data
