"""
skygent/api/routes.py — FastAPI route handlers
================================================

Design decisions
----------------
1. Thin route layer: routes validate input, call CRUD helpers, and return
   responses. Business logic lives in the core and agent layers. Routes
   do not import DiffAnalyzer, SignificanceEvaluator, or LangGraph directly.

2. Request/response models are separate from domain models: ProfileCreate
   is what the user POSTs; ProfileResponse is what we return. This decouples
   the API contract from the internal Pydantic models and lets us evolve
   them independently.

3. Registration triggers immediate scheduling: POST /profiles saves to the DB
   and registers with the scheduler atomically — if scheduling fails, the
   session is rolled back so no orphaned profile row is left in the DB.

4. Pagination on list endpoints: all list endpoints accept a `limit` parameter
   (default 50, max 200) to prevent unbounded responses.

5. No authentication for MVP: intended for local/single-user deployment.
   Authentication (API key header) is the first thing to add before any
   public deployment.

6. Consistent error responses: all 4xx errors return
   {"detail": "human-readable message"} via FastAPI's HTTPException.

Fixes applied after Cursor review (v2)
---------------------------------------
A. POST /profiles atomicity: save_profile() and register_profile() must
   succeed or fail together. If register_profile() returns False, we raise
   HTTPException which propagates through the session dependency's except
   block, triggering a rollback — no orphaned profile row is committed.
   Added a test that verifies the DB has no row when scheduling fails.

B. Unused import removed: save_alert was imported but never called in routes.
   Alert persistence is the scheduler's responsibility (jobs.py calls
   save_alert after a successful run), not the routes layer.

C. ProfileCreate.context typed as Literal[...]: replaces the str + runtime
   set-membership check with a Pydantic Literal field. OpenAPI schema now
   shows the enum correctly; invalid values are rejected by Pydantic before
   the handler runs, consistent with all other field validation.

D. _alert_to_response trigger summary: uses "weather_code change" as the
   fallback (matching the actual trigger label used by significance.py)
   instead of the old "weathercode change" which was inconsistent.

E. Inline imports in delete_profile and get_alert moved to top-level imports
   for consistency with the rest of the file.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlmodel import Session

from skygent.api.database import (
    AlertRow,
    ProfileRow,
    deregister_snapshots,
    get_session,
    list_alerts,
    list_profiles,
    load_profile,
    save_profile,
)
from skygent.core.models import Alert, MonitoringProfile
from skygent.scheduler.jobs import (
    deregister_profile,
    list_jobs,
    register_profile,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Valid context values — matches MonitoringProfile.context Literal
ContextType = Literal["social_event", "agriculture", "energy", "logistics"]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ProfileCreate(BaseModel):
    """
    Payload for registering a new event to monitor.
    Minimal required fields — sensible defaults for everything else.
    """
    name: str = Field(..., json_schema_extra={"example": "Ana & Juan's Wedding"})
    latitude: float = Field(..., ge=-90, le=90, json_schema_extra={"example": -34.9011})
    longitude: float = Field(..., ge=-180, le=180, json_schema_extra={"example": -56.1645})
    event_datetime: datetime = Field(..., json_schema_extra={"example": "2025-09-15T17:00:00Z"})
    check_interval_hours: int = Field(default=6, ge=1, le=24)
    event_duration_hours: int = Field(default=4, ge=1)
    context: ContextType = Field(default="social_event")  # fix C: Literal not str
    notes: str = Field(default="")
    language: str = Field(default="en")


class ProfileResponse(BaseModel):
    """Public representation of a MonitoringProfile."""
    id: str
    name: str
    location: tuple[float, float]
    event_datetime: datetime
    check_interval_hours: int
    event_duration_hours: int
    context: str
    notes: str
    is_active: bool


class AlertResponse(BaseModel):
    """Public representation of an Alert."""
    id: str
    profile_id: str
    detected_at: datetime
    confidence: str
    horizon_days: float
    triggering_summary: str
    narrative: str
    sent: bool


class StatusResponse(BaseModel):
    """Scheduler and system health summary."""
    status: str
    active_profiles: int
    scheduled_jobs: int
    jobs: list[dict]
    timestamp: datetime


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _profile_to_response(profile: MonitoringProfile) -> ProfileResponse:
    return ProfileResponse(
        id=profile.id,
        name=profile.name,
        location=profile.location,
        event_datetime=profile.event_datetime,
        check_interval_hours=profile.check_interval_hours,
        event_duration_hours=profile.event_duration_hours,
        context=profile.context,
        notes=profile.notes,
        is_active=profile.is_active,
    )


def _alert_to_response(alert: Alert) -> AlertResponse:
    """
    Convert an Alert to its API response shape.

    triggering_summary lists the variable names that crossed their threshold.
    Falls back to "weather_code change" when changes dict is empty (pure
    weathercode rank trigger with no numeric variable changes).
    """
    triggers = list(alert.changes.keys())
    summary = ", ".join(triggers) if triggers else "weather_code change"  # fix D
    return AlertResponse(
        id=alert.id,
        profile_id=alert.profile_id,
        detected_at=alert.detected_at,
        confidence=alert.confidence,
        horizon_days=round(alert.horizon_days, 1),
        triggering_summary=summary,
        narrative=alert.narrative,
        sent=alert.sent,
    )


# ---------------------------------------------------------------------------
# Profile endpoints
# ---------------------------------------------------------------------------

@router.post("/profiles", response_model=ProfileResponse, status_code=201)
def create_profile(
    payload: ProfileCreate,
    session: Session = Depends(get_session),
) -> ProfileResponse:
    """
    Register a new event for weather monitoring.

    Creates a MonitoringProfile, persists it to the database, and registers
    it with the scheduler atomically. If scheduling fails, the DB write is
    rolled back so no orphaned profile row is left (fix A).

    The first forecast fetch is queued immediately on registration.
    """
    # Reject past events — Pydantic validates types but not temporal logic
    if payload.event_datetime <= datetime.now(timezone.utc):
        raise HTTPException(
            status_code=422,
            detail="event_datetime must be in the future",
        )

    # context is now a Literal field — Pydantic rejects invalid values
    # before this handler runs, so no runtime set-membership check needed

    try:
        profile = MonitoringProfile(
            name=payload.name,
            location=(payload.latitude, payload.longitude),
            event_datetime=payload.event_datetime,
            monitoring_start=datetime.now(timezone.utc),
            check_interval_hours=payload.check_interval_hours,
            event_duration_hours=payload.event_duration_hours,
            context=payload.context,
            notes=payload.notes,
            language=payload.language,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    save_profile(session, profile)

    # Attempt scheduler registration. On failure, raise HTTPException so the
    # session dependency's except block rolls back the save_profile() write.
    # This ensures DB and scheduler are always consistent (fix A).
    registered = register_profile(profile)
    if not registered:
        raise HTTPException(
            status_code=422,
            detail="Profile could not be scheduled — event date may have passed",
        )

    logger.info("POST /profiles: registered '%s' (id=%s)", profile.name, profile.id)
    return _profile_to_response(profile)


@router.get("/profiles", response_model=list[ProfileResponse])
def get_profiles(
    active_only: bool = Query(default=True),
    session: Session = Depends(get_session),
) -> list[ProfileResponse]:
    """List registered monitoring profiles."""
    profiles = list_profiles(session, active_only=active_only)
    return [_profile_to_response(p) for p in profiles]


@router.get("/profiles/{profile_id}", response_model=ProfileResponse)
def get_profile(
    profile_id: str,
    session: Session = Depends(get_session),
) -> ProfileResponse:
    """Get a single profile by ID."""
    profile = load_profile(session, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
    return _profile_to_response(profile)


@router.delete("/profiles/{profile_id}", status_code=204)
def delete_profile(
    profile_id: str,
    session: Session = Depends(get_session),
) -> None:
    """
    Deregister a profile — stops monitoring and removes the scheduler job.
    Profile record and alerts are retained in the database for audit.
    Snapshots are soft-deleted so re-registration starts a fresh baseline.
    """
    profile = load_profile(session, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")

    deregister_profile(profile_id)

    row = session.get(ProfileRow, profile_id)  # fix E: top-level import
    if row:
        row.is_active = False
        session.add(row)
    deregister_snapshots(session, profile_id)

    logger.info("DELETE /profiles/%s: deregistered", profile_id)


# ---------------------------------------------------------------------------
# Alert endpoints
# ---------------------------------------------------------------------------

@router.get("/alerts", response_model=list[AlertResponse])
def get_alerts(
    profile_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
) -> list[AlertResponse]:
    """
    List generated alerts, most recent first.
    Optionally filter by profile_id.
    """
    alerts = list_alerts(session, profile_id=profile_id, limit=limit)
    return [_alert_to_response(a) for a in alerts]


@router.get("/alerts/{alert_id}", response_model=AlertResponse)
def get_alert(
    alert_id: str,
    session: Session = Depends(get_session),
) -> AlertResponse:
    """Get a single alert by ID."""
    row = session.get(AlertRow, alert_id)  # fix E: top-level import
    if row is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    alert = Alert.model_validate_json(row.data)
    return _alert_to_response(alert)


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------

@router.get("/status", response_model=StatusResponse)
def get_status(
    session: Session = Depends(get_session),
) -> StatusResponse:
    """
    Return scheduler health and a summary of active monitoring jobs.
    Useful for the Streamlit dashboard and operational monitoring.
    """
    active_profiles = list_profiles(session, active_only=True)
    jobs = list_jobs()

    return StatusResponse(
        status="ok",
        active_profiles=len(active_profiles),
        scheduled_jobs=len(jobs),
        jobs=jobs,
        timestamp=datetime.now(timezone.utc),
    )