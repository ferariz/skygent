"""
skygent/api/database.py — SQLModel database layer
===================================================

Design decisions
----------------
1. SQLModel over raw SQLAlchemy: SQLModel combines Pydantic v2 validation
   with SQLAlchemy ORM in one class hierarchy. Domain models (Pydantic) and
   DB models (SQLModel) share field definitions rather than duplicating them.

2. Separate DB models from domain models: MonitoringProfile, ForecastSnapshot,
   and Alert in models.py are pure Pydantic. The DB models here (ProfileRow,
   SnapshotRow, AlertRow) store serialized versions. This keeps the agent,
   diff, and significance layers independent of the database.

3. JSON blob for complex fields: full objects are stored as JSON in a `data`
   TEXT column so MonitoringProfile schema changes do not require DB migrations
   during early development. Only the fields needed for queries/filtering are
   stored as indexed columns.

4. SQLite for MVP, PostgreSQL for production: SQLite requires no server.
   The connection string is the only change needed to switch to PostgreSQL.

5. get_session() as a FastAPI dependency: yields a session per request,
   commits on success, rolls back on error. The scheduler uses
   get_session_sync() (a context manager) outside the request cycle.

6. DBSnapshotStore replaces the in-memory SnapshotStore from jobs.py via
   the same get/set/clear interface so the scheduler requires no changes.

Fixes applied after Cursor review (v2)
---------------------------------------
A. is_active staleness: list_profiles(active_only=True) now filters by
   event_datetime > now() at query time rather than trusting a stored boolean.
   The boolean column is kept for explicit deregistration (DELETE /profiles)
   but is no longer the sole gate for active_only queries.

B. DBSnapshotStore.clear() behavioral parity: rather than a no-op, we now
   mark the profile's snapshots as deregistered by setting a deregistered_at
   timestamp on a dedicated column. load_latest_snapshot() skips deregistered
   snapshots so a re-registered profile starts fresh. Audit history is kept.

C. AlertRow denormalized columns removed: confidence, horizon_days, sent,
   narrative, and the dead significant field are dropped from the table.
   The only source of truth is data (JSON). Reads go through
   Alert.model_validate_json(row.data) consistently — no drift possible.
   Indexed columns retained: id, profile_id, detected_at (for sorting/filtering).

D. Unused imports removed: json and Optional were not used.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from sqlmodel import Field, Session, SQLModel, create_engine, select

from skygent.core.models import (
    Alert,
    ForecastSnapshot,
    MonitoringProfile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database engine
# ---------------------------------------------------------------------------

DATABASE_URL = "sqlite:///skygent.db"

engine = create_engine(
    DATABASE_URL,
    echo=False,                    # set True to log SQL for debugging
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
)


def create_db_and_tables() -> None:
    """Create all tables. Called once at application startup."""
    SQLModel.metadata.create_all(engine)
    logger.info("database: tables created/verified")


# ---------------------------------------------------------------------------
# DB table models
# ---------------------------------------------------------------------------

class ProfileRow(SQLModel, table=True):
    """
    Persisted MonitoringProfile.

    is_active is set to False on explicit deregistration (DELETE /profiles).
    Active queries use event_datetime > now() as the primary filter so
    expired profiles are excluded even if is_active was never flipped.
    """
    __tablename__ = "profiles"

    id: str = Field(primary_key=True)
    name: str
    event_datetime: datetime          # indexed for active_only filtering
    is_active: bool = True            # flipped on explicit deregistration
    data: str                         # JSON-serialized MonitoringProfile


class SnapshotRow(SQLModel, table=True):
    """
    Persisted ForecastSnapshot.

    deregistered_at is set when a profile is deregistered. Snapshots with
    a non-null deregistered_at are excluded from diff baseline lookups so
    a re-registered profile starts fresh while retaining audit history.
    """
    __tablename__ = "snapshots"

    id: str = Field(primary_key=True)
    profile_id: str = Field(index=True)
    fetched_at: datetime
    horizon_days: float
    deregistered_at: datetime | None = Field(default=None)  # fix B
    data: str                         # JSON-serialized ForecastSnapshot


class AlertRow(SQLModel, table=True):
    """
    Persisted Alert. Only columns needed for filtering/sorting are stored
    separately from the JSON blob — everything else is read from data.
    Denormalized columns (confidence, horizon_days, sent, narrative) removed
    to eliminate drift between columns and the JSON payload (fix C, D).
    """
    __tablename__ = "alerts"

    id: str = Field(primary_key=True)
    profile_id: str = Field(index=True)
    detected_at: datetime             # for ORDER BY DESC
    data: str                         # JSON-serialized Alert (single source of truth)


# ---------------------------------------------------------------------------
# Session dependency (FastAPI)
# ---------------------------------------------------------------------------

def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session per request.
    Commits on success, rolls back on exception, always closes.
    """
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


@contextmanager
def get_session_sync() -> Generator[Session, None, None]:
    """
    Synchronous context manager for use outside the FastAPI request cycle
    (e.g. scheduler jobs, startup hooks).
    """
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def save_profile(session: Session, profile: MonitoringProfile) -> ProfileRow:
    """Insert or replace a MonitoringProfile in the database."""
    row = ProfileRow(
        id=profile.id,
        name=profile.name,
        event_datetime=profile.event_datetime,
        is_active=True,               # always True on save; flip via deregister
        data=profile.model_dump_json(),
    )
    session.merge(row)
    return row


def load_profile(session: Session, profile_id: str) -> MonitoringProfile | None:
    """Load a MonitoringProfile by ID, or None if not found."""
    row = session.get(ProfileRow, profile_id)
    if row is None:
        return None
    return MonitoringProfile.model_validate_json(row.data)


def list_profiles(session: Session, active_only: bool = True) -> list[MonitoringProfile]:
    """
    Return profiles, optionally filtering to currently active ones.

    active_only=True applies two filters:
    1. is_active=True  — excludes explicitly deregistered profiles
    2. event_datetime > now()  — excludes profiles whose event has passed,
       even if is_active was never explicitly flipped (fix A)
    """
    stmt = select(ProfileRow)
    if active_only:
        now = datetime.now(timezone.utc)
        stmt = stmt.where(
            ProfileRow.is_active == True,          # noqa: E712
            ProfileRow.event_datetime > now,
        )
    rows = session.exec(stmt).all()
    return [MonitoringProfile.model_validate_json(r.data) for r in rows]


def save_snapshot(session: Session, snapshot: ForecastSnapshot) -> SnapshotRow:
    """Insert or replace a ForecastSnapshot."""
    row = SnapshotRow(
        id=snapshot.id,
        profile_id=snapshot.profile_id,
        fetched_at=snapshot.fetched_at,
        horizon_days=snapshot.horizon_days,
        deregistered_at=None,
        data=snapshot.model_dump_json(),
    )
    session.merge(row)
    return row


def load_latest_snapshot(
    session: Session, profile_id: str
) -> ForecastSnapshot | None:
    """
    Load the most recently fetched non-deregistered snapshot for a profile.

    Snapshots with deregistered_at set are skipped so a re-registered profile
    starts a fresh diff baseline rather than diffing against stale history.
    """
    stmt = (
        select(SnapshotRow)
        .where(
            SnapshotRow.profile_id == profile_id,
            SnapshotRow.deregistered_at == None,   # noqa: E711
        )
        .order_by(SnapshotRow.fetched_at.desc())
        .limit(1)
    )
    row = session.exec(stmt).first()
    if row is None:
        return None
    return ForecastSnapshot.model_validate_json(row.data)


def deregister_snapshots(session: Session, profile_id: str) -> None:
    """
    Mark all active snapshots for a profile as deregistered.

    Called when a profile is deregistered so that if it is re-registered
    later, load_latest_snapshot() returns None and the agent starts fresh.
    Snapshots are retained for audit — not deleted.
    """
    now = datetime.now(timezone.utc)
    rows = session.exec(
        select(SnapshotRow).where(
            SnapshotRow.profile_id == profile_id,
            SnapshotRow.deregistered_at == None,   # noqa: E711
        )
    ).all()
    for row in rows:
        row.deregistered_at = now
        session.add(row)
    logger.info(
        "database: marked %d snapshot(s) as deregistered for profile %s",
        len(rows), profile_id,
    )


def save_alert(session: Session, alert: Alert) -> AlertRow:
    """Insert or replace an Alert. Only id, profile_id, detected_at, and
    the full JSON payload are stored — no denormalized columns."""
    row = AlertRow(
        id=alert.id,
        profile_id=alert.profile_id,
        detected_at=alert.detected_at,
        data=alert.model_dump_json(),
    )
    session.merge(row)
    return row


def list_alerts(
    session: Session,
    profile_id: str | None = None,
    limit: int = 50,
) -> list[Alert]:
    """
    Return recent alerts, most recent first.
    Optionally filtered by profile_id.
    """
    stmt = select(AlertRow).order_by(AlertRow.detected_at.desc()).limit(limit)
    if profile_id:
        stmt = stmt.where(AlertRow.profile_id == profile_id)
    rows = session.exec(stmt).all()
    return [Alert.model_validate_json(r.data) for r in rows]


# ---------------------------------------------------------------------------
# DB-backed SnapshotStore (replaces in-memory store from jobs.py)
# ---------------------------------------------------------------------------

class DBSnapshotStore:
    """
    DB-backed implementation of the SnapshotStore interface.

    Implements the same get/set/clear API as the in-memory SnapshotStore
    in jobs.py so the scheduler requires no changes — only the store
    instance is swapped at application startup.
    """

    def get(self, profile_id: str) -> ForecastSnapshot | None:
        with get_session_sync() as session:
            return load_latest_snapshot(session, profile_id)

    def set(self, snapshot: ForecastSnapshot) -> None:
        with get_session_sync() as session:
            save_snapshot(session, snapshot)
            logger.debug(
                "DBSnapshotStore: saved snapshot %s for profile %s",
                snapshot.id, snapshot.profile_id,
            )

    def clear(self, profile_id: str) -> None:
        """
        Mark all active snapshots as deregistered so a re-registered profile
        starts a fresh diff baseline. Audit history is retained (fix B).
        """
        with get_session_sync() as session:
            deregister_snapshots(session, profile_id)

    @property
    def profile_ids(self) -> list[str]:
        with get_session_sync() as session:
            rows = session.exec(select(SnapshotRow.profile_id).distinct()).all()
            return list(rows)