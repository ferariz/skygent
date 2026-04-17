"""
skygent/core/models.py — Core domain models
=============================================

Design decisions
----------------
1. Pydantic v2 throughout — free JSON serialization, schema generation,
   runtime validation. Used by FastAPI, the agent nodes, and the DB layer.

2. MonitoringProfile is the unit of configuration — all agent behavior
   (variables, thresholds, poll interval) lives here. No global config file.

3. Thresholds are per-variable dicts — vertical-agnostic reuse. Agriculture
   needs different wind sensitivity than a wedding. Defaults are social-tuned.

4. ForecastSnapshot stores raw API response + derived horizon_days — we keep
   the raw dict so no information is lost. horizon_days is stored at fetch
   time because it drives confidence scoring; recomputing it later is fragile.

5. Alert.confidence is a string enum — downstream consumers need a label,
   not a float. The horizon→category mapping lives in significance.py.

6. UUIDs for IDs — no dependency on DB auto-increment; IDs can be
   constructed before a DB write.

Fixes applied after Cursor review (v2)
---------------------------------------
A. Timezone-aware datetimes: all defaults now use datetime.now(timezone.utc)
   instead of the deprecated datetime.utcnow(). Mixed aware/naive comparisons
   are handled defensively (strip tzinfo + warn) rather than raising TypeError.

B. Location validated as real lat/lon: latitude ∈ [-90, 90], longitude ∈
   [-180, 180]. Open-Meteo silently returns wrong data for invalid coordinates.

C. check_interval_hours enforced >= 1 via Field(ge=1). Zero or negative values
   would spin APScheduler into a tight loop or crash at job registration.

D. thresholds vs variables consistency: every key in thresholds must appear in
   variables. A threshold for a never-fetched variable will silently never fire.
   The reverse (variable without threshold) is allowed — weathercode is in
   variables but evaluated categorically, not by numeric threshold.

E. ForecastSnapshot.data typed as dict[str, float | int | None]. Still flexible
   for any Open-Meteo response; None is allowed for sparse locations.

F. Alert.changes typed via VariableChange TypedDict instead of bare dict.
   Gives IDE autocompletion and a clear contract on expected keys.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VariableChange — typed structure for one variable's change record in Alert
# ---------------------------------------------------------------------------

class VariableChange(TypedDict):
    """
    Holds the numeric change for one forecast variable inside an Alert.

    from_value / to_value are in the variable's native Open-Meteo units.
    delta_pct is None when from_value == 0 (avoids division by zero).
    """
    from_value: float
    to_value: float
    delta: float
    delta_pct: float | None


# ---------------------------------------------------------------------------
# MonitoringProfile
# ---------------------------------------------------------------------------

class MonitoringProfile(BaseModel):
    """
    Represents one user-defined event to monitor.

    A profile drives the entire agent loop: the scheduler reads active
    profiles, triggers one graph run per profile, and passes the profile
    into every node so threshold and context decisions are self-contained.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str  # e.g. "Ana & Juan's Wedding"

    # Geographic target — Open-Meteo uses (lat, lon) as floats.
    # Validated: lat ∈ [-90, 90], lon ∈ [-180, 180].
    location: tuple[float, float]

    # The moment the event starts — agent stops monitoring after this.
    # Use timezone-aware datetimes (UTC strongly recommended).
    event_datetime: datetime

    # When to begin monitoring. Defaults to now (UTC, aware).
    monitoring_start: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # How often the scheduler triggers a graph run. Must be >= 1.
    check_interval_hours: int = Field(default=6, ge=1)

    # Open-Meteo daily variables to request and watch.
    variables: list[str] = Field(default_factory=lambda: [
        "precipitation_probability_max",
        "temperature_2m_max",
        "windspeed_10m_max",
        "weathercode",
    ])

    # Per-variable change magnitude that triggers an alert.
    #   precipitation_probability_max → percentage points
    #   temperature_2m_max            → °C
    #   windspeed_10m_max             → km/h
    # weathercode is intentionally absent: evaluated via severity rank
    # in significance.py, not by numeric threshold.
    thresholds: dict[str, float] = Field(default_factory=lambda: {
        "precipitation_probability_max": 20.0,
        "temperature_2m_max": 4.0,
        "windspeed_10m_max": 15.0,
    })

    notification_channel: str = "telegram"

    # Used by narrator node to tailor language. Does NOT affect thresholds.
    context: Literal["social_event", "agriculture", "energy", "logistics"] = "social_event"

    # Optional free-text for the narrator (e.g. "outdoor, no tent backup")
    notes: str = ""

    # --- Field validators ---------------------------------------------------

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Reject coordinates outside real-world lat/lon bounds."""
        lat, lon = v
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitude {lat} out of range [-90, 90]")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitude {lon} out of range [-180, 180]")
        return v

    @field_validator("thresholds")
    @classmethod
    def validate_threshold_values(cls, v: dict[str, float]) -> dict[str, float]:
        """All threshold values must be strictly positive."""
        for variable, value in v.items():
            if value <= 0:
                raise ValueError(
                    f"Threshold for '{variable}' must be > 0, got {value}"
                )
        return v

    # --- Model validators (cross-field) -------------------------------------

    @model_validator(mode="after")
    def monitoring_start_before_event(self) -> "MonitoringProfile":
        """
        Ensure monitoring_start precedes event_datetime.

        Handles mixed aware/naive datetimes defensively: strips tzinfo for
        the comparison and logs a warning. Always pass aware datetimes.
        """
        start = self.monitoring_start
        event = self.event_datetime

        if (start.tzinfo is None) != (event.tzinfo is None):
            logger.warning(
                "Mixed aware/naive datetimes in MonitoringProfile '%s'. "
                "Use timezone-aware datetimes (UTC). Comparing without tzinfo.",
                self.name,
            )
            start = start.replace(tzinfo=None)
            event = event.replace(tzinfo=None)

        if start >= event:
            raise ValueError("monitoring_start must be before event_datetime")
        return self

    @model_validator(mode="after")
    def thresholds_reference_known_variables(self) -> "MonitoringProfile":
        """
        Every key in thresholds must appear in variables.

        A threshold for a variable not in `variables` will never be fetched
        and will silently never fire — always a configuration bug.
        """
        unknown = set(self.thresholds) - set(self.variables)
        if unknown:
            raise ValueError(
                f"Thresholds defined for variables not in `variables`: {unknown}. "
                "Add them to `variables` or remove them from `thresholds`."
            )
        return self

    # --- Properties ---------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True while the event is still in the future."""
        event = self.event_datetime
        if event.tzinfo is None:
            return datetime.utcnow() < event
        return datetime.now(timezone.utc) < event


# ---------------------------------------------------------------------------
# ForecastSnapshot
# ---------------------------------------------------------------------------

class ForecastSnapshot(BaseModel):
    """
    A point-in-time capture of the Open-Meteo forecast for a profile's
    location and event date.

    Snapshots are immutable once written. The diff engine compares two
    snapshots (previous vs current) to detect changes.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str

    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    target_datetime: datetime

    # Raw Open-Meteo response for the target date.
    # None is allowed: Open-Meteo returns null for sparse locations.
    data: dict[str, float | int | None]

    # Precomputed at fetch time — drives confidence scoring.
    horizon_days: float

    @model_validator(mode="after")
    def horizon_must_be_non_negative(self) -> "ForecastSnapshot":
        if self.horizon_days < 0:
            raise ValueError("horizon_days cannot be negative")
        return self


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

class Alert(BaseModel):
    """
    Generated when the diff engine detects a significant forecast change.

    narrative is filled by the LLM narrator node. Everything else is
    populated deterministically before the LLM is called.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str

    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    previous_snapshot_id: str
    current_snapshot_id: str

    # Typed change records — one entry per diffed variable.
    changes: dict[str, VariableChange]

    horizon_days: float

    # Derived by significance.py — not inferred by the LLM.
    confidence: Literal["high", "medium", "low"]

    narrative: str = ""   # filled by narrate node
    sent: bool = False    # flipped by notify node