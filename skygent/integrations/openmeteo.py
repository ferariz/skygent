"""
skygent/integrations/openmeteo.py — Open-Meteo API client
==========================================================

Design decisions
----------------
1. httpx over requests: httpx supports both sync and async with an identical
   API. We expose an async fetch function so the LangGraph nodes can await it
   without blocking the event loop. A sync wrapper is provided for scripts
   and tests that run outside an async context.

2. timezone=UTC always: Gemini's tip to use timezone=auto is correct for
   display purposes, but wrong for our use case. We compare daily aggregates
   (max temperature, max wind, precipitation probability) across snapshots
   fetched at different wall-clock times. If the API returned local-time date
   strings, our date-matching logic would need to know the local UTC offset
   to find the right row, and horizon_days computation would vary with DST.
   UTC date strings are stable, unambiguous, and consistent across all
   profiles regardless of location. A Montevideo event and a Tokyo event
   both get UTC date strings; the diff engine treats them identically.

3. Daily endpoint, not hourly: MonitoringProfile.variables references daily
   aggregates (temperature_2m_max, wind_speed_10m_max, etc). The daily
   endpoint returns exactly these — one value per variable per date — which
   maps cleanly to ForecastSnapshot.data. Hourly would require an aggregation
   step we do not need.

4. We fetch a 16-day window and extract the target date's row: Open-Meteo
   returns a list of dates with parallel value arrays. We request enough
   days to always include the event date, then find the row by date match.
   This is more reliable than requesting a single date because Open-Meteo
   sometimes shifts the available window by ±1 day at the edges.

5. horizon_days is computed here and stored on the snapshot: it is derived
   from (target_date - fetched_at.date()) at fetch time. Storing it avoids
   recomputation and makes confidence scoring deterministic on replay.

6. HTTP errors raise OpenMeteoError (not raw httpx exceptions): callers
   (agent nodes, tests) should not need to import httpx to handle fetch
   failures. A single domain exception is easier to catch and log.

7. Retries are NOT implemented here: retry logic belongs at the scheduler
   level (APScheduler job error handling) or in an httpx Transport, not
   embedded in the fetch function. Adding retries here would make the
   function harder to test and would hide transient failures from the
   agent's state machine.

Timezone note for Montevideo (Gemini's feedback)
-------------------------------------------------
Open-Meteo daily aggregates (max temperature, precipitation probability max,
etc.) are computed over the 24-hour UTC day, not the local day. For a
Montevideo event at 16:00 local time (19:00 UTC), the UTC-day aggregate is
correct: it covers the full calendar day the event falls on. We never compare
hourly timestamps, so the UTC-3 offset is irrelevant to our diff logic.
The narrator node receives horizon_days and the profile's notes field — if
the user wants local-time framing in the alert message, that belongs there.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import httpx

from skygent.core.models import ForecastSnapshot, MonitoringProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.open-meteo.com/v1/forecast"

# How many forecast days to request. 16 is Open-Meteo's maximum and ensures
# we can always reach an event date up to ~15 days out.
FORECAST_DAYS = 16

# Request timeout in seconds. Open-Meteo is fast (<200 ms typical) but we
# allow 10 s to absorb occasional latency spikes.
REQUEST_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Domain exception
# ---------------------------------------------------------------------------

class OpenMeteoError(Exception):
    """
    Raised when the Open-Meteo API returns an error response or when the
    response cannot be parsed into a ForecastSnapshot.

    Wraps the original exception as __cause__ when applicable so callers
    can inspect the root cause if needed.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_params(profile: MonitoringProfile, target_date: date) -> dict:
    """
    Build the Open-Meteo query parameter dict for a given profile and date.

    Parameters
    ----------
    profile:     MonitoringProfile supplying location and variables
    target_date: the event date we want forecast data for

    Returns
    -------
    Dict suitable for passing as `params` to httpx.get().
    """
    lat, lon = profile.location

    # profile.variables contains the exact Open-Meteo daily variable names
    # (e.g. wind_speed_10m_max, weather_code). They are joined and passed
    # verbatim — no translation needed here since models.py owns the defaults.
    daily_vars = ",".join(profile.variables)

    # Open-Meteo treats forecast_days as mutually exclusive with start_date/end_date.
    # We use start_date + end_date so we can target a specific event date precisely.
    # end_date is capped at FORECAST_DAYS - 1 days out to stay within API limits.
    today = datetime.now(timezone.utc).date()
    end_date = min(target_date, today + timedelta(days=FORECAST_DAYS - 1))

    return {
        "latitude":   lat,
        "longitude":  lon,
        "daily":      daily_vars,
        "timezone":   "UTC",   # always UTC — see module docstring
        "start_date": today.isoformat(),
        "end_date":   end_date.isoformat(),
    }


def _extract_target_row(
    response_json: dict,
    target_date: date,
    profile: MonitoringProfile,
) -> dict[str, float | int | None]:
    """
    Extract the forecast values for a single target date from the API response.

    Open-Meteo returns parallel arrays: response["daily"]["time"] is a list
    of ISO date strings, and each variable is a list of values at the same
    index. We find the index of target_date and extract the corresponding
    value from each variable array.

    Parameters
    ----------
    response_json: parsed JSON from Open-Meteo
    target_date:   the date we want values for
    profile:       used to know which variables to extract

    Returns
    -------
    dict[str, float | int | None] — one entry per profile variable.
    None for any variable not present in the response (sparse location).

    Raises
    ------
    OpenMeteoError if the target date is not found in the response.
    """
    daily = response_json.get("daily", {})
    time_list: list[str] = daily.get("time", [])
    target_str = target_date.isoformat()  # e.g. "2025-09-01"

    try:
        idx = time_list.index(target_str)
    except ValueError:
        raise OpenMeteoError(
            f"Target date {target_str} not found in Open-Meteo response. "
            f"Available dates: {time_list[:3]}...{time_list[-3:] if len(time_list) > 3 else ''}"
        )

    data: dict[str, float | int | None] = {}
    for variable in profile.variables:
        values = daily.get(variable)
        if values is None:
            logger.warning(
                "Variable '%s' not in Open-Meteo response for profile '%s'",
                variable, profile.name,
            )
            data[variable] = None
        else:
            data[variable] = values[idx]  # may be None for sparse locations

    return data


def _compute_horizon_days(fetched_at: datetime, target_date: date) -> float:
    """
    Compute horizon_days from fetch time to target date.

    We use UTC date for fetched_at to stay consistent with the UTC date
    strings returned by Open-Meteo. The result is a float (days + fraction)
    to preserve sub-day precision for snapshots fetched mid-day.

    Parameters
    ----------
    fetched_at:  UTC-aware datetime of the API call
    target_date: the event date being forecast

    Returns
    -------
    Non-negative float. 0.0 if the event is today (same UTC date).
    """
    fetched_date = fetched_at.astimezone(timezone.utc).date()
    delta_days = (target_date - fetched_date).days

    # Add intra-day fraction: how far through today we are (0.0 at midnight,
    # 0.5 at noon, 1.0 at next midnight). This makes horizon_days more precise
    # for confidence scoring when events are 3 or 7 days out exactly.
    utc_now = fetched_at.astimezone(timezone.utc)
    day_fraction = (utc_now.hour * 3600 + utc_now.minute * 60 + utc_now.second) / 86400.0

    return max(0.0, delta_days - day_fraction)


# ---------------------------------------------------------------------------
# Public async fetch function
# ---------------------------------------------------------------------------

async def fetch_forecast(
    profile: MonitoringProfile,
    *,
    client: httpx.AsyncClient | None = None,
) -> ForecastSnapshot:
    """
    Fetch a daily forecast snapshot for a MonitoringProfile's event date.

    This is the primary entry point for the LangGraph fetch_forecast node.

    Parameters
    ----------
    profile: MonitoringProfile supplying location, variables, and event date
    client:  optional httpx.AsyncClient to use (for testing / connection reuse).
             If None, a one-shot client is created for this call.

    Returns
    -------
    ForecastSnapshot with data populated for the event date.

    Raises
    ------
    OpenMeteoError on HTTP errors or unparseable responses.
    """
    target_date = profile.event_datetime.date()
    fetched_at = datetime.now(timezone.utc)
    params = _build_params(profile, target_date)

    logger.info(
        "Fetching forecast for '%s' at %s for %s (horizon ~%.1f days)",
        profile.name,
        profile.location,
        target_date,
        _compute_horizon_days(fetched_at, target_date),
    )

    async def _do_fetch(c: httpx.AsyncClient) -> dict:
        try:
            response = await c.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise OpenMeteoError(
                f"Open-Meteo returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise OpenMeteoError(
                f"Network error fetching forecast: {exc}"
            ) from exc

    if client is not None:
        response_json = await _do_fetch(client)
    else:
        async with httpx.AsyncClient() as c:
            response_json = await _do_fetch(c)

    data = _extract_target_row(response_json, target_date, profile)
    horizon_days = _compute_horizon_days(fetched_at, target_date)

    snapshot = ForecastSnapshot(
        profile_id=profile.id,
        fetched_at=fetched_at,
        target_datetime=profile.event_datetime,
        data=data,
        horizon_days=horizon_days,
    )

    logger.info(
        "Snapshot %s created: horizon=%.2f days, variables=%s",
        snapshot.id,
        snapshot.horizon_days,
        list(data.keys()),
    )

    return snapshot


# ---------------------------------------------------------------------------
# Sync convenience wrapper
# ---------------------------------------------------------------------------

def fetch_forecast_sync(
    profile: MonitoringProfile,
) -> ForecastSnapshot:
    """
    Synchronous wrapper around fetch_forecast for use in scripts, CLIs,
    and non-async test contexts.

    Do NOT use this inside an already-running event loop (e.g. inside a
    LangGraph node) — use the async fetch_forecast instead.
    """
    import asyncio
    return asyncio.run(fetch_forecast(profile))