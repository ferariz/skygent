"""
tests/test_openmeteo.py — Unit and integration tests for openmeteo.py
======================================================================

Test philosophy
---------------
- Unit tests mock httpx entirely — zero network calls, fully deterministic.
- One integration test (TestLiveOpenMeteo) makes a real HTTP call.
  It is skipped by default and activated with: pytest -m integration
  This keeps CI fast while allowing manual verification against the real API.
- We test the internal helpers (_build_params, _extract_target_row,
  _compute_horizon_days) as pure functions, independent of httpx, so the
  logic can be validated without any async plumbing.

Test structure
--------------
TestBuildParams          — query parameter construction
TestComputeHorizonDays   — horizon_days float computation and edge cases
TestExtractTargetRow     — date-row lookup, missing variable, absent date
TestFetchForecast        — async fetch with mocked httpx client
TestLiveOpenMeteo        — real API call (integration, skipped by default)
"""

from __future__ import annotations

import json
import pytest
from datetime import date, datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from skygent.core.models import ForecastSnapshot, MonitoringProfile
from skygent.integrations.openmeteo import (
    BASE_URL,
    FORECAST_DAYS,
    OpenMeteoError,
    _build_params,
    _compute_horizon_days,
    _extract_target_row,
    fetch_forecast,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_profile(**overrides) -> MonitoringProfile:
    defaults = dict(
        name="Test Wedding",
        location=(-34.9011, -56.1645),   # Montevideo, Uruguay
        event_datetime=datetime(2025, 9, 1, 16, 0, tzinfo=timezone.utc),
        monitoring_start=datetime(2025, 8, 1, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return MonitoringProfile(**defaults)


PROFILE = make_profile()
EVENT_DATE = PROFILE.event_datetime.date()  # date(2025, 9, 1)


def make_api_response(target_date: date, overrides: dict | None = None) -> dict:
    """
    Build a minimal but realistic Open-Meteo daily response containing
    exactly the variables in PROFILE.variables for the given target_date.
    Includes one extra day before and after to test index selection.
    """
    prev_date = target_date - timedelta(days=1)
    next_date = target_date + timedelta(days=1)

    base = {
        "latitude": -34.9,
        "longitude": -56.16,
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "timezone_abbreviation": "UTC",
        "daily": {
            "time": [
                prev_date.isoformat(),
                target_date.isoformat(),
                next_date.isoformat(),
            ],
            "precipitation_probability_max": [5.0, 20.0, 35.0],
            "temperature_2m_max":            [22.0, 25.0, 27.0],
            "wind_speed_10m_max":             [12.0, 18.0, 22.0],
            "weather_code":                   [1, 3, 61],
        }
    }
    if overrides:
        base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestBuildParams
# ---------------------------------------------------------------------------

class TestBuildParams:
    def test_latitude_longitude_correct(self):
        params = _build_params(PROFILE, EVENT_DATE)
        assert params["latitude"] == -34.9011
        assert params["longitude"] == -56.1645

    def test_timezone_is_always_utc(self):
        """We always request UTC regardless of profile location."""
        params = _build_params(PROFILE, EVENT_DATE)
        assert params["timezone"] == "UTC"

    def test_daily_variables_match_profile(self):
        params = _build_params(PROFILE, EVENT_DATE)
        requested = set(params["daily"].split(","))
        expected = set(PROFILE.variables)
        assert requested == expected

    def test_forecast_days_not_in_params(self):
        """
        forecast_days is mutually exclusive with start_date/end_date on the
        Open-Meteo API. We use start_date + end_date, so forecast_days must
        never appear in the outgoing request params.
        """
        params = _build_params(PROFILE, EVENT_DATE)
        assert "forecast_days" not in params

    def test_start_and_end_date_present(self):
        """start_date and end_date must both be present instead of forecast_days."""
        params = _build_params(PROFILE, EVENT_DATE)
        assert "start_date" in params
        assert "end_date" in params

    def test_start_date_is_today_utc(self):
        params = _build_params(PROFILE, EVENT_DATE)
        today = datetime.now(timezone.utc).date().isoformat()
        assert params["start_date"] == today

    def test_custom_variables_included(self):
        profile = make_profile(
            variables=["temperature_2m_max", "weather_code"],
            thresholds={"temperature_2m_max": 4.0},
        )
        params = _build_params(profile, EVENT_DATE)
        assert "temperature_2m_max" in params["daily"]
        assert "weather_code" in params["daily"]


# ---------------------------------------------------------------------------
# TestComputeHorizonDays
# ---------------------------------------------------------------------------

class TestComputeHorizonDays:
    def test_same_utc_date_is_less_than_one(self):
        """Event today: horizon < 1.0 (fraction of the day already elapsed)."""
        fetched_at = datetime(2025, 9, 1, 12, 0, tzinfo=timezone.utc)  # noon
        horizon = _compute_horizon_days(fetched_at, date(2025, 9, 1))
        assert 0.0 <= horizon < 1.0

    def test_five_days_out_at_midnight(self):
        """Fetched at UTC midnight, event in 5 days → horizon ≈ 5.0."""
        fetched_at = datetime(2025, 8, 27, 0, 0, tzinfo=timezone.utc)
        horizon = _compute_horizon_days(fetched_at, date(2025, 9, 1))
        assert horizon == pytest.approx(5.0, abs=0.01)

    def test_five_days_out_at_noon(self):
        """Fetched at noon UTC, event in 5 days → horizon ≈ 4.5."""
        fetched_at = datetime(2025, 8, 27, 12, 0, tzinfo=timezone.utc)
        horizon = _compute_horizon_days(fetched_at, date(2025, 9, 1))
        assert horizon == pytest.approx(4.5, abs=0.01)

    def test_never_negative(self):
        """Even if called after the event date, horizon is clamped to 0."""
        fetched_at = datetime(2025, 9, 5, 0, 0, tzinfo=timezone.utc)
        horizon = _compute_horizon_days(fetched_at, date(2025, 9, 1))
        assert horizon == 0.0

    def test_aware_non_utc_input_converted(self):
        """
        Non-UTC aware datetimes are converted to UTC before computing.
        Montevideo is UTC-3: 09:00 local = 12:00 UTC = 0.5 day fraction.
        """
        from datetime import timezone as tz
        montevideo = tz(timedelta(hours=-3))
        fetched_at = datetime(2025, 8, 27, 9, 0, tzinfo=montevideo)  # = 12:00 UTC
        horizon = _compute_horizon_days(fetched_at, date(2025, 9, 1))
        assert horizon == pytest.approx(4.5, abs=0.01)


# ---------------------------------------------------------------------------
# TestExtractTargetRow
# ---------------------------------------------------------------------------

class TestExtractTargetRow:
    def test_extracts_correct_index(self):
        """Values at index 1 (target_date) are returned, not index 0 or 2."""
        resp = make_api_response(EVENT_DATE)
        data = _extract_target_row(resp, EVENT_DATE, PROFILE)
        assert data["precipitation_probability_max"] == pytest.approx(20.0)
        assert data["temperature_2m_max"] == pytest.approx(25.0)
        assert data["wind_speed_10m_max"] == pytest.approx(18.0)
        assert data["weather_code"] == 3

    def test_all_profile_variables_present(self):
        resp = make_api_response(EVENT_DATE)
        data = _extract_target_row(resp, EVENT_DATE, PROFILE)
        for variable in PROFILE.variables:
            assert variable in data

    def test_missing_variable_in_response_returns_none(self):
        """
        If a variable is absent from the API response (sparse location),
        its value should be None rather than raising KeyError.
        """
        resp = make_api_response(EVENT_DATE)
        del resp["daily"]["wind_speed_10m_max"]
        data = _extract_target_row(resp, EVENT_DATE, PROFILE)
        assert data["wind_speed_10m_max"] is None
        # Other variables still extracted correctly
        assert data["temperature_2m_max"] == pytest.approx(25.0)

    def test_none_value_in_response_preserved(self):
        """
        Open-Meteo returns null for sparse locations. None values in the
        array must be preserved — not converted to 0.0 or skipped.
        """
        resp = make_api_response(EVENT_DATE)
        resp["daily"]["precipitation_probability_max"] = [5.0, None, 35.0]
        data = _extract_target_row(resp, EVENT_DATE, PROFILE)
        assert data["precipitation_probability_max"] is None

    def test_target_date_not_in_response_raises(self):
        """If the target date is not in the time list, raise OpenMeteoError."""
        resp = make_api_response(EVENT_DATE)
        wrong_date = date(2025, 12, 25)
        with pytest.raises(OpenMeteoError, match="not found in Open-Meteo response"):
            _extract_target_row(resp, wrong_date, PROFILE)

    def test_first_date_in_list_extracted_correctly(self):
        """Edge: target is the first element (index 0), not middle or last."""
        prev_date = EVENT_DATE - timedelta(days=1)
        resp = make_api_response(EVENT_DATE)
        data = _extract_target_row(resp, prev_date, PROFILE)
        assert data["temperature_2m_max"] == pytest.approx(22.0)  # index 0


# ---------------------------------------------------------------------------
# TestFetchForecast — async with mocked httpx
# ---------------------------------------------------------------------------

def make_mock_client(response_json: dict, status_code: int = 200) -> MagicMock:
    """
    Build a mock httpx.AsyncClient whose get() returns a fake response.
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = response_json
    mock_response.raise_for_status = MagicMock(
        side_effect=None if status_code < 400 else Exception("HTTP Error")
    )

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.mark.asyncio
class TestFetchForecast:
    async def test_returns_forecast_snapshot(self):
        """Happy path: valid response → ForecastSnapshot instance."""
        resp = make_api_response(EVENT_DATE)
        client = make_mock_client(resp)
        snapshot = await fetch_forecast(PROFILE, client=client)
        assert isinstance(snapshot, ForecastSnapshot)

    async def test_snapshot_profile_id_matches(self):
        resp = make_api_response(EVENT_DATE)
        client = make_mock_client(resp)
        snapshot = await fetch_forecast(PROFILE, client=client)
        assert snapshot.profile_id == PROFILE.id

    async def test_snapshot_target_datetime_matches_event(self):
        resp = make_api_response(EVENT_DATE)
        client = make_mock_client(resp)
        snapshot = await fetch_forecast(PROFILE, client=client)
        assert snapshot.target_datetime == PROFILE.event_datetime

    async def test_snapshot_data_contains_all_variables(self):
        resp = make_api_response(EVENT_DATE)
        client = make_mock_client(resp)
        snapshot = await fetch_forecast(PROFILE, client=client)
        for variable in PROFILE.variables:
            assert variable in snapshot.data

    async def test_snapshot_horizon_days_non_negative(self):
        resp = make_api_response(EVENT_DATE)
        client = make_mock_client(resp)
        snapshot = await fetch_forecast(PROFILE, client=client)
        assert snapshot.horizon_days >= 0.0

    async def test_fetched_at_is_utc_aware(self):
        resp = make_api_response(EVENT_DATE)
        client = make_mock_client(resp)
        snapshot = await fetch_forecast(PROFILE, client=client)
        assert snapshot.fetched_at.tzinfo is not None
        assert snapshot.fetched_at.tzinfo == timezone.utc

    async def test_correct_values_extracted(self):
        """Values in snapshot.data must match the target row, not adjacent rows."""
        resp = make_api_response(EVENT_DATE)
        client = make_mock_client(resp)
        snapshot = await fetch_forecast(PROFILE, client=client)
        assert snapshot.data["precipitation_probability_max"] == pytest.approx(20.0)
        assert snapshot.data["temperature_2m_max"] == pytest.approx(25.0)
        assert snapshot.data["weather_code"] == 3

    async def test_http_error_raises_openmeteo_error(self):
        """HTTP 4xx/5xx responses must raise OpenMeteoError, not httpx internals."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        import httpx
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_response
        )
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(OpenMeteoError):
            await fetch_forecast(PROFILE, client=mock_client)

    async def test_network_error_raises_openmeteo_error(self):
        """Network-level failures (DNS, timeout) must raise OpenMeteoError."""
        import httpx
        mock_client = MagicMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        with pytest.raises(OpenMeteoError):
            await fetch_forecast(PROFILE, client=mock_client)

    async def test_correct_url_called(self):
        """The fetch must hit BASE_URL, not a typo or different endpoint."""
        resp = make_api_response(EVENT_DATE)
        client = make_mock_client(resp)
        await fetch_forecast(PROFILE, client=client)
        call_args = client.get.call_args
        assert call_args[0][0] == BASE_URL

    async def test_utc_timezone_in_request_params(self):
        """timezone=UTC must always be in the outgoing request params."""
        resp = make_api_response(EVENT_DATE)
        client = make_mock_client(resp)
        await fetch_forecast(PROFILE, client=client)
        call_kwargs = client.get.call_args[1]
        assert call_kwargs["params"]["timezone"] == "UTC"


# ---------------------------------------------------------------------------
# TestLiveOpenMeteo — real API call (integration only)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLiveOpenMeteo:
    """
    Makes a real HTTP call to Open-Meteo. Skipped in normal test runs.

    Run with:  pytest -m integration tests/test_openmeteo.py -v

    These tests verify that our parameter construction and response parsing
    are correct against the actual API — not just against our mock.
    """

    @pytest.mark.asyncio
    async def test_live_fetch_returns_snapshot(self):
        """Live fetch for Montevideo returns a valid ForecastSnapshot."""
        # Use an event date 7 days out to ensure it's within the forecast window
        future_event = datetime.now(timezone.utc) + timedelta(days=7)
        future_event = future_event.replace(hour=16, minute=0, second=0, microsecond=0)

        profile = make_profile(
            event_datetime=future_event,
            monitoring_start=datetime.now(timezone.utc),
        )
        snapshot = await fetch_forecast(profile)

        assert isinstance(snapshot, ForecastSnapshot)
        assert snapshot.profile_id == profile.id
        assert snapshot.horizon_days > 0.0
        assert snapshot.fetched_at.tzinfo == timezone.utc

        # All variables should be populated (Montevideo has full coverage)
        for variable in profile.variables:
            assert variable in snapshot.data, f"Missing variable: {variable}"

        print(f"\nLive snapshot data: {snapshot.data}")
        print(f"Horizon: {snapshot.horizon_days:.2f} days")