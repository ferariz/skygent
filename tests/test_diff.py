"""
tests/test_diff.py — Unit tests for DiffAnalyzer and compute_delta
===================================================================

Test philosophy
---------------
- Every test is deterministic and requires zero external I/O (no LLM,
  no HTTP, no database).
- Fixtures are defined at the top so test bodies stay readable.
- We test the pure function `compute_delta` independently of the class
  to validate the math before testing higher-level behaviour.
- Edge cases covered: missing variables, None values from sparse API
  responses, zero base values, categorical exclusion, identity guards.

Changes from v1
---------------
- Removed duplicate `from datetime import ...` inside test methods;
  all datetime imports are at the top level.
- Removed `test_weathercode_not_in_categorical_numeric_diff`: that test
  asserted a module constant, not observable behaviour. The behaviour it
  cares about (weathercode absent from compare() output) is already covered
  by `test_returns_dict_with_numeric_variables_only`.
- Added `TestDiffAnalyzerNoneValues` for Open-Meteo sparse-location responses
  where a variable is present as a key but its value is None.
- Added `test_profile_arg_does_not_guard_snapshot_profile_id` to document
  the known gap: compare() checks snapshot-to-snapshot profile_id parity
  but does not verify that the snapshots match the *profile argument* passed
  in. This is intentional — the scheduler always passes the correct profile —
  but the gap is documented so a future reviewer does not add a redundant guard.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from skygent.core.diff import DiffAnalyzer, compute_delta, CATEGORICAL_VARIABLES
from skygent.core.models import ForecastSnapshot, MonitoringProfile


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_profile(**overrides) -> MonitoringProfile:
    """Return a minimal MonitoringProfile, with optional field overrides."""
    defaults = dict(
        name="Test Event",
        location=(40.0, -3.0),
        event_datetime=datetime(2025, 9, 1, 16, 0, tzinfo=timezone.utc),
        monitoring_start=datetime(2025, 8, 1, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return MonitoringProfile(**defaults)


def make_snapshot(profile_id: str, data: dict, horizon_days: float = 5.0) -> ForecastSnapshot:
    """Return a ForecastSnapshot targeting 2025-09-01 with the given data."""
    return ForecastSnapshot(
        profile_id=profile_id,
        fetched_at=datetime(2025, 8, 27, 0, 0, tzinfo=timezone.utc),
        target_datetime=datetime(2025, 9, 1, 16, 0, tzinfo=timezone.utc),
        data=data,
        horizon_days=horizon_days,
    )


PROFILE = make_profile()

PREVIOUS_DATA = {
    "precipitation_probability_max": 10.0,
    "temperature_2m_max": 28.0,
    "wind_speed_10m_max": 20.0,
    "weather_code": 1,
}

CURRENT_DATA = {
    "precipitation_probability_max": 55.0,  # +45 pp — above threshold
    "temperature_2m_max": 30.0,             # +2 °C  — below threshold
    "wind_speed_10m_max": 40.0,              # +20 km/h — above threshold
    "weather_code": 63,                      # categorical — excluded from numeric diff
}


# ---------------------------------------------------------------------------
# compute_delta — pure function
# ---------------------------------------------------------------------------

class TestComputeDelta:
    def test_positive_delta(self):
        vd = compute_delta("temperature_2m_max", 20.0, 26.0)
        assert vd.delta == pytest.approx(6.0)
        assert vd.delta_pct == pytest.approx(30.0)

    def test_negative_delta(self):
        vd = compute_delta("temperature_2m_max", 26.0, 20.0)
        assert vd.delta == pytest.approx(-6.0)
        assert vd.delta_pct == pytest.approx(-23.076923, rel=1e-4)

    def test_zero_delta(self):
        vd = compute_delta("temperature_2m_max", 25.0, 25.0)
        assert vd.delta == pytest.approx(0.0)
        assert vd.delta_pct == pytest.approx(0.0)

    def test_zero_base_value_sets_delta_pct_to_none(self):
        """delta_pct must be None when from_value == 0 to avoid division by zero."""
        vd = compute_delta("precipitation_probability_max", 0.0, 40.0)
        assert vd.delta == pytest.approx(40.0)
        assert vd.delta_pct is None

    def test_all_fields_populated(self):
        vd = compute_delta("wind_speed_10m_max", 10.0, 25.0)
        assert vd.variable == "wind_speed_10m_max"
        assert vd.from_value == pytest.approx(10.0)
        assert vd.to_value == pytest.approx(25.0)
        assert vd.delta == pytest.approx(15.0)
        assert vd.delta_pct == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# DiffAnalyzer.compare — happy path
# ---------------------------------------------------------------------------

class TestDiffAnalyzerCompare:
    def setup_method(self):
        self.analyzer = DiffAnalyzer()
        self.previous = make_snapshot(PROFILE.id, PREVIOUS_DATA)
        self.current = make_snapshot(PROFILE.id, CURRENT_DATA)

    def test_weather_code_excluded_numeric_variables_included(self):
        """
        weather_code is categorical and must never appear in the output.
        All other profile variables must be present.
        """
        changes = self.analyzer.compare(self.previous, self.current, PROFILE)
        assert "weather_code" not in changes
        assert "precipitation_probability_max" in changes
        assert "temperature_2m_max" in changes
        assert "wind_speed_10m_max" in changes

    def test_precipitation_delta_values(self):
        changes = self.analyzer.compare(self.previous, self.current, PROFILE)
        precip = changes["precipitation_probability_max"]
        assert precip["from_value"] == pytest.approx(10.0)
        assert precip["to_value"] == pytest.approx(55.0)
        assert precip["delta"] == pytest.approx(45.0)
        assert precip["delta_pct"] == pytest.approx(450.0)

    def test_temperature_delta_value(self):
        changes = self.analyzer.compare(self.previous, self.current, PROFILE)
        assert changes["temperature_2m_max"]["delta"] == pytest.approx(2.0)

    def test_wind_delta_value(self):
        changes = self.analyzer.compare(self.previous, self.current, PROFILE)
        assert changes["wind_speed_10m_max"]["delta"] == pytest.approx(20.0)

    def test_identical_snapshots_produce_zero_deltas(self):
        """No change between snapshots → every delta is zero."""
        data = {
            "precipitation_probability_max": 20.0,
            "temperature_2m_max": 25.0,
            "wind_speed_10m_max": 15.0,
        }
        prev = make_snapshot(PROFILE.id, data)
        curr = make_snapshot(PROFILE.id, data.copy())
        changes = self.analyzer.compare(prev, curr, PROFILE)
        for variable, entry in changes.items():
            assert entry["delta"] == pytest.approx(0.0), f"Non-zero delta for {variable}"

    def test_no_thresholds_in_profile_still_produces_changes(self):
        """
        compare() returns deltas for all numeric variables regardless of
        whether a threshold is set. Threshold filtering is significance.py's job.
        """
        profile_no_thresholds = make_profile(thresholds={})
        prev = make_snapshot(profile_no_thresholds.id, PREVIOUS_DATA)
        curr = make_snapshot(profile_no_thresholds.id, CURRENT_DATA)
        changes = self.analyzer.compare(prev, curr, profile_no_thresholds)
        assert "precipitation_probability_max" in changes
        assert "temperature_2m_max" in changes
        assert "wind_speed_10m_max" in changes


# ---------------------------------------------------------------------------
# DiffAnalyzer.compare — missing and None variable values
# ---------------------------------------------------------------------------

class TestDiffAnalyzerMissingAndNoneValues:
    """
    Open-Meteo can produce two distinct absence patterns:
      1. The key is entirely absent from the response dict.
      2. The key is present but its value is None (sparse location).
    Both must be handled gracefully — skipped with a log warning, not raised.
    """

    def setup_method(self):
        self.analyzer = DiffAnalyzer()

    def test_key_absent_from_previous_snapshot_is_skipped(self):
        prev = make_snapshot(PROFILE.id, {
            "temperature_2m_max": 28.0,
            "wind_speed_10m_max": 20.0,
            # precipitation_probability_max intentionally absent
        })
        curr = make_snapshot(PROFILE.id, {
            "precipitation_probability_max": 50.0,
            "temperature_2m_max": 30.0,
            "wind_speed_10m_max": 25.0,
        })
        changes = self.analyzer.compare(prev, curr, PROFILE)
        assert "precipitation_probability_max" not in changes
        assert "temperature_2m_max" in changes

    def test_key_absent_from_current_snapshot_is_skipped(self):
        prev = make_snapshot(PROFILE.id, {
            "precipitation_probability_max": 10.0,
            "temperature_2m_max": 28.0,
            "wind_speed_10m_max": 20.0,
        })
        curr = make_snapshot(PROFILE.id, {
            "precipitation_probability_max": 50.0,
            # temperature_2m_max intentionally absent
            "wind_speed_10m_max": 25.0,
        })
        changes = self.analyzer.compare(prev, curr, PROFILE)
        assert "temperature_2m_max" not in changes
        assert "precipitation_probability_max" in changes

    def test_none_value_in_previous_snapshot_is_skipped(self):
        """
        Open-Meteo returns null (→ None) for some variables in sparse regions.
        A None value in the previous snapshot must be treated the same as a
        missing key — skipped, not used as 0.0 in arithmetic.
        """
        prev = make_snapshot(PROFILE.id, {
            "precipitation_probability_max": None,  # sparse-location null
            "temperature_2m_max": 28.0,
            "wind_speed_10m_max": 20.0,
        })
        curr = make_snapshot(PROFILE.id, {
            "precipitation_probability_max": 40.0,
            "temperature_2m_max": 30.0,
            "wind_speed_10m_max": 25.0,
        })
        changes = self.analyzer.compare(prev, curr, PROFILE)
        assert "precipitation_probability_max" not in changes
        assert "temperature_2m_max" in changes

    def test_none_value_in_current_snapshot_is_skipped(self):
        """None in the current snapshot must also be skipped."""
        prev = make_snapshot(PROFILE.id, {
            "precipitation_probability_max": 10.0,
            "temperature_2m_max": 28.0,
            "wind_speed_10m_max": 20.0,
        })
        curr = make_snapshot(PROFILE.id, {
            "precipitation_probability_max": None,  # sparse-location null
            "temperature_2m_max": 30.0,
            "wind_speed_10m_max": 25.0,
        })
        changes = self.analyzer.compare(prev, curr, PROFILE)
        assert "precipitation_probability_max" not in changes
        assert "temperature_2m_max" in changes


# ---------------------------------------------------------------------------
# DiffAnalyzer._assert_compatible — identity guards
# ---------------------------------------------------------------------------

class TestDiffAnalyzerIdentityGuards:
    """
    Guards against the silent bug class where the caller passes snapshots
    from different profiles or different target dates. Both cases produce
    numerically plausible but meaningless deltas that would fire false alerts.
    We raise ValueError rather than warn so the bug surfaces immediately.
    """

    def setup_method(self):
        self.analyzer = DiffAnalyzer()

    def test_different_profile_ids_raise(self):
        other_profile = make_profile(name="Other Event")
        prev = make_snapshot(PROFILE.id, PREVIOUS_DATA)
        curr = make_snapshot(other_profile.id, CURRENT_DATA)
        with pytest.raises(ValueError, match="different profiles"):
            self.analyzer.compare(prev, curr, PROFILE)

    def test_different_target_dates_raise(self):
        prev = make_snapshot(PROFILE.id, PREVIOUS_DATA)
        curr = ForecastSnapshot(
            profile_id=PROFILE.id,
            fetched_at=datetime(2025, 8, 27, 0, 0, tzinfo=timezone.utc),
            target_datetime=datetime(2025, 9, 2, 16, 0, tzinfo=timezone.utc),  # day after
            data=CURRENT_DATA,
            horizon_days=6.0,
        )
        with pytest.raises(ValueError, match="different dates"):
            self.analyzer.compare(prev, curr, PROFILE)

    def test_compatible_snapshots_do_not_raise(self):
        """Baseline: correct inputs must never raise."""
        prev = make_snapshot(PROFILE.id, PREVIOUS_DATA)
        curr = make_snapshot(PROFILE.id, CURRENT_DATA)
        self.analyzer.compare(prev, curr, PROFILE)  # must not raise

    def test_same_date_different_fetch_times_are_compatible(self):
        """
        Snapshots fetched hours apart but targeting the same calendar date
        are compatible. We compare .date(), not the full datetime, to avoid
        rejecting snapshots fetched at e.g. 06:00 vs 18:00 UTC.
        """
        prev = ForecastSnapshot(
            profile_id=PROFILE.id,
            fetched_at=datetime(2025, 8, 27, 0, 0, tzinfo=timezone.utc),
            target_datetime=datetime(2025, 9, 1, 6, 0, tzinfo=timezone.utc),
            data=PREVIOUS_DATA,
            horizon_days=5.25,
        )
        curr = ForecastSnapshot(
            profile_id=PROFILE.id,
            fetched_at=datetime(2025, 8, 27, 6, 0, tzinfo=timezone.utc),
            target_datetime=datetime(2025, 9, 1, 16, 0, tzinfo=timezone.utc),
            data=CURRENT_DATA,
            horizon_days=5.0,
        )
        self.analyzer.compare(prev, curr, PROFILE)  # must not raise

    def test_profile_arg_does_not_guard_snapshot_profile_id(self):
        """
        Known gap (intentional): compare() checks that the two snapshots share
        a profile_id, but it does NOT verify that they match the `profile`
        argument. The scheduler always passes the correct profile, so this
        is not a runtime risk — but we document it here so a future reviewer
        does not add a redundant or overly strict guard.

        If this behaviour changes, update this test and the docstring in diff.py.
        """
        unrelated_profile = make_profile(name="Unrelated Event")
        prev = make_snapshot(PROFILE.id, PREVIOUS_DATA)
        curr = make_snapshot(PROFILE.id, CURRENT_DATA)
        # Both snapshots share a profile_id → no raise, even though
        # `unrelated_profile` is a different object.
        self.analyzer.compare(prev, curr, unrelated_profile)