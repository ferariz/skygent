"""
tests/test_significance.py — Unit tests for SignificanceEvaluator
==================================================================

Test philosophy
---------------
- All tests are deterministic; no LLM or network calls.
- horizon_to_confidence is tested as a pure function first (boundary values
  included), then the class methods, then the full build_alert pipeline.
- Weathercode rank tests use weathercode_rank() lookups rather than
  hardcoded integers so the tests stay correct if the rank table is updated.
- build_alert payload is verified end-to-end: ids, confidence, horizon_days,
  changes passthrough, and default field values.

Changes from v1 (significance.py v2 alignment)
-----------------------------------------------
- test_negative_horizon_raises added to TestHorizonToConfidence: previously
  horizon_to_confidence(-1.0) silently returned "high" because -1.0 <= 3.0.
  The function now raises ValueError for negative inputs; test verifies this.
- test_large_improving_rank_jump_triggers added: v2 significance.py alerts
  on significant improvement (|Δrank| >= threshold) as well as worsening.
  A thunderstorm → clear sky forecast change is equally actionable.
- test_minor_improving_rank_jump_does_not_trigger added: confirms the
  threshold applies symmetrically — small improvements are not noisy alerts.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from skygent.core.significance import (
    SignificanceEvaluator,
    horizon_to_confidence,
    weathercode_rank,
    WMO_SEVERITY_RANK,
)
from skygent.core.models import ForecastSnapshot, MonitoringProfile


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_profile(**overrides) -> MonitoringProfile:
    defaults = dict(
        name="Test Event",
        location=(40.0, -3.0),
        event_datetime=datetime(2025, 9, 1, 16, 0, tzinfo=timezone.utc),
        monitoring_start=datetime(2025, 8, 1, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return MonitoringProfile(**defaults)


def make_snapshot(profile_id: str, data: dict, horizon_days: float) -> ForecastSnapshot:
    return ForecastSnapshot(
        profile_id=profile_id,
        fetched_at=datetime(2025, 8, 27, 0, 0, tzinfo=timezone.utc),
        target_datetime=datetime(2025, 9, 1, 16, 0, tzinfo=timezone.utc),
        data=data,
        horizon_days=horizon_days,
    )


PROFILE = make_profile()


# ---------------------------------------------------------------------------
# horizon_to_confidence — pure function, boundary values
# ---------------------------------------------------------------------------

class TestHorizonToConfidence:
    def test_within_three_days_is_high(self):
        assert horizon_to_confidence(1.0) == "high"
        assert horizon_to_confidence(2.9) == "high"

    def test_exactly_three_days_is_high(self):
        """Boundary: 3.0 is the inclusive upper edge of 'high'."""
        assert horizon_to_confidence(3.0) == "high"

    def test_just_above_three_days_is_medium(self):
        assert horizon_to_confidence(3.1) == "medium"
        assert horizon_to_confidence(5.0) == "medium"

    def test_exactly_seven_days_is_medium(self):
        """Boundary: 7.0 is the inclusive upper edge of 'medium'."""
        assert horizon_to_confidence(7.0) == "medium"

    def test_just_above_seven_days_is_low(self):
        assert horizon_to_confidence(7.1) == "low"
        assert horizon_to_confidence(10.0) == "low"
        assert horizon_to_confidence(30.0) == "low"

    def test_zero_horizon_is_high(self):
        """Edge: event is today — maximum forecast confidence."""
        assert horizon_to_confidence(0.0) == "high"

    def test_negative_horizon_raises(self):
        """
        Negative horizon means the event is in the past. The scheduler guards
        against this via MonitoringProfile.is_active, but horizon_to_confidence
        must not silently return "high" (which -1.0 <= 3.0 would do without
        the guard). A loud ValueError surfaces the upstream scheduling bug.
        """
        with pytest.raises(ValueError, match="horizon_days must be >= 0"):
            horizon_to_confidence(-1.0)

        with pytest.raises(ValueError):
            horizon_to_confidence(-0.001)


# ---------------------------------------------------------------------------
# weathercode_rank — rank table lookups
# ---------------------------------------------------------------------------

class TestWeathercodeRank:
    def test_clear_sky_is_rank_zero(self):
        assert weathercode_rank(0) == 0

    def test_thunderstorm_ranks(self):
        assert weathercode_rank(95) == 5
        assert weathercode_rank(99) == 6

    def test_unknown_code_returns_zero(self):
        """Unknown WMO codes default to rank 0 rather than raising."""
        assert weathercode_rank(999) == 0

    def test_heavy_rain_rank_exceeds_slight_rain(self):
        assert weathercode_rank(65) > weathercode_rank(61)

    def test_all_listed_ranks_are_non_negative(self):
        for code, rank in WMO_SEVERITY_RANK.items():
            assert rank >= 0, f"Negative rank for WMO code {code}"

    def test_rank_ordering_is_monotonically_sensible(self):
        """
        Sanity-check a representative severity ladder:
        clear < drizzle < rain < heavy rain < thunderstorm.
        Uses weathercode_rank() so the test stays valid if values change.
        """
        assert weathercode_rank(0) < weathercode_rank(51)   # clear < light drizzle
        assert weathercode_rank(51) < weathercode_rank(65)  # light drizzle < heavy rain
        assert weathercode_rank(65) < weathercode_rank(95)  # heavy rain < thunderstorm


# ---------------------------------------------------------------------------
# SignificanceEvaluator.is_significant — numeric thresholds
# ---------------------------------------------------------------------------

class TestIsSignificantNumeric:
    def setup_method(self):
        self.evaluator = SignificanceEvaluator()

    def _make_snapshots(self, prev_data, curr_data, horizon=5.0):
        prev = make_snapshot(PROFILE.id, prev_data, horizon)
        curr = make_snapshot(PROFILE.id, curr_data, horizon)
        return prev, curr

    def test_large_precip_change_triggers(self):
        """45 pp change >> 20 pp threshold → significant."""
        changes = {"precipitation_probability_max": {
            "from_value": 10, "to_value": 55, "delta": 45, "delta_pct": 450,
        }}
        prev, curr = self._make_snapshots(
            {"precipitation_probability_max": 10, "weathercode": 1},
            {"precipitation_probability_max": 55, "weathercode": 1},
        )
        significant, triggers = self.evaluator.is_significant(changes, PROFILE, curr, prev)
        assert significant is True
        assert "precipitation_probability_max" in triggers

    def test_small_precip_change_does_not_trigger(self):
        """10 pp change < 20 pp threshold → not significant."""
        changes = {"precipitation_probability_max": {
            "from_value": 10, "to_value": 20, "delta": 10, "delta_pct": 100,
        }}
        prev, curr = self._make_snapshots(
            {"precipitation_probability_max": 10, "weathercode": 1},
            {"precipitation_probability_max": 20, "weathercode": 1},
        )
        significant, triggers = self.evaluator.is_significant(changes, PROFILE, curr, prev)
        assert significant is False
        assert triggers == []

    def test_delta_exactly_at_threshold_triggers(self):
        """delta == threshold satisfies >= threshold → should trigger."""
        changes = {"precipitation_probability_max": {
            "from_value": 10, "to_value": 30, "delta": 20, "delta_pct": 200,
        }}
        prev, curr = self._make_snapshots(
            {"precipitation_probability_max": 10, "weathercode": 1},
            {"precipitation_probability_max": 30, "weathercode": 1},
        )
        significant, triggers = self.evaluator.is_significant(changes, PROFILE, curr, prev)
        assert significant is True
        assert "precipitation_probability_max" in triggers

    def test_negative_delta_uses_absolute_value(self):
        """A drop of 25 pp triggers the same as a rise of 25 pp."""
        changes = {"precipitation_probability_max": {
            "from_value": 60, "to_value": 35, "delta": -25, "delta_pct": -41.6,
        }}
        prev, curr = self._make_snapshots(
            {"precipitation_probability_max": 60, "weathercode": 1},
            {"precipitation_probability_max": 35, "weathercode": 1},
        )
        significant, triggers = self.evaluator.is_significant(changes, PROFILE, curr, prev)
        assert significant is True
        assert "precipitation_probability_max" in triggers

    def test_variable_without_threshold_is_ignored(self):
        """Variables absent from profile.thresholds must never trigger."""
        changes = {"some_unlisted_variable": {
            "from_value": 0, "to_value": 9999, "delta": 9999, "delta_pct": None,
        }}
        prev, curr = self._make_snapshots({"weathercode": 1}, {"weathercode": 1})
        significant, triggers = self.evaluator.is_significant(changes, PROFILE, curr, prev)
        assert significant is False

    def test_multiple_variables_trigger_simultaneously(self):
        changes = {
            "precipitation_probability_max": {
                "from_value": 10, "to_value": 55, "delta": 45, "delta_pct": 450,
            },
            "windspeed_10m_max": {
                "from_value": 15, "to_value": 40, "delta": 25, "delta_pct": 166,
            },
        }
        prev, curr = self._make_snapshots(
            {"precipitation_probability_max": 10, "windspeed_10m_max": 15, "weathercode": 1},
            {"precipitation_probability_max": 55, "windspeed_10m_max": 40, "weathercode": 1},
        )
        significant, triggers = self.evaluator.is_significant(changes, PROFILE, curr, prev)
        assert significant is True
        assert "precipitation_probability_max" in triggers
        assert "windspeed_10m_max" in triggers


# ---------------------------------------------------------------------------
# SignificanceEvaluator.is_significant — weathercode categorical changes
# ---------------------------------------------------------------------------

class TestIsSignificantWeathercode:
    def setup_method(self):
        self.evaluator = SignificanceEvaluator(weathercode_rank_threshold=2)

    def test_large_rank_jump_triggers(self):
        """
        Clear sky → heavy rain: rank jump derived from the actual rank table.
        Using weathercode_rank() instead of magic numbers so the test stays
        valid if WMO_SEVERITY_RANK is updated.
        """
        from_code, to_code = 0, 65  # clear sky → heavy rain
        assert weathercode_rank(to_code) - weathercode_rank(from_code) >= 2

        changes = {}
        prev = make_snapshot(PROFILE.id, {"weathercode": from_code}, horizon_days=5.0)
        curr = make_snapshot(PROFILE.id, {"weathercode": to_code}, horizon_days=5.0)
        significant, triggers = self.evaluator.is_significant(changes, PROFILE, curr, prev)
        assert significant is True
        assert "weathercode" in triggers

    def test_zero_rank_jump_does_not_trigger(self):
        """
        Partly cloudy (code 2, rank 1) → overcast (code 3, rank 1): rank
        jump is 0, which is below every positive threshold. The test uses
        weathercode_rank() to assert the premise before testing behaviour.
        """
        from_code, to_code = 2, 3
        assert weathercode_rank(from_code) == weathercode_rank(to_code), (
            "Test premise broken: codes 2 and 3 are no longer the same rank. "
            "Update the test to use two codes that share a rank."
        )
        changes = {}
        prev = make_snapshot(PROFILE.id, {"weathercode": from_code}, horizon_days=5.0)
        curr = make_snapshot(PROFILE.id, {"weathercode": to_code}, horizon_days=5.0)
        significant, triggers = self.evaluator.is_significant(changes, PROFILE, curr, prev)
        assert significant is False

    def test_same_weathercode_does_not_trigger(self):
        """Identical codes → rank delta 0 → no alert."""
        changes = {}
        prev = make_snapshot(PROFILE.id, {"weathercode": 63}, horizon_days=5.0)
        curr = make_snapshot(PROFILE.id, {"weathercode": 63}, horizon_days=5.0)
        significant, triggers = self.evaluator.is_significant(changes, PROFILE, curr, prev)
        assert significant is False

    def test_custom_rank_threshold_respected(self):
        """With threshold=1, a rank jump of exactly 1 must trigger."""
        evaluator = SignificanceEvaluator(weathercode_rank_threshold=1)
        from_code, to_code = 0, 2  # clear (rank 0) → partly cloudy (rank 1)
        assert weathercode_rank(to_code) - weathercode_rank(from_code) == 1

        changes = {}
        prev = make_snapshot(PROFILE.id, {"weathercode": from_code}, horizon_days=5.0)
        curr = make_snapshot(PROFILE.id, {"weathercode": to_code}, horizon_days=5.0)
        significant, triggers = evaluator.is_significant(changes, PROFILE, curr, prev)
        assert significant is True

    def test_weathercode_absent_from_snapshots_does_not_crash(self):
        """
        If neither snapshot contains a weathercode key, the categorical check
        must be silently skipped — not raise a KeyError or AttributeError.
        """
        changes = {}
        prev = make_snapshot(PROFILE.id, {"temperature_2m_max": 28.0}, horizon_days=5.0)
        curr = make_snapshot(PROFILE.id, {"temperature_2m_max": 30.0}, horizon_days=5.0)
        significant, triggers = self.evaluator.is_significant(changes, PROFILE, curr, prev)
        assert "weathercode" not in triggers

    def test_large_improving_rank_jump_triggers(self):
        """
        Product decision (v2): significant improvement also warrants an alert.
        A user who planned around a thunderstorm forecast and receives
        "actually it's clear now" has equally actionable new information.

        Thunderstorm (code 95) → clear sky (code 0): rank drops by 5,
        |Δrank| = 5 >= threshold 2 → triggers.
        """
        from_code, to_code = 95, 0  # thunderstorm → clear sky
        assert abs(weathercode_rank(to_code) - weathercode_rank(from_code)) >= 2

        changes = {}
        prev = make_snapshot(PROFILE.id, {"weathercode": from_code}, horizon_days=5.0)
        curr = make_snapshot(PROFILE.id, {"weathercode": to_code}, horizon_days=5.0)
        significant, triggers = self.evaluator.is_significant(changes, PROFILE, curr, prev)
        assert significant is True
        assert "weathercode" in triggers

    def test_minor_improving_rank_jump_does_not_trigger(self):
        """
        A small improvement below the rank threshold must not trigger.
        Heavy rain (code 65, rank 4) → moderate rain (code 63, rank 3):
        |Δrank| = 1 < threshold 2 → no alert.
        """
        from_code, to_code = 65, 63  # heavy rain → moderate rain
        assert abs(weathercode_rank(to_code) - weathercode_rank(from_code)) == 1

        changes = {}
        prev = make_snapshot(PROFILE.id, {"weathercode": from_code}, horizon_days=5.0)
        curr = make_snapshot(PROFILE.id, {"weathercode": to_code}, horizon_days=5.0)
        significant, triggers = self.evaluator.is_significant(changes, PROFILE, curr, prev)
        assert significant is False


# ---------------------------------------------------------------------------
# SignificanceEvaluator.build_alert — full payload verification
# ---------------------------------------------------------------------------

class TestBuildAlert:
    def setup_method(self):
        self.evaluator = SignificanceEvaluator()
        self.prev = make_snapshot(PROFILE.id, {"weathercode": 1}, horizon_days=5.0)

    def test_ids_wired_correctly(self):
        curr = make_snapshot(PROFILE.id, {"weathercode": 63}, horizon_days=2.0)
        changes = {"precipitation_probability_max": {
            "from_value": 10, "to_value": 60, "delta": 50, "delta_pct": 500,
        }}
        alert = self.evaluator.build_alert(PROFILE, self.prev, curr, changes)

        assert alert.profile_id == PROFILE.id
        assert alert.previous_snapshot_id == self.prev.id
        assert alert.current_snapshot_id == curr.id

    def test_changes_dict_passed_through_unchanged(self):
        """build_alert must not transform or filter the changes dict."""
        curr = make_snapshot(PROFILE.id, {"weathercode": 63}, horizon_days=2.0)
        changes = {"precipitation_probability_max": {
            "from_value": 10, "to_value": 60, "delta": 50, "delta_pct": 500,
        }}
        alert = self.evaluator.build_alert(PROFILE, self.prev, curr, changes)
        assert alert.changes == changes

    def test_horizon_days_taken_from_current_snapshot(self):
        """horizon_days on the alert must match the current snapshot, not previous."""
        curr = make_snapshot(PROFILE.id, {"weathercode": 63}, horizon_days=2.5)
        alert = self.evaluator.build_alert(PROFILE, self.prev, curr, {})
        assert alert.horizon_days == pytest.approx(2.5)

    def test_confidence_high_for_short_horizon(self):
        curr = make_snapshot(PROFILE.id, {}, horizon_days=2.0)
        alert = self.evaluator.build_alert(PROFILE, self.prev, curr, {})
        assert alert.confidence == "high"

    def test_confidence_medium_for_mid_horizon(self):
        curr = make_snapshot(PROFILE.id, {}, horizon_days=5.0)
        alert = self.evaluator.build_alert(PROFILE, self.prev, curr, {})
        assert alert.confidence == "medium"

    def test_confidence_low_for_long_horizon(self):
        curr = make_snapshot(PROFILE.id, {}, horizon_days=9.0)
        alert = self.evaluator.build_alert(PROFILE, self.prev, curr, {})
        assert alert.confidence == "low"

    def test_narrative_starts_empty(self):
        """narrative must be '' — the narrator node fills it in later."""
        curr = make_snapshot(PROFILE.id, {}, horizon_days=5.0)
        alert = self.evaluator.build_alert(PROFILE, self.prev, curr, {})
        assert alert.narrative == ""

    def test_sent_starts_false(self):
        """sent must be False — the notify node flips it after delivery."""
        curr = make_snapshot(PROFILE.id, {}, horizon_days=5.0)
        alert = self.evaluator.build_alert(PROFILE, self.prev, curr, {})
        assert alert.sent is False