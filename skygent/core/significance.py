"""
skygent/core/significance.py — Significance evaluator
======================================================

Design decisions
----------------
1. Deterministic, LLM-free: threshold decisions are rules, not reasoning
   tasks. This makes the evaluator fast, free, and fully unit-testable.

2. Confidence derived here, not by the narrator: horizon_days → confidence
   is meteorological domain knowledge. Computing it here and passing the
   label to the LLM prevents inconsistent narrator output and token waste.

3. Weathercode severity ranking: WMO codes are not linear. We maintain a
   rank table (lower = better) so a rank 1 → rank 4 jump is "significant"
   regardless of the numeric code difference.

4. is_significant returns (bool, list[str]): the trigger list lets callers
   log which variables fired and give the narrator targeted context without
   re-running the logic.

5. Weathercode check is additive: numeric and categorical checks both run;
   we do not short-circuit. A profile can trigger on both simultaneously.

6. Thresholds come from MonitoringProfile, not hardcoded here: keeps the
   evaluator vertical-agnostic across social, agriculture, and energy uses.

Fixes applied after Cursor review (v2)
---------------------------------------
A. Usage docstring corrected: the class docstring showed
   `is_significant(changes, profile, snapshot)` but the actual signature
   is `(changes, profile, current_snapshot, previous_snapshot)`.

B. changes type tightened: both is_significant and build_alert now accept
   `dict[str, VariableChange]` instead of the loose `dict[str, dict[str, float]]`.
   This matches the TypedDict defined in models.py and gives callers IDE
   autocompletion on payload keys.

C. Weathercode direction — explicit product decision: previously only
   worsening forecasts (positive rank_delta) triggered an alert. We now
   also alert on significant improvement (negative rank_delta whose absolute
   value meets the threshold). Rationale: a user who planned around a
   thunderstorm forecast and received an "actually it's clear now" alert
   has just as much reason to act (uncancel the outdoor ceremony, adjust
   logistics) as one receiving bad news. The trigger variable name is still
   "weathercode" regardless of direction; the narrator receives the full
   change detail and can frame the message accordingly.

D. Negative horizon guard: horizon_to_confidence previously accepted
   negative values silently, returning "high" (because -1.0 <= 3.0).
   Negative horizon means the event is in the past — the scheduler guards
   against this via MonitoringProfile.is_active, but a defensive ValueError
   here surfaces any bug in that guard immediately rather than issuing a
   misleading "high confidence" label for a past event.
"""

from __future__ import annotations

import logging
from typing import Literal

from skygent.core.models import Alert, ForecastSnapshot, MonitoringProfile, VariableChange

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

# Meteorological rule of thumb for NWP (Numerical Weather Prediction) models:
#   ≤ 3 days  → deterministic models are reliable         → high
#   3–7 days  → ensemble spread grows, uncertainty rises  → medium
#   > 7 days  → beyond reliable NWP horizon               → low
ConfidenceLabel = Literal["high", "medium", "low"]

CONFIDENCE_THRESHOLDS: list[tuple[float, ConfidenceLabel]] = [
    (3.0, "high"),
    (7.0, "medium"),
]


def horizon_to_confidence(horizon_days: float) -> ConfidenceLabel:
    """
    Map forecast horizon to a categorical confidence label.

    Parameters
    ----------
    horizon_days: days between forecast fetch time and event datetime.
                  Must be >= 0; negative values indicate a past event
                  and are rejected with ValueError.

    Returns
    -------
    "high"   if horizon_days <= 3
    "medium" if 3 < horizon_days <= 7
    "low"    if horizon_days > 7

    Raises
    ------
    ValueError if horizon_days < 0. The scheduler must not call this for
    events that have already passed (MonitoringProfile.is_active guards
    against this, but we fail loudly rather than silently return "high"
    for a past event).
    """
    if horizon_days < 0:
        raise ValueError(
            f"horizon_days must be >= 0, got {horizon_days:.3f}. "
            "Negative horizon means the event is in the past — "
            "check MonitoringProfile.is_active before scheduling a run."
        )
    for threshold, label in CONFIDENCE_THRESHOLDS:
        if horizon_days <= threshold:
            return label
    return "low"


# ---------------------------------------------------------------------------
# WMO weathercode severity ranking
# ---------------------------------------------------------------------------

# WMO code → severity rank (higher rank = worse conditions).
# Not all codes are listed; unlisted codes default to rank 0.
# Source: https://open-meteo.com/en/docs (WMO Weather interpretation codes)
WMO_SEVERITY_RANK: dict[int, int] = {
    0: 0,   # Clear sky
    1: 0,   # Mainly clear
    2: 1,   # Partly cloudy
    3: 1,   # Overcast
    45: 2,  # Fog
    48: 2,  # Icy fog
    51: 2,  # Light drizzle
    53: 2,  # Moderate drizzle
    55: 3,  # Dense drizzle
    61: 3,  # Slight rain
    63: 3,  # Moderate rain
    65: 4,  # Heavy rain
    71: 3,  # Slight snow
    73: 3,  # Moderate snow
    75: 4,  # Heavy snow
    80: 3,  # Slight showers
    81: 4,  # Moderate showers
    82: 5,  # Violent showers
    85: 4,  # Slight snow showers
    86: 5,  # Heavy snow showers
    95: 5,  # Thunderstorm
    96: 5,  # Thunderstorm with slight hail
    99: 6,  # Thunderstorm with heavy hail
}

# A weathercode change triggers an alert when the absolute severity rank
# difference meets or exceeds this value — in either direction.
# Default = 2 covers both worsening (clear → rain) and improving
# (thunderstorm → overcast) forecast changes worth acting on.
DEFAULT_WEATHERCODE_RANK_THRESHOLD = 2


def weathercode_rank(code: int) -> int:
    """Return the severity rank for a WMO weather code (0 if unknown)."""
    return WMO_SEVERITY_RANK.get(int(code), 0)


# ---------------------------------------------------------------------------
# Main significance evaluator
# ---------------------------------------------------------------------------

class SignificanceEvaluator:
    """
    Determines whether a set of forecast changes warrants an alert,
    and computes the confidence level for that alert.

    Usage
    -----
    evaluator = SignificanceEvaluator()

    # Check whether a diff is significant
    significant, triggers = evaluator.is_significant(
        changes=diff_output,          # dict[str, VariableChange]
        profile=monitoring_profile,
        current_snapshot=curr,
        previous_snapshot=prev,
    )

    # Build a ready-to-narrate Alert (narrative filled in by the LLM node)
    if significant:
        alert = evaluator.build_alert(profile, prev, curr, changes)
    """

    def __init__(
        self,
        weathercode_rank_threshold: int = DEFAULT_WEATHERCODE_RANK_THRESHOLD,
    ) -> None:
        self.weathercode_rank_threshold = weathercode_rank_threshold

    def is_significant(
        self,
        changes: dict[str, VariableChange],
        profile: MonitoringProfile,
        current_snapshot: ForecastSnapshot,
        previous_snapshot: ForecastSnapshot,
    ) -> tuple[bool, list[str]]:
        """
        Evaluate whether forecast changes meet significance thresholds.

        Parameters
        ----------
        changes:           output of DiffAnalyzer.compare() — one entry per
                           numeric variable that was diffed
        profile:           MonitoringProfile supplying per-variable thresholds
        current_snapshot:  newer snapshot; supplies current weathercode
        previous_snapshot: older snapshot; supplies previous weathercode

        Returns
        -------
        (significant, triggering_variables)

        significant: True if any variable crossed its threshold or if the
                     weathercode rank change (in either direction) met the
                     rank threshold.

        triggering_variables: list of variable names that fired, including
                     "weathercode" when applicable. Empty list when not
                     significant.

        Notes
        -----
        Weathercode alerts fire on both worsening AND improving forecasts.
        A large improvement (thunderstorm → clear) is equally actionable
        for the user as a large deterioration. The narrator receives the
        full change detail and frames the message accordingly.
        """
        triggers: list[str] = []

        # --- Numeric variable thresholds ---
        for variable, delta_info in changes.items():
            threshold = profile.thresholds.get(variable)
            if threshold is None:
                logger.debug(
                    "No threshold configured for '%s' — skipping",
                    variable,
                )
                continue

            abs_delta = abs(delta_info["delta"])
            if abs_delta >= threshold:
                logger.info(
                    "Threshold crossed: %s Δ=%.2f (threshold %.2f)",
                    variable, abs_delta, threshold,
                )
                triggers.append(variable)

        # --- Weathercode categorical change (both directions) ---
        prev_code = previous_snapshot.data.get("weathercode")
        curr_code = current_snapshot.data.get("weathercode")

        if prev_code is not None and curr_code is not None:
            prev_rank = weathercode_rank(int(prev_code))
            curr_rank = weathercode_rank(int(curr_code))
            rank_delta = curr_rank - prev_rank          # signed
            abs_rank_delta = abs(rank_delta)            # direction-agnostic

            if abs_rank_delta >= self.weathercode_rank_threshold:
                direction = "worsening" if rank_delta > 0 else "improving"
                logger.info(
                    "Weathercode %s: %s (rank %d) → %s (rank %d), |Δrank|=%d",
                    direction,
                    prev_code, prev_rank,
                    curr_code, curr_rank,
                    abs_rank_delta,
                )
                triggers.append("weathercode")

        significant = len(triggers) > 0
        return significant, triggers

    def confidence(self, horizon_days: float) -> ConfidenceLabel:
        """
        Return a confidence label for a given forecast horizon.
        Delegates to the module-level pure function so it can be tested
        independently of the class.

        Raises ValueError for negative horizon_days (see horizon_to_confidence).
        """
        return horizon_to_confidence(horizon_days)

    def build_alert(
        self,
        profile: MonitoringProfile,
        previous: ForecastSnapshot,
        current: ForecastSnapshot,
        changes: dict[str, VariableChange],
    ) -> Alert:
        """
        Build a fully-populated Alert (minus narrative) from two snapshots
        and their diff. The narrator node fills in Alert.narrative before
        delivery.

        Parameters
        ----------
        profile:  MonitoringProfile the alert belongs to
        previous: baseline ForecastSnapshot
        current:  newer ForecastSnapshot; its horizon_days sets confidence
        changes:  output of DiffAnalyzer.compare()

        Returns
        -------
        Alert with narrative="" and sent=False.
        """
        return Alert(
            profile_id=profile.id,
            previous_snapshot_id=previous.id,
            current_snapshot_id=current.id,
            changes=changes,
            horizon_days=current.horizon_days,
            confidence=self.confidence(current.horizon_days),
        )