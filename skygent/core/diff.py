"""
skygent/core/diff.py — Forecast diff engine
=============================================

Design decisions
----------------
1. Pure function core, class wrapper for convenience: the heavy lifting is
   done by `compute_delta` (a pure function) so it is trivially unit-testable.
   `DiffAnalyzer` is a thin stateless class that makes the call-site clean.

2. Absolute delta AND percentage delta are both stored: percentage change
   alone is misleading near zero (1%→3% precipitation = 200% relative change
   but is meteorological noise). The significance evaluator uses absolute
   delta; percentage is stored for display.

3. weathercode excluded from numeric diff: WMO codes are categorical.
   Arithmetic delta is meaningless (code 3→61 looks like +58 but is a major
   change; 60→61 looks like +1 but is trivial). Handled via severity rank
   in significance.py.

4. Missing variables are skipped, not errored: Open-Meteo occasionally omits
   variables for sparse locations. A KeyError would crash the agent run.

5. No floating-point rounding in the delta dict: rounding would silently
   swallow small real changes. The significance evaluator applies its own
   threshold — that is the right place for that judgment.

6. horizon_days comes from the current snapshot: we are asking "how reliable
   is this new forecast?", not "how reliable was the old one?"

Fixes applied after Cursor review (v2)
---------------------------------------
A. Return type updated to dict[str, VariableChange] — matches the TypedDict
   defined in models.py; gives IDE autocompletion on payload keys.

B. Docstring updated: payload shape now documents from_value/to_value and
   explicitly notes delta_pct can be None.

C. Identity guards added via _assert_compatible(): raises ValueError when
   snapshots belong to different profiles or target different dates. Comparing
   mismatched snapshots is always a caller bug and produces meaningless deltas.
   Implemented as a hard raise (not a warning) because silently diffing the
   wrong pair would produce alerts for non-existent changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from skygent.core.models import ForecastSnapshot, MonitoringProfile, VariableChange

logger = logging.getLogger(__name__)

# Variables that use categorical codes and must not be diff'd numerically.
CATEGORICAL_VARIABLES = {"weathercode"}


@dataclass
class VariableDelta:
    """
    Internal computation result for a single variable's change.

    This is an intermediate dataclass used only inside compute_delta and
    DiffAnalyzer.compare. It is converted to a VariableChange dict before
    being returned to callers — keeping the public API as plain dicts avoids
    forcing callers to import an extra type.

    Attributes
    ----------
    variable:   Open-Meteo variable name
    from_value: value in the previous snapshot (native units)
    to_value:   value in the current snapshot (native units)
    delta:      absolute change (to_value - from_value)
    delta_pct:  relative change as a percentage; None when from_value == 0
    """
    variable: str
    from_value: float
    to_value: float
    delta: float
    delta_pct: float | None


def compute_delta(variable: str, from_value: float, to_value: float) -> VariableDelta:
    """
    Compute the absolute and relative change for one variable.

    Parameters
    ----------
    variable:   variable name (used only for the returned dataclass label)
    from_value: previous forecast value (native Open-Meteo units)
    to_value:   current forecast value (native Open-Meteo units)

    Returns
    -------
    VariableDelta with:
      - delta     = to_value - from_value  (signed, native units)
      - delta_pct = (delta / |from_value|) * 100  or  None if from_value == 0
    """
    delta = to_value - from_value
    delta_pct: float | None = None

    if from_value != 0:
        delta_pct = (delta / abs(from_value)) * 100.0

    return VariableDelta(
        variable=variable,
        from_value=from_value,
        to_value=to_value,
        delta=delta,
        delta_pct=delta_pct,
    )


class DiffAnalyzer:
    """
    Compares two ForecastSnapshot objects and returns a dict of deltas
    suitable for populating Alert.changes.

    Usage
    -----
    analyzer = DiffAnalyzer()
    changes = analyzer.compare(previous_snapshot, current_snapshot, profile)
    """

    def compare(
        self,
        previous: ForecastSnapshot,
        current: ForecastSnapshot,
        profile: MonitoringProfile,
    ) -> dict[str, VariableChange]:
        """
        Compute deltas for all numeric variables defined in the profile.

        Parameters
        ----------
        previous: earlier ForecastSnapshot (baseline)
        current:  newer ForecastSnapshot (what we just fetched)
        profile:  the MonitoringProfile driving this run (determines which
                  variables to compare)

        Returns
        -------
        Dict keyed by variable name. Each value is a VariableChange with:
            {
                "from_value": <previous value in native units>,
                "to_value":   <current value in native units>,
                "delta":      <signed absolute change>,
                "delta_pct":  <relative change %> or None if from_value == 0,
            }
        Only variables present in BOTH snapshots are included.
        Categorical variables (weathercode) are always skipped — they are
        evaluated via severity rank in significance.py.

        Raises
        ------
        ValueError
            If the two snapshots belong to different profiles or target
            different event dates. Diffing mismatched snapshots is always
            a caller bug and would produce meaningless deltas.
        """
        self._assert_compatible(previous, current)

        changes: dict[str, VariableChange] = {}

        numeric_variables = [
            v for v in profile.variables
            if v not in CATEGORICAL_VARIABLES
        ]

        for variable in numeric_variables:
            prev_val = previous.data.get(variable)
            curr_val = current.data.get(variable)

            if prev_val is None:
                logger.warning(
                    "Variable '%s' missing from previous snapshot %s — skipping",
                    variable, previous.id,
                )
                continue

            if curr_val is None:
                logger.warning(
                    "Variable '%s' missing from current snapshot %s — skipping",
                    variable, current.id,
                )
                continue

            vd = compute_delta(variable, float(prev_val), float(curr_val))

            changes[variable] = VariableChange(
                from_value=vd.from_value,
                to_value=vd.to_value,
                delta=vd.delta,
                delta_pct=vd.delta_pct,  # None when from_value == 0
            )

        self._log_weathercode_change(previous, current)

        return changes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_compatible(
        self,
        previous: ForecastSnapshot,
        current: ForecastSnapshot,
    ) -> None:
        """
        Raise ValueError if the two snapshots should not be compared.

        Two snapshots are incompatible when:
        - They belong to different profiles: the variables and location
          differ, so deltas are meaningless.
        - They target different dates: we would be comparing, e.g.,
          a Saturday forecast against a Sunday forecast for the same
          location — also meaningless.

        We use .date() for the target_datetime comparison so that
        snapshots fetched at slightly different times of day (e.g.
        06:00 UTC vs 06:05 UTC) for the same event date still compare
        correctly.
        """
        if previous.profile_id != current.profile_id:
            raise ValueError(
                f"Cannot compare snapshots from different profiles: "
                f"{previous.profile_id!r} vs {current.profile_id!r}"
            )

        if previous.target_datetime.date() != current.target_datetime.date():
            raise ValueError(
                f"Cannot compare snapshots targeting different dates: "
                f"{previous.target_datetime.date()} vs {current.target_datetime.date()}"
            )

    def _log_weathercode_change(
        self,
        previous: ForecastSnapshot,
        current: ForecastSnapshot,
    ) -> None:
        """
        Log a weathercode transition without attempting numeric diff.
        The significance evaluator handles weathercode via severity ranks.
        """
        prev_code = previous.data.get("weathercode")
        curr_code = current.data.get("weathercode")

        if prev_code is not None and curr_code is not None:
            if prev_code != curr_code:
                logger.info(
                    "Weathercode changed: %s → %s (snapshots %s → %s)",
                    prev_code, curr_code, previous.id, current.id,
                )