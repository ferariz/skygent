"""
skygent/agent/state.py — Agent state definition
=================================================

Design decisions
----------------
1. AgentState is a plain TypedDict, not a Pydantic model: LangGraph 1.x
   expects state to be a dict-like object. TypedDict gives us type hints
   and IDE support without the overhead of Pydantic validation on every
   state transition. Domain objects (MonitoringProfile, ForecastSnapshot,
   Alert) are already validated by Pydantic when constructed — we do not
   need to re-validate them as they flow through the graph.

2. All fields are Optional with None defaults: LangGraph merges partial
   state updates returned by each node. A node only needs to return the
   fields it changed — unmentioned fields are left as-is. This means every
   field must have a default so the initial state can be constructed with
   only a profile.

3. The profile field is the only required input: every graph run starts
   with a MonitoringProfile. Everything else — snapshots, diff output,
   alert — is produced by the nodes and accumulated into state.

4. error carries a human-readable message when any node fails: the graph
   uses a conditional edge after each node to short-circuit to END on
   error rather than propagating exceptions through the graph. The
   scheduler reads this field to decide whether to log/retry.

5. significant is a plain bool set by the evaluate_significance node:
   the conditional edge after that node reads it to decide whether to
   call the narrate node or exit early. Most runs will exit here.
"""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict

from skygent.core.models import Alert, ForecastSnapshot, MonitoringProfile
from skygent.core.models import VariableChange


class AgentState(TypedDict, total=False):
    """
    State that flows through the Skygent LangGraph agent.

    Fields are populated incrementally by each node:

    fetch_forecast      → sets current_snapshot
    analyze_diff        → sets changes
    evaluate_significance → sets significant, triggering_variables, alert (skeleton)
    narrate             → sets alert.narrative (fills in the narrative field)
    notify              → sets alert.sent = True

    error is set by any node that fails; the graph short-circuits to END.
    """

    # ── Input (required to start a run) ─────────────────────────────────────
    profile: MonitoringProfile

    # ── Populated by fetch_forecast ─────────────────────────────────────────
    # The snapshot from the previous run, loaded from the DB before fetching.
    # None on the very first run for a profile.
    previous_snapshot: Optional[ForecastSnapshot]

    # The freshly fetched snapshot for this run.
    current_snapshot: Optional[ForecastSnapshot]

    # ── Populated by analyze_diff ────────────────────────────────────────────
    changes: Optional[dict[str, VariableChange]]

    # ── Populated by evaluate_significance ──────────────────────────────────
    significant: Optional[bool]
    triggering_variables: Optional[list[str]]

    # The alert skeleton — confidence, changes, snapshot IDs are set here.
    # narrative is filled in by the narrate node.
    alert: Optional[Alert]

    # ── Set by any node on failure ───────────────────────────────────────────
    error: Optional[str]