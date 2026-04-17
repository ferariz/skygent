"""
skygent/agent/graph.py — LangGraph graph definition
=====================================================

Design decisions
----------------
1. Conditional edges after every node check state["error"]: if any node
   sets an error, the graph routes to END immediately. This prevents a
   failing fetch from cascading into a diff on stale/None data.

2. The significance gate is the primary early-exit: evaluate_significance
   routes to END when significant=False. This is the common case — most
   polls detect no meaningful change. The LLM is never called on these runs.

3. The first-run gate is in fetch_forecast_node: when previous_snapshot is
   None, the node sets significant=False and the graph exits after
   fetch_forecast without running diff or significance. The conditional
   edge after fetch_forecast checks this case.

4. Graph is compiled once at module level and reused across all scheduler
   invocations. LangGraph compiled graphs are thread-safe and async-safe.

5. run_agent() is the single public entry point for the scheduler. It
   accepts a profile and an optional previous_snapshot, runs the graph,
   and returns the final state. The scheduler is responsible for loading
   the previous snapshot from the DB and persisting the current snapshot
   and alert after the run.

Graph structure
---------------
    fetch_forecast
         │
         ▼
    [error or first-run?] ──► END
         │
         ▼
    analyze_diff
         │
         ▼
    [error?] ──► END
         │
         ▼
    evaluate_significance
         │
         ▼
    [error or not significant?] ──► END
         │
         ▼
      narrate
         │
         ▼
    [error?] ──► END
         │
         ▼
      notify
         │
         ▼
        END
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from skygent.agent.nodes import (
    analyze_diff_node,
    evaluate_significance_node,
    fetch_forecast_node,
    narrate_node,
    notify_node,
)
from skygent.agent.state import AgentState
from skygent.core.models import ForecastSnapshot, MonitoringProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def after_fetch(state: AgentState) -> Literal["analyze_diff", "__end__"]:
    """
    Route to analyze_diff if the fetch succeeded and there is a previous
    snapshot to diff against. Exit to END on error or first run.
    """
    if state.get("error"):
        logger.warning("Graph exiting after fetch_forecast: %s", state["error"])
        return END

    # First run: fetch_forecast_node sets significant=False when previous is None
    if state.get("significant") is False:
        logger.info("Graph exiting after fetch_forecast: first run, no diff possible")
        return END

    return "analyze_diff"


def after_analyze(state: AgentState) -> Literal["evaluate_significance", "__end__"]:
    """Route to evaluate_significance, or END on error."""
    if state.get("error"):
        logger.warning("Graph exiting after analyze_diff: %s", state["error"])
        return END
    return "evaluate_significance"


def after_significance(state: AgentState) -> Literal["narrate", "__end__"]:
    """
    Route to narrate if the change is significant, otherwise END.
    This is the primary early-exit — the common case for most polls.
    """
    if state.get("error"):
        logger.warning(
            "Graph exiting after evaluate_significance: %s", state["error"]
        )
        return END

    if not state.get("significant"):
        logger.info(
            "Graph exiting after evaluate_significance: no significant change detected"
        )
        return END

    return "narrate"


def after_narrate(state: AgentState) -> Literal["notify", "__end__"]:
    """Route to notify, or END on error."""
    if state.get("error"):
        logger.warning("Graph exiting after narrate: %s", state["error"])
        return END
    return "notify"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the Skygent agent graph.

    Returns a compiled LangGraph StateGraph ready to invoke.
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("fetch_forecast",        fetch_forecast_node)
    builder.add_node("analyze_diff",          analyze_diff_node)
    builder.add_node("evaluate_significance", evaluate_significance_node)
    builder.add_node("narrate",               narrate_node)
    builder.add_node("notify",                notify_node)

    # Entry point
    builder.set_entry_point("fetch_forecast")

    # Conditional edges
    builder.add_conditional_edges("fetch_forecast",        after_fetch)
    builder.add_conditional_edges("analyze_diff",          after_analyze)
    builder.add_conditional_edges("evaluate_significance", after_significance)
    builder.add_conditional_edges("narrate",               after_narrate)

    # notify always goes to END
    builder.add_edge("notify", END)

    return builder.compile()


# Compiled graph — reused across all scheduler invocations
_graph = build_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_agent(
    profile: MonitoringProfile,
    previous_snapshot: ForecastSnapshot | None = None,
) -> AgentState:
    """
    Run one full agent cycle for a MonitoringProfile.

    Parameters
    ----------
    profile:           the event to monitor
    previous_snapshot: the last stored snapshot for this profile, or None
                       on the first run

    Returns
    -------
    Final AgentState after the graph completes. The caller (scheduler) is
    responsible for:
    - Persisting state["current_snapshot"] to the DB
    - Persisting state["alert"] to the DB if state["significant"] is True
    - Logging state["error"] if present

    This function never raises — all errors are captured in state["error"].
    """
    initial_state: AgentState = {
        "profile": profile,
        "previous_snapshot": previous_snapshot,
        "current_snapshot": None,
        "changes": None,
        "significant": None,
        "triggering_variables": None,
        "alert": None,
        "error": None,
    }

    logger.info(
        "run_agent: starting for '%s' (previous_snapshot=%s)",
        profile.name,
        previous_snapshot.id if previous_snapshot else "None",
    )

    final_state: AgentState = await _graph.ainvoke(initial_state)

    if final_state.get("error"):
        logger.error(
            "run_agent: completed with error for '%s': %s",
            profile.name, final_state["error"],
        )
    elif final_state.get("significant"):
        logger.info(
            "run_agent: alert generated for '%s' (alert_id=%s)",
            profile.name, final_state["alert"].id,
        )
    else:
        logger.info(
            "run_agent: no significant change for '%s'",
            profile.name,
        )

    return final_state