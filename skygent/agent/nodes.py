"""
skygent/agent/nodes.py — LangGraph node implementations
=========================================================

Design decisions
----------------
1. Each node is an async function that accepts AgentState and returns a
   partial AgentState dict. LangGraph merges the returned dict into the
   current state — nodes only need to return fields they changed.

2. Every node catches its own exceptions and returns {"error": message}
   rather than raising. This lets the graph's conditional edges route to
   END cleanly instead of crashing the entire graph run. The scheduler
   reads state["error"] after the run to decide whether to retry.

3. The narrate node is the only node that calls the LLM. It receives a
   fully-populated Alert (minus narrative) and asks Claude to write the
   human-readable summary. All threshold and significance decisions have
   already been made deterministically upstream.

4. The fetch node handles the "first run" case (no previous snapshot)
   by storing the current snapshot and returning significant=False so
   the graph exits without generating an alert. There is nothing to diff
   on a first run.

5. The notify node is a stub that logs the alert. The Telegram integration
   will be wired in when telegram.py is implemented (Step 6). The node
   interface is final — only the delivery implementation changes.

6. Nodes receive the full AgentState but only read what they need. This
   keeps coupling loose: the fetch node does not know about alerts, the
   narrate node does not know about thresholds.

Fixes applied after review (v2)
---------------------------------
A. Lazy LLM initialization: _llm was instantiated at module import time,
   which caused an immediate failure (missing OPENAI_API_KEY) even in
   test runs that mock the LLM. The LLM is now created on first use via
   _get_llm(). DiffAnalyzer and SignificanceEvaluator have no credentials
   and remain module-level singletons — they are safe to construct at import.

B. Explicit stale state clearing: LangGraph merges partial dicts into state
   without clearing unmentioned fields. This means fields populated in run N
   (changes, alert, triggering_variables, significant) would persist into
   run N+1 if the new run exits early. Each node now explicitly resets the
   downstream fields it owns to None/[] at the start of its return dict,
   ensuring state is always consistent with the current run's results.

   Ownership:
     fetch_forecast_node      clears: changes, significant, triggering_variables, alert, error
     analyze_diff_node        clears: significant, triggering_variables, alert
     evaluate_significance_node clears: alert (when not significant)
     narrate_node             clears: nothing (alert is updated in place)
     notify_node              clears: nothing (alert.sent flipped in place)

LLM prompt design (narrate node)
---------------------------------
The system prompt establishes the narrator's role and constraints:
- Never invent data — only describe what is in the alert's changes dict
- Always state the confidence level and its meaning
- Always mention the next check window (check_interval_hours)
- Tailor tone to the profile's context field (social_event vs agriculture)
- Keep the message under 200 words — it will be sent via Telegram

The user message provides the structured alert data as JSON so the LLM
has an unambiguous, parseable input rather than a prose description.
"""

from __future__ import annotations

import json
import logging
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from skygent.agent.state import AgentState
from skygent.core.diff import DiffAnalyzer
from skygent.core.models import Alert
from skygent.core.significance import SignificanceEvaluator
from skygent.integrations.openmeteo import OpenMeteoError, fetch_forecast
from skygent.integrations.telegram import TelegramError, send_alert

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Module-level singletons — stateless, no credentials, safe at import time
# ---------------------------------------------------------------------------

_diff_analyzer = DiffAnalyzer()
_significance_evaluator = SignificanceEvaluator()

# LLM is NOT initialized here — see _get_llm() below (fix A)
_llm_instance: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    """
    Return the shared LLM instance, creating it on first call.

    Lazy initialization ensures importing nodes.py never fails due to a
    missing OPENAI_API_KEY — the key is only required when the narrate
    node actually runs, not at import time. This makes test collection and
    all non-LLM tests work without any environment variable.
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model="gpt-4o-mini",
            max_tokens=400,
        )
    return _llm_instance


# ---------------------------------------------------------------------------
# Retry logic for Open-Meteo fetches
# ---------------------------------------------------------------------------

_openmeteo_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(OpenMeteoError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


async def _fetch_with_retry(profile):
    return await _openmeteo_retry(fetch_forecast)(profile)


# ---------------------------------------------------------------------------
# Node: fetch_forecast
# ---------------------------------------------------------------------------

async def fetch_forecast_node(state: AgentState) -> dict:
    """
    Fetch the current forecast snapshot for the profile's event date.

    On the first run for a profile, previous_snapshot is None. In that case
    we store the current snapshot and return significant=False — there is
    nothing to diff yet so no alert should be generated.

    Clears downstream fields (fix B): changes, significant,
    triggering_variables, alert, error — guarantees a clean slate for
    each run regardless of what a previous run left in state.

    Returns partial state: {current_snapshot, ...cleared fields} on success,
                           {error} on failure.
    """
    profile = state["profile"]
    log = logger.bind(profile_name=profile.name, profile_id=str(profile.id))
    log.info("fetch_forecast: starting")

    try:
        snapshot = await _fetch_with_retry(profile)
    except OpenMeteoError as exc:
        log.error("fetch_forecast: API error", error=str(exc))
        return {"error": f"fetch_forecast failed: {exc}"}
    except Exception as exc:
        log.error("fetch_forecast: unexpected error", error=str(exc))
        return {"error": f"fetch_forecast unexpected error: {exc}"}

    previous = state.get("previous_snapshot")

    # Clear all downstream fields so stale data from a prior run never bleeds
    # through — LangGraph merges, it does not reset unmentioned fields
    cleared = {
        "changes": None,
        "significant": None,
        "triggering_variables": None,
        "alert": None,
        "error": None,
    }

    if previous is None:
        log.info("fetch_forecast: first run — storing snapshot, no diff possible")
        return {**cleared, "current_snapshot": snapshot, "significant": False}

    log.info(
        "fetch_forecast: snapshot fetched",
        snapshot_id=str(snapshot.id),
        horizon_days=round(snapshot.horizon_days, 2),
    )
    return {**cleared, "current_snapshot": snapshot}


# ---------------------------------------------------------------------------
# Node: analyze_diff
# ---------------------------------------------------------------------------

async def analyze_diff_node(state: AgentState) -> dict:
    """
    Compare the previous and current snapshots and compute variable deltas.

    Clears downstream fields (fix B): significant, triggering_variables, alert.

    Returns partial state: {changes, ...cleared fields} on success,
                           {error} on failure.
    """
    profile = state["profile"]
    previous = state.get("previous_snapshot")
    current = state.get("current_snapshot")

    if previous is None or current is None:
        return {"error": "analyze_diff: missing snapshot(s) in state"}

    log = logger.bind(profile_name=profile.name)
    log.info("analyze_diff: comparing snapshots")

    try:
        changes = _diff_analyzer.compare(previous, current, profile)
    except ValueError as exc:
        return {"error": f"analyze_diff: incompatible snapshots — {exc}"}
    except Exception as exc:
        log.error("analyze_diff: unexpected error", error=str(exc))
        return {"error": f"analyze_diff unexpected error: {exc}"}

    log.info("analyze_diff: result", variables_diffed=len(changes))

    # Clear downstream fields so they reflect this run only
    return {
        "changes": changes,
        "significant": None,
        "triggering_variables": None,
        "alert": None,
    }


# ---------------------------------------------------------------------------
# Node: evaluate_significance
# ---------------------------------------------------------------------------

async def evaluate_significance_node(state: AgentState) -> dict:
    """
    Evaluate whether the diff crosses significance thresholds and build
    the alert skeleton if so.

    When not significant, explicitly sets alert=None (fix B) — a prior run
    may have left a stale Alert in state that would otherwise persist.

    Returns partial state: {significant, triggering_variables} always,
                           plus {alert} when significant=True,
                           or {error} on failure.
    """
    profile = state["profile"]
    previous = state.get("previous_snapshot")
    current = state.get("current_snapshot")
    changes = state.get("changes") or {}

    if previous is None or current is None:
        return {"error": "evaluate_significance: missing snapshot(s) in state"}

    log = logger.bind(profile_name=profile.name)

    try:
        significant, triggers = _significance_evaluator.is_significant(
            changes=changes,
            profile=profile,
            current_snapshot=current,
            previous_snapshot=previous,
        )
    except Exception as exc:
        log.error("evaluate_significance: unexpected error", error=str(exc))
        return {"error": f"evaluate_significance unexpected error: {exc}"}

    log.info(
        "evaluate_significance: result",
        significant=significant,
        triggers=triggers,
    )

    if not significant:
        return {
            "significant": False,
            "triggering_variables": [],
            "alert": None,          # explicitly clear any stale alert from prior run
        }

    alert = _significance_evaluator.build_alert(profile, previous, current, changes)

    return {
        "significant": True,
        "triggering_variables": triggers,
        "alert": alert,
    }


# ---------------------------------------------------------------------------
# Node: narrate
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a weather alert narrator for Skygent, an AI weather monitoring agent.
Your job is to write a clear, concise alert message when a forecast changes \
significantly for a user-defined event.

Rules:
- Write in plain language. No markdown, no bullet points — plain prose only.
- Keep the message under 200 words.
- Always state what changed, by how much, and in which direction.
- Always mention the forecast confidence level and what it means for reliability.
- Always mention when the next check will happen.
- Tailor the tone to the event context (social_event = warm and personal, \
agriculture = practical and direct, energy/logistics = technical and precise).
- Never invent data. Only describe what is in the alert JSON provided.
- Do not include a subject line or greeting — start directly with the update.
"""


def _build_narrate_prompt(state: AgentState) -> str:
    """Build the user message for the narrate node from current state."""
    profile = state["profile"]
    alert: Alert = state["alert"]

    confidence_meaning = {
        "high":   "forecast is reliable (≤3 days out)",
        "medium": "forecast has moderate uncertainty (3–7 days out)",
        "low":    "forecast has high uncertainty (>7 days out)",
    }

    payload = {
        "event": {
            "name": profile.name,
            "datetime": profile.event_datetime.isoformat(),
            "context": profile.context,
            "duration_hours": profile.event_duration_hours,
            "notes": profile.notes or None,
        },
        "forecast_change": {
            "confidence": alert.confidence,
            "confidence_meaning": confidence_meaning[alert.confidence],
            "horizon_days": round(alert.horizon_days, 1),
            "triggering_variables": state.get("triggering_variables", []),
            "changes": {
                var: {
                    "from": round(info["from_value"], 1),
                    "to":   round(info["to_value"], 1),
                    "delta": round(info["delta"], 1),
                }
                for var, info in alert.changes.items()
            },
        },
        "monitoring": {
            "check_interval_hours": profile.check_interval_hours,
            "next_check": f"in approximately {profile.check_interval_hours} hours",
        },
    }

    return json.dumps(payload, indent=2, default=str)


async def narrate_node(state: AgentState) -> dict:
    """
    Call Claude to generate a human-readable narrative for the alert.

    The LLM receives a structured JSON payload — not raw state — so it
    has an unambiguous input and cannot hallucinate field names or values.

    Returns partial state: {alert} with narrative filled in,
                           or {error} on failure.
    """
    alert: Alert = state.get("alert")
    if alert is None:
        return {"error": "narrate: no alert in state"}

    profile = state["profile"]
    log = logger.bind(profile_name=profile.name)
    log.info("narrate: generating narrative")

    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_build_narrate_prompt(state)),
        ]
        response = await llm.ainvoke(messages)
        narrative = response.content.strip()
    except Exception as exc:
        log.error("narrate: LLM error", error=str(exc))
        return {"error": f"narrate LLM error: {exc}"}

    alert_with_narrative = alert.model_copy(update={"narrative": narrative})

    log.info(
        "narrate: narrative generated",
        narrative_chars=len(narrative),
        model="gpt-4o-mini",
    )
    return {"alert": alert_with_narrative}


# ---------------------------------------------------------------------------
# Node: notify
# ---------------------------------------------------------------------------

async def notify_node(state: AgentState) -> dict:
    """
    Deliver the alert to the configured notification channel.

    Routes to the appropriate delivery function based on
    profile.notification_channel. Currently supports "telegram".
    Adding new channels means adding a branch here — the node
    interface and the rest of the graph are unchanged.

    Returns partial state: {alert} with sent=True on success,
                           or {error} on failure.
    """
    alert: Alert = state.get("alert")
    profile = state["profile"]

    if alert is None:
        return {"error": "notify: no alert in state"}

    if not alert.narrative:
        return {"error": "notify: alert has no narrative — narrate node must run first"}

    chat_id = profile.telegram_chat_id or None
    log = logger.bind(profile_name=profile.name, alert_id=str(alert.id))

    log.info(
        "notify: delivering alert",
        channel=profile.notification_channel,
        chat_id=chat_id,
    )

    try:
        if profile.notification_channel == "telegram":
            # If registered via the bot, route to the user's own chat.
            # Otherwise fall back to the TELEGRAM_CHAT_ID env var.
            await send_alert(alert, profile, chat_id=chat_id)
        else:
            # Unknown channel — log and mark sent so the pipeline does not stall.
            # Add new channel handlers here as the system grows.
            log.warning(
                "notify: unknown channel — logging alert and marking as sent",
                channel=profile.notification_channel,
            )
            log.info(
                "\n%s\n[ALERT — %s]\n%s\nConfidence: %s | Horizon: %.1f days\n%s\n",
                "=" * 60, profile.name, alert.narrative,
                alert.confidence, alert.horizon_days, "=" * 60,
            )
    except TelegramError as exc:
        log.error("notify: Telegram delivery failed", error=str(exc))
        return {"error": f"notify Telegram error: {exc}"}
    except Exception as exc:
        log.error("notify: unexpected error", error=str(exc))
        return {"error": f"notify unexpected error: {exc}"}

    alert_sent = alert.model_copy(update={"sent": True})
    log.info("notify: alert marked as sent")
    return {"alert": alert_sent}

