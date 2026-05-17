# Skygent — Architecture

This document explains the design of Skygent: what problem it solves, how it
is structured, and why specific decisions were made. It is written for
engineers who want to understand the system before contributing or extending it.

---

## The problem

Modern NWP systems like ECMWF IFS produce high-quality forecasts updated every
6 hours at ~9 km resolution. The problem is not data quality or availability.

The problem is **decision fatigue from continuous data**.

A user monitoring a forecast for an outdoor event does not need to see every
update. They need to know: *did something change in a way that matters?*
That question has a precise answer — and that precision is exactly why it
should not be delegated to an LLM.

---

## Core design principle

> **Deterministic code owns decisions. LLMs own communication.**

This produces three non-negotiable rules:

1. All significance logic is deterministic and testable
2. LLMs are never used to evaluate thresholds or make decisions
3. LLMs are invoked only when there is information worth communicating

A useful test: **if you can write an `assert` for it, it does not belong
in an LLM call.**

This principle has a practical consequence: the entire decision pipeline —
204 tests — runs with zero LLM dependencies. LLM behavior is tested
separately and only at the narration boundary.

---

## System layers

Skygent has three layers with strict separation:

```
┌─────────────────────────────────────────────────────┐
│  Layer 1 — Decision Engine (deterministic)          │
│  DiffAnalyzer + SignificanceEvaluator               │
│  Computes deltas, applies thresholds, builds alerts │
└─────────────────────┬───────────────────────────────┘
                      │ structured Alert object
┌─────────────────────▼───────────────────────────────┐
│  Layer 2 — LLM Narrator (communication only)        │
│  GPT-4o-mini via LangGraph                          │
│  Receives structured data, writes narrative only    │
└─────────────────────┬───────────────────────────────┘
                      │ alert with narrative
┌─────────────────────▼───────────────────────────────┐
│  Layer 3 — Delivery + Persistence                   │
│  Telegram bot, FastAPI, SQLite, APScheduler         │
│  Schedules polls, stores state, routes alerts       │
└─────────────────────────────────────────────────────┘
```

The LLM is **never in the critical path of decision-making**. An LLM outage
degrades narration quality but never suppresses a real alert — the fallback
narrative assembles a plain-text message from the structured alert data the
engine already computed.

---

## Data flow

Every poll cycle follows this path:

```
fetch_forecast → analyze_diff → evaluate_significance → narrate → notify
```

Implemented as a LangGraph graph where each step is a node:

**`fetch_forecast_node`**
Calls Open-Meteo (ECMWF IFS Best Match) for the monitored location and
timeframe. Returns a `ForecastSnapshot`. On the first run for a profile,
there is no previous snapshot to diff — the node stores the baseline and
exits without generating an alert.

**`analyze_diff_node`**
Compares the previous and current snapshots using `DiffAnalyzer`. Computes
per-variable deltas (absolute and percentage). Returns a `changes` dict.

**`evaluate_significance_node`**
Applies `SignificanceEvaluator` to the changes dict. Checks each variable
against profile-configured thresholds. Returns `significant: bool` and
a list of triggering variables. If significant, builds an `Alert` object
with confidence scoring based on forecast horizon.

**`narrate_node`**
The only node that calls an LLM. Receives the structured `Alert` and asks
GPT-4o-mini to write a plain-text narrative. If the LLM fails after 3
retries, assembles a fallback narrative from the alert's structured fields —
the alert is always delivered.

**`notify_node`**
Routes the alert to the configured delivery channel (currently Telegram).
Uses `profile.telegram_chat_id` for per-user routing if registered via bot,
falls back to `TELEGRAM_CHAT_ID` env var.

---

## Confidence scoring

Skygent maps forecast horizon to a confidence label using known NWP skill
limits. These are physics-informed heuristics, not calibrated probabilities:

| Horizon     | Confidence | Meaning                          |
|-------------|------------|----------------------------------|
| ≤ 3 days    | High       | Deterministic skill reliable     |
| 3–7 days    | Medium     | Moderate uncertainty             |
| > 7 days    | Low        | Ensemble spread dominates        |

This reflects the known behavior of global NWP systems: deterministic skill
degrades with lead time, uncertainty grows non-linearly. The confidence label
is always included in the narrated alert.

---

## Scheduling

APScheduler (`AsyncIOScheduler`) runs one job per active `MonitoringProfile`,
each on an `IntervalTrigger` (default: 6 hours). Jobs are persisted to a
SQLite job store so they survive application restarts.

Every job execution writes a `PollRun` row to the database in a `finally`
block — the audit record is written regardless of whether the run succeeded,
errored, or found no significant change. This is the observability foundation
for both operational monitoring and the planned forecast Q&A feature.

**Economics:** most runs exit early (no significant change → no LLM call).
At 5 users monitored daily, this produces ~1–2 LLM calls per user per week
rather than ~28 — a 90%+ reduction in token usage compared to a naive
polling-and-narrating approach.

---

## Persistence

**SQLite with WAL mode.** Enabled at startup:

```python
PRAGMA journal_mode=WAL
PRAGMA synchronous=NORMAL
```

WAL mode allows one writer and multiple concurrent readers without locking,
which is necessary for the FastAPI process and Telegram bot process to share
the same database file on Railway's persistent volume.

**Why not PostgreSQL:** SQLite with WAL is correct and sufficient for the
current scale (1–5 users, one writer per poll cycle). PostgreSQL adds
operational complexity without adding capability at this load. The migration
path is straightforward when concurrency requirements change.

---

## Resilience

Three layers of fault tolerance:

**Open-Meteo fetch retries:** `tenacity` retries up to 3 times with
exponential backoff (2s, 4s, 8s) on `OpenMeteoError`. All httpx network
failures are wrapped into `OpenMeteoError` by the client before reaching
the retry layer.

**LLM narration retries:** `tenacity` retries up to 3 times with exponential
backoff (1s, 2s, 4s) on any exception. If all retries fail, `_build_fallback_narrative`
assembles a plain-text alert from the structured data the significance
evaluator already computed. **An LLM failure never suppresses a real alert.**

**PollRun write protection:** the `finally` block DB write is wrapped in its
own `try/except` — a database write failure logs an error but never crashes
the scheduler job.

---

## Logging

Structured logging via `structlog`. In development (`ENV=development`),
colored key-value output to console. In production (`ENV=production`),
JSON lines — one object per log event, queryable in any log aggregator.

Every meaningful event carries typed fields:

```json
{
  "event": "scheduler: alert generated",
  "profile_name": "Ana's Wedding",
  "profile_id": "ee01578f-...",
  "alert_id": "56156322-...",
  "confidence": "medium",
  "horizon_days": 5.0,
  "sent": true,
  "timestamp": "2026-05-17T00:05:06.657Z",
  "level": "info"
}
```

This makes production incidents diagnosable without SSH access to the server.

---

## Key non-decisions

Decisions explicitly not made, and why:

**No ReAct / Plan-and-Execute agent loop:** Skygent's decision boundaries are
explicit and computable. An agentic loop would introduce non-determinism into
significance evaluation — exactly the failure mode the architecture is designed
to prevent.

**No Redis:** APScheduler with SQLite job store handles the scheduling
requirements. Redis would add infrastructure complexity without adding
capability.

**No multi-model routing:** GPT-4o-mini is invoked once per alert cycle for
narration only. The narration layer is isolated enough that switching models
is a one-line change. Routing logic adds complexity that isn't justified at
this stage.

**No user authentication:** the system is designed for 1–5 known users
onboarded via Telegram. Authentication is a product decision, not an
engineering one, at this scale.

---

## Extending the system

**New weather variables:** add threshold configuration to `MonitoringProfile`
and a comparison rule to `SignificanceEvaluator`. No changes to the agent
graph, narration, or delivery.

**New delivery channels:** implement a delivery function in
`skygent/integrations/`, add a branch in `notify_node`. The LLM narration
layer and graph are unchanged.

**New event contexts:** `MonitoringProfile.context` is currently
`social_event | agriculture | energy | logistics`. The narrator's system
prompt adjusts tone based on this field. Adding a new context means adding
a tone description to the system prompt — no code changes.

**Forecast Q&A:** users with no pending alerts can ask context-aware questions
answered using the current forecast snapshot and the `PollRun` change history.
This preserves the architecture principle: the LLM receives structured data
the engine has already computed and validated, and narrates it in response to
a user query. The `poll_runs` audit table built in v0.2.0 is the context
source for this feature.
