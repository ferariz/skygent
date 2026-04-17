# Skygent — Design Document

> Steps 1–4 complete. Last updated April 2026.

---

## Table of contents

1. [Architecture overview](#1-architecture-overview)
2. [Core data models](#2-core-data-models)
3. [Step 2 — Open-Meteo integration](#3-step-2--open-meteo-integration)
4. [Step 3 — LangGraph agent](#4-step-3--langgraph-agent)
5. [Step 5 — Scheduler](#5-step-4--scheduler)
6. [Key design decisions](#6-key-design-decisions)
7. [Test coverage](#7-test-coverage)
8. [What comes next](#8-what-comes-next)

---

## 1. Architecture overview

Skygent runs as a scheduled agentic loop. Each active monitoring profile triggers one graph run every N hours (default: 6). A graph run goes through five steps:

| Node | LLM? | What it does |
|---|---|---|
| `fetch_forecast` | No | Calls Open-Meteo, returns `ForecastSnapshot` |
| `analyze_diff` | No | Runs `DiffAnalyzer`, returns `dict[str, VariableChange]` |
| `evaluate_significance` | No | Runs `SignificanceEvaluator`, builds `Alert` skeleton if triggered |
| `narrate` | **Yes** | Calls Claude with structured JSON payload, fills `Alert.narrative` |
| `notify` | No | Delivers alert (log stub — Telegram wired in Step 6) |

The graph has a conditional edge after every node. Most runs exit after `evaluate_significance` with `significant=False` — the LLM is never called on these runs.

**Core principle — hybrid intelligence:** deterministic code makes all threshold and significance decisions. The LLM is called exactly once per alert cycle, only to write the human-readable message. This separation makes the system fast, cheap, testable, and auditable.

---

## 2. Core data models

Three Pydantic v2 models form the data contract shared by every layer.

### MonitoringProfile

The unit of configuration. One profile = one event. The scheduler, agent nodes, diff engine, and significance evaluator all receive a profile.

| Field | Purpose / constraint |
|---|---|
| `id` | UUID — generated before DB write |
| `name` | Human label, e.g. `"Ana & Juan's Wedding"` |
| `location` | Validated `(lat, lon)` — Open-Meteo silently misbehaves on bad coords |
| `event_datetime` | UTC-aware — monitoring stops after this |
| `monitoring_start` | UTC-aware — defaults to `now()` |
| `check_interval_hours` | `int >= 1` — enforced; 0 would spin APScheduler |
| `event_duration_hours` | `int >= 1` — MVP unused, documents hourly upgrade path |
| `variables` | Open-Meteo variable names to fetch (`wind_speed_10m_max`, `weather_code`, etc.) |
| `thresholds` | Per-variable alert magnitude — must be a subset of `variables` |
| `context` | `social_event` \| `agriculture` \| `energy` \| `logistics` |

### ForecastSnapshot

An immutable point-in-time capture of the forecast. Snapshots are never mutated — the diff engine always compares two distinct objects.

`horizon_days` is computed at fetch time and stored because it drives confidence scoring. Recomputing it later would require knowing the original fetch time, which is fragile.

### Alert

Generated when the significance evaluator finds a threshold crossing. `narrative` starts empty and is filled by the narrator node before delivery.

```
confidence: "high" | "medium" | "low"   ← derived by significance.py, not the LLM
narrative:  ""                           ← filled by narrate node
sent:       False                        ← flipped by notify node
```

### VariableChange (TypedDict)

```python
class VariableChange(TypedDict):
    from_value: float
    to_value:   float
    delta:      float
    delta_pct:  float | None   # None when from_value == 0
```

---

## 3. Step 2 — Open-Meteo integration

### Timezone strategy

We always request `timezone=UTC`. Using `timezone=auto` would return local-time date strings, making date-matching dependent on the UTC offset of each location. UTC date strings are stable, unambiguous, and consistent across all profiles worldwide.

Open-Meteo daily aggregates are computed over the 24-hour UTC day. For a Montevideo event at 16:00 local (19:00 UTC), the UTC-day aggregate is correct for planning purposes.

### API parameter design

`forecast_days` and `start_date`/`end_date` are mutually exclusive on the Open-Meteo API. We use `start_date` + `end_date` to target a specific event date precisely. `FORECAST_DAYS` is used only to compute the `end_date` cap.

### Variable name correctness

The correct Open-Meteo daily variable names are:
- `wind_speed_10m_max` (not `windspeed_10m_max`)
- `weather_code` (not `weathercode`)

Using the wrong names produces a silent failure: the HTTP call succeeds, the response parses correctly, but the row extraction returns `None` for both variables with no error. This was caught during Cursor review and fixed in `models.py` defaults and all downstream references.

`CATEGORICAL_VARIABLES` in `diff.py` must use `"weather_code"` to match `profile.variables`, otherwise the filter never fires and WMO codes enter the numeric diff loop producing meaningless arithmetic.

### Daily vs hourly — documented tradeoff

**MVP uses daily aggregates** (`temperature_2m_max`, `wind_speed_10m_max`, `precipitation_probability_max`). These represent the worst case for the full calendar day, not a specific event window. A wedding at 7 PM and one at noon receive the same daily aggregate.

**Why daily is defensible for MVP:**

The 6-hour polling cadence is the safety net. A storm that misses a 7 PM ceremony window almost certainly appeared in an earlier poll as a deteriorating forecast and triggered an alert. If conditions improve, the next poll fires a second alert (bidirectional `weather_code` logic handles both directions). The LLM narrator frames residual uncertainty explicitly in every message: *"medium confidence 5 days out, re-checked every 6 hours — conditions may still shift."*

**Production upgrade path:**

`event_duration_hours` (default 4h) is already stored on `MonitoringProfile`. To upgrade:
1. Switch `openmeteo.py` to the hourly endpoint
2. Extract the window `[event_datetime, event_datetime + event_duration_hours]`
3. Aggregate: `precipitation_probability` → max, `temperature_2m` → mean, `wind_speed_10m` → max, `weather_code` → worst severity rank
4. The diff engine and significance evaluator require **no changes** — they operate on one scalar per variable regardless of derivation

---

## 4. Step 3 — LangGraph agent

### AgentState

`AgentState` is a `TypedDict` with all fields `Optional` and defaulting to `None`. LangGraph merges partial dicts returned by nodes — nodes only return the fields they changed. All fields are present in the initial state dict to avoid `KeyError` on any access pattern.

### Node design principles

**Nodes never raise.** An unhandled exception in a LangGraph node does not route to `END` — it surfaces as an unhandled coroutine error. Every node catches its own exceptions and returns `{"error": message}` so the conditional edges always have a clean state to read.

**Lazy LLM initialization.** `ChatAnthropic` is created on first use via `_get_llm()`, not at import time. Import-time initialization would require `ANTHROPIC_API_KEY` even in test runs that mock the LLM, breaking test collection for all 148 unit tests.

**Stale state clearing.** LangGraph merges partial dicts without resetting unmentioned fields. `fetch_forecast_node` clears all downstream fields (`changes`, `significant`, `alert`, `triggering_variables`, `error`) at the start of every return dict so data from run N never bleeds into run N+1.

Ownership of clearing:
- `fetch_forecast_node` — clears everything downstream
- `analyze_diff_node` — clears `significant`, `triggering_variables`, `alert`
- `evaluate_significance_node` — clears `alert` when `significant=False`

### Narrator prompt design

The LLM receives a **structured JSON payload** — not a prose description — so it cannot hallucinate field names or values.

System prompt constraints:
- Plain prose only, no markdown
- Under 200 words
- Always state: what changed and by how much, confidence level and its meaning, when the next check runs
- Tone tailored to `profile.context`
- Never invent data — only describe what is in the provided JSON

### Graph structure

```
fetch_forecast
     │
     ▼
[error or first run?] ──► END
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
[error or not significant?] ──► END   ← common case, LLM never called
     │
     ▼
narrate  ← only LLM call in the system
     │
     ▼
[error?] ──► END
     │
     ▼
notify → END
```

---

## 5. Step 4 — Scheduler

### AsyncIOScheduler

`AsyncIOScheduler` runs jobs directly on the existing asyncio event loop. `BackgroundScheduler` runs jobs in a thread pool, requiring `asyncio.run()` inside each job — creating a new event loop per invocation. Since the entire agent stack is async, `AsyncIOScheduler` is the correct choice.

### One job per profile

Each `MonitoringProfile` gets its own `IntervalTrigger` job keyed by profile ID. Jobs are registered with `next_run_time=datetime.now(UTC)` so the first fetch happens immediately — giving the system a baseline snapshot without waiting a full interval. Without a baseline, the second run is also a no-op (nothing to diff against).

`max_instances=1` prevents overlapping runs for the same profile if a run takes longer than the interval.

### SnapshotStore

The scheduler needs one thing between runs: the previous snapshot to pass to `run_agent()`. For the MVP this is an in-memory dict with a `get`/`set`/`clear` interface. The FastAPI layer (Step 5) will replace it with a SQLModel-backed store. Because the scheduler only calls `store.get()` and `store.set()`, that replacement requires **zero changes** to `jobs.py`.

### Expiry handling

`register_profile()` checks `profile.is_active` before adding a job. The job itself also checks at runtime — handling profiles that expire between scheduler restarts. On expiry detection, the job removes itself **and clears the snapshot store entry**, preventing stale in-memory data from persisting until the next restart.

---

## 6. Key design decisions

These are the decisions that would look arbitrary without context.

### `weather_code` not `weathercode`

Open-Meteo API uses `weather_code` with an underscore. Wrong name → silent `None` values in every snapshot, no error, no alert ever fired for wind or weather conditions. `CATEGORICAL_VARIABLES` must match `profile.variables` exactly or the filter never fires and WMO codes enter the numeric diff loop.

### Bidirectional weathercode alerts

Significant improvement (thunderstorm → clear) is as actionable as deterioration. A user who cancelled outdoor plans because of a storm forecast deserves an *"actually it's clear now"* alert. `abs(rank_delta) >= threshold` handles both directions; the narrator receives the direction and frames the message accordingly.

### Negative horizon → ValueError

`horizon_to_confidence(-1.0)` would silently return `"high"` without the guard (because `-1.0 <= 3.0`). That is the worst possible answer for a past event — it would label an already-occurred wedding as high-confidence. The `ValueError` surfaces the upstream scheduling bug immediately.

### No retry logic in openmeteo.py

Retry belongs at the scheduler level (job error handling) or in an httpx Transport, not embedded in the fetch function. Adding it there would make the function harder to test and hide transient failures from the agent's state machine.

### Internal/API name split abandoned

An earlier version used `"weathercode"` internally and `"weather_code"` as the API key, with translation at `.data.get()` call sites. This introduced the exact bug it was meant to prevent: `CATEGORICAL_VARIABLES` used the internal name, so the filter never matched `profile.variables`. One name everywhere is simpler and auditable with a single grep.

### `next_run_time=datetime.now()` on registration

Without this, a profile registered at 14:00 with a 6-hour interval would first run at 20:00. Immediate first run gives a baseline snapshot so the second run has something to diff against. The parameter is `datetime.now(timezone.utc)` — not `None`. In APScheduler 3.x, `None` means "don't schedule an immediate run."

---

## 7. Test coverage

148 unit tests, 2 integration tests (deselected by default via `pytest.ini`).

| Test file | Tests | What is covered |
|---|---|---|
| `test_diff.py` | 20 | Delta math, None values, identity guards, categorical exclusion |
| `test_significance.py` | 34 | Confidence boundaries, weathercode ranks, bidirectional alerts, alert factory |
| `test_openmeteo.py` | 30 | Params, horizon computation, row extraction, mocked HTTP, live API |
| `test_agent.py` | 37 | Routing logic, all nodes, full graph, lazy LLM init, stale state clearing |
| `test_scheduler.py` | 29 | SnapshotStore, job registration/expiry, job function, lifecycle |

Run integration tests:

```bash
# Live Open-Meteo API
pytest -m integration tests/test_openmeteo.py -v

# Real LLM narration (requires ANTHROPIC_API_KEY)
pytest -m integration tests/test_agent.py -v
```

---

## 8. What comes next

**Step 5 — FastAPI routes (`skygent/api/routes.py`)**
- `POST /profiles` — register an event for monitoring
- `GET /profiles` — list active profiles and their scheduler status
- `GET /alerts` — list generated alerts
- `GET /status` — scheduler health and job summary
- Replace in-memory `SnapshotStore` with SQLModel + SQLite persistence

**Step 6 — Telegram notifications (`skygent/integrations/telegram.py`)**
- Replace the `notify_node` log stub with real Telegram Bot API delivery
- Node interface is already final — only the delivery implementation changes

**Step 7 — Streamlit dashboard (`ui/app.py`)**
- Register and manage profiles
- View alert history with narrative and confidence indicators
- Live scheduler status
