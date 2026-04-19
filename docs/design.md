# Skygent — Design Document

> All 7 steps complete. Last updated April 2026.

---

## Table of contents

1. [Architecture overview](#1-architecture-overview)
2. [Core data models](#2-core-data-models)
3. [Open-Meteo integration](#3-open-meteo-integration)
4. [LangGraph agent](#4-langgraph-agent)
5. [Scheduler](#5-scheduler)
6. [FastAPI + database layer](#6-fastapi--database-layer)
7. [Telegram notifications](#7-telegram-notifications)
8. [Streamlit dashboard](#8-streamlit-dashboard)
9. [Key design decisions](#9-key-design-decisions)
10. [Test coverage](#10-test-coverage)
11. [Portfolio strengthening — branch feat/portfolio-strengthening](#11-portfolio-strengthening)

---

## 1. Architecture overview

Skygent runs as a scheduled agentic loop. Each active monitoring profile triggers one graph run every N hours (default: 6). A graph run goes through five steps:

| Node | LLM? | What it does |
|---|---|---|
| `fetch_forecast` | No | Calls Open-Meteo, returns `ForecastSnapshot` |
| `analyze_diff` | No | Runs `DiffAnalyzer`, returns `dict[str, VariableChange]` |
| `evaluate_significance` | No | Runs `SignificanceEvaluator`, builds `Alert` skeleton if triggered |
| `narrate` | **Yes** | Calls GPT-4o-mini with structured JSON payload, fills `Alert.narrative` |
| `notify` | No | Delivers alert via Telegram |

The graph has a conditional edge after every node. Most runs exit after `evaluate_significance` with `significant=False` — the LLM is never called on these runs.

**Core principle — hybrid intelligence:** deterministic code makes all threshold and significance decisions. The LLM is called exactly once per alert cycle, only to write the human-readable message. This separation makes the system fast, cheap, testable, and auditable.

---

## 2. Core data models

Three Pydantic v2 models form the data contract shared by every layer.

### MonitoringProfile

The unit of configuration. One profile = one event.

| Field | Purpose / constraint |
|---|---|
| `id` | UUID — generated before DB write |
| `name` | Human label, e.g. `"Ana & Juan's Wedding"` |
| `location` | Validated `(lat, lon)` — Open-Meteo silently misbehaves on bad coords |
| `event_datetime` | UTC-aware — monitoring stops after this |
| `monitoring_start` | UTC-aware — defaults to `now()` |
| `check_interval_hours` | `int >= 1` — enforced; 0 would spin APScheduler |
| `event_duration_hours` | `int >= 1` — MVP unused, documents hourly upgrade path |
| `variables` | Open-Meteo variable names to fetch |
| `thresholds` | Per-variable alert magnitude — must be a subset of `variables` |
| `context` | `social_event` \| `agriculture` \| `energy` \| `logistics` |
| `notification_channel` | `"telegram"` (default) — routes `notify_node` |

### ForecastSnapshot

An immutable point-in-time capture of the forecast. Snapshots are never mutated — the diff engine always compares two distinct objects.

`horizon_days` is computed at fetch time and stored because it drives confidence scoring. Recomputing it later would require knowing the original fetch time, which is fragile.

`model_used` (added in portfolio-strengthening branch) stores the NWP model name returned by Open-Meteo (e.g. `"best_match"`, `"ecmwf_ifs025"`). `None` for snapshots created before this field was added or in tests that do not set it explicitly. This makes each snapshot a complete audit artifact — not just numbers, but also provenance.

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

## 3. Open-Meteo integration

### Timezone strategy

We always request `timezone=UTC`. UTC date strings are stable, unambiguous, and consistent across all profiles regardless of location.

### API parameter design

`forecast_days` and `start_date`/`end_date` are mutually exclusive on the Open-Meteo API. We use `start_date` + `end_date` to target a specific event date precisely.

### model_used capture

The Open-Meteo response includes a top-level `"model"` field naming the NWP model selected by Best Match. We extract it and store it on `ForecastSnapshot.model_used`. For Montevideo (34.9°S), Best Match typically selects ECMWF IFS HRES (9 km) as the global backbone. The log line now reads:

```
Snapshot abc123 created: horizon=5.20 days, model=best_match, variables=[...]
```

### Daily vs hourly — documented tradeoff

**MVP uses daily aggregates.** These represent the worst case for the full calendar day, not a specific event window.

**Production upgrade path** — `event_duration_hours` is already stored on `MonitoringProfile`. To upgrade:
1. Switch `openmeteo.py` to the hourly endpoint
2. Extract the window `[event_datetime, event_datetime + event_duration_hours]`
3. Aggregate: `precipitation_probability_max` → max, `temperature_2m_max` → mean, `wind_speed_10m_max` → max, `weather_code` → worst severity rank
4. The diff engine and significance evaluator require **no changes**

### Error handling

Both `raise_for_status()` and `response.json()` are in separate try/except blocks. A proxy returning HTML on a 200 (common during maintenance) would fail on JSON parsing, not on status — previously this escaped as a raw `ValueError`. Now always wrapped as `OpenMeteoError`.

A WARNING is logged when `target_date` exceeds the forecast window so the subsequent `OpenMeteoError` from `_extract_target_row` has context.

---

## 4. LangGraph agent

### Node design principles

**Nodes never raise.** Every node catches its own exceptions and returns `{"error": message}`.

**Lazy LLM initialization.** `ChatOpenAI` is created on first use via `_get_llm()`. Import-time initialization would require `OPENAI_API_KEY` even in unit tests that mock the LLM.

**Stale state clearing.** `fetch_forecast_node` clears all downstream fields on every run so data from run N never bleeds into run N+1.

### Narrator prompt design

The LLM receives a **structured JSON payload** — not a prose description — so it cannot hallucinate field names or values. System prompt constraints: plain prose, under 200 words, always state confidence level and next check time, tone tailored to `profile.context`.

### Graph structure

```
fetch_forecast
     │
     ▼
[error or first run?] ──► END
     │
     ▼
analyze_diff → evaluate_significance
     │
     ▼
[not significant?] ──► END   ← common case, LLM never called
     │
     ▼
narrate ← only LLM call in the system
     │
     ▼
notify → END
```

---

## 5. Scheduler

### AsyncIOScheduler

`AsyncIOScheduler` runs jobs directly on the existing asyncio event loop. `BackgroundScheduler` would require `asyncio.run()` inside each job, creating a new event loop per invocation — wrong for an async stack.

### One job per profile

Each profile gets its own `IntervalTrigger` job with `next_run_time=datetime.now(UTC)` so the first fetch happens immediately, giving the system a baseline snapshot.

### SnapshotStore and set_snapshot_store()

The scheduler uses a module-level `_snapshot_store` with a `get`/`set`/`clear` interface. `set_snapshot_store()` is a public setter (added after Cursor review) that `main.py` calls at startup to swap in the `DBSnapshotStore`. Using an explicit setter decouples `main.py` from the private attribute name.

---

## 6. FastAPI + database layer

### Separate DB models from domain models

`ProfileRow`, `SnapshotRow`, `AlertRow` are SQLModel table classes that store JSON-serialized domain objects in a `data` TEXT column. Only fields needed for queries/filtering are stored as indexed columns — this avoids schema migrations during early development.

### is_active staleness fix

`list_profiles(active_only=True)` filters by both `is_active=True` AND `event_datetime > now()`. A stored boolean alone would return expired profiles whose `is_active` was never explicitly flipped.

### Snapshot deregistration

`deregister_snapshots()` sets a `deregistered_at` timestamp rather than deleting rows. `load_latest_snapshot()` skips deregistered snapshots so a re-registered profile starts a fresh diff baseline. Audit history is retained.

### AlertRow — no denormalized columns

`AlertRow` stores only `id`, `profile_id`, `detected_at`, and the full JSON `data`. All reads go through `Alert.model_validate_json(row.data)` — one source of truth, no drift possible between columns and JSON payload.

### SnapshotRow.model_used

Added as part of the portfolio-strengthening branch. Stored as a nullable column alongside the JSON blob — queryable for diagnostics without a full JSON parse.

### ProfileRow.event_datetime index

`Field(index=True)` added after Cursor review found the column was described as indexed in a comment but the SQLModel field definition was missing it. Fixes a full table scan on every active-profile query.

### POST /profiles atomicity

`save_profile()` and `register_profile()` must succeed or fail together. If scheduling fails, raising `HTTPException` triggers the session dependency's rollback — no orphaned profile row is committed to the DB.

---

## 7. Telegram notifications

### HTML over MarkdownV2

Telegram's MarkdownV2 requires escaping dozens of special characters that LLM-generated narratives may contain. HTML only requires escaping `<`, `>`, `&` — much safer for dynamic text.

### Message structure

Narrative first (the important part), structured metadata below (horizon, triggers, next check). This gives the user the actionable information first.

### Error handling

`response.json()` is in a separate try/except from `raise_for_status()`, converting `JSONDecodeError` to `TelegramError`. A proxy returning HTML on a 200 previously escaped as a raw `ValueError`.

### HTML-safe truncation

Messages exceeding 4096 chars are truncated back to before any unclosed `<` tag to avoid Telegram rejecting malformed HTML.

### notify_node routing

Routes on `profile.notification_channel`. Unknown channels log and mark as sent rather than erroring — the alert was generated and attempted, delivery through an unsupported channel is not a pipeline failure.

---

## 8. Streamlit dashboard

### HTTP-only, decoupled from backend

`ui/app.py` talks to the FastAPI backend via `requests`, not by importing `skygent` modules. The dashboard can run on a different machine from the API. The API URL is configurable in the Settings sidebar and stored in `st.session_state`.

### Map picker

`streamlit-folium` renders an interactive Folium map. The user clicks to drop a pin; coordinates update in `st.session_state` and persist across reruns. Defaults to Montevideo (-34.9011, -56.1645). CartoDB Positron tile layer — no API key required.

Marker icon uses `folium.Icon(color="blue", icon="map-pin", prefix="fa")` with a try/except fallback to the default marker in case Font Awesome is unavailable in the Leaflet environment.

### Notes display: st.text() not st.markdown()

Free-text fields (notes) are rendered with `st.text()` to prevent user-supplied markdown from injecting formatting. Input uses `st.text_area()` for usability; display uses `st.text()` for safety.

### Manual refresh

Refresh buttons rather than timer-based `st.rerun()` to avoid hammering the API during development.

---

## 9. Key design decisions

### weather_code not weathercode

Open-Meteo API uses `weather_code` with an underscore. Wrong name → silent `None` values, no error, no alert ever fired. `CATEGORICAL_VARIABLES` must match `profile.variables` exactly.

### Bidirectional weathercode alerts

`abs(rank_delta) >= threshold` handles both deterioration and improvement. A user who cancelled outdoor plans deserves an "actually it's clear now" alert.

### Negative horizon → ValueError

`horizon_to_confidence(-1.0)` would silently return `"high"` (because `-1.0 <= 3.0`). That is the worst possible answer for a past event. The guard surfaces the upstream scheduling bug immediately.

### set_snapshot_store() over _snapshot_store mutation

Direct attribute mutation from `main.py` couples it to the private naming of `jobs.py`. A public setter is a stable contract, testable, and makes the injection point visible in the module's public API.

### DB deletion strategy: soft-delete snapshots, hard-delete nothing

Snapshots are never deleted — `deregistered_at` timestamps them out of the diff baseline while retaining full audit history. Alerts and profiles are similarly retained on deregistration (`is_active=False`). The DB is append-only from a data integrity perspective.

### SQLite for MVP, migration story documented

`create_db_and_tables()` uses `SQLModel.metadata.create_all()` which only creates missing tables — it never alters existing ones. Adding a new column (e.g. `model_used`) requires either deleting the DB file (acceptable for test data) or a proper migration via Alembic (required for production data). This tradeoff is documented here so it is not a surprise.

---

## 10. Test coverage

204 unit tests, 3 integration tests (deselected by default).

| Test file | Tests | What is covered |
|---|---|---|
| `test_diff.py` | 20 | Delta math, None values, identity guards, categorical exclusion |
| `test_significance.py` | 34 | Confidence boundaries, weathercode ranks, bidirectional alerts |
| `test_openmeteo.py` | 31 | Params, horizon, row extraction, mocked HTTP, non-JSON error, live API |
| `test_agent.py` | 37 | Routing logic, all nodes, full graph, lazy LLM init, stale state clearing |
| `test_scheduler.py` | 30 | SnapshotStore, set_snapshot_store(), job registration/expiry, lifecycle |
| `test_api.py` | 42 | CRUD helpers, all endpoints, POST atomicity, deregister flow |
| `test_telegram.py` | 21 | Message formatting, HTML escaping, HTTP errors, non-JSON error, live delivery |

---

## 11. Portfolio strengthening

Changes made on branch `feat/portfolio-strengthening`, merged to `main`.

### ForecastSnapshot.model_used

Added `model_used: str | None = None` to `ForecastSnapshot`. Open-Meteo's response includes a top-level `"model"` field. Storing it on every snapshot makes each one a complete audit artifact — not just forecast values, but also which NWP model produced them. `None`-safe for backwards compatibility with existing rows and tests.

Propagated through: `openmeteo.py` (capture from response) → `ForecastSnapshot` (domain model) → `SnapshotRow.model_used` (DB column) → log line (now shows model alongside horizon).

### Map picker in dashboard

`streamlit-folium` and `folium` added as dependencies. The Register event page now shows an interactive map — the user clicks to place a pin rather than typing raw coordinates. Coordinates persist in `st.session_state` across Streamlit reruns.

### Error hardening across the stack

Consistent `response.json()` try/except pattern applied to both `openmeteo.py` and the existing `telegram.py` — converts JSON parse failures to domain exceptions (`OpenMeteoError`, `TelegramError`) rather than letting raw `ValueError` escape to callers.

### Doc drift fixes in models.py

Variable names in the `event_duration_hours` upgrade path comment corrected: `windspeed_10m` → `wind_speed_10m_max`, `temperature_2m` → `temperature_2m_max`. The comment now matches the actual API variable names used throughout the codebase.

`datetime.utcnow()` in `is_active` replaced with `datetime.utcfromtimestamp(datetime.now(timezone.utc).timestamp())` — `utcnow()` was deprecated in Python 3.12.

### Database correctness fixes

- `ProfileRow.event_datetime`: `Field(index=True)` — previously described as indexed in a comment but the definition was missing it
- `list_alerts`: `.where()` applied before `.order_by().limit()` — conventional SQLAlchemy ordering
- `profile_ids`: single-column `Row` tuples unpacked to plain strings — SQLModel returns `Row` objects from column-level selects, not scalars

### SQLite migration note

Adding `model_used` to `SnapshotRow` is a schema change. `create_db_and_tables()` does not alter existing tables. For test/development environments, deleting `skygent.db` and restarting is the correct approach. For production environments with real data, use Alembic to generate an `ALTER TABLE snapshots ADD COLUMN model_used TEXT` migration.