# Changelog

All notable changes to Skygent are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0] — 2026-05-17

Session 1 hardening: observability, resilience, and audit trail.

### Added

- **WAL mode** — SQLite WAL journal mode enabled on startup for safe
  concurrent reads from the Streamlit dashboard alongside API writes
- **structlog** — native structured logging across all modules
  (`skygent/agent/nodes.py`, `skygent/scheduler/jobs.py`); key-value
  fields on every log call; JSON renderer in production, console renderer
  in development; `before_sleep_log` wired to tenacity retry hooks
- **Tenacity retries on Open-Meteo** — `_openmeteo_retry` decorator with
  `stop_after_attempt(3)`, `wait_exponential(min=2, max=10)`,
  `retry_if_exception_type(OpenMeteoError)`, `reraise=True`
- **Tenacity retries on GPT** — `_llm_retry` decorator with
  `stop_after_attempt(3)`, `wait_exponential(min=1, max=8)`,
  `retry_if_exception_type(Exception)`, `reraise=True`
- **Fallback narrative** — `_build_fallback_narrative(profile, alert)`
  assembles a plain-text summary from structured data when the LLM is
  unavailable after all retries; alert is always delivered
- **`PollRun` audit table** — SQLModel table recording every scheduler
  tick: `profile_id`, `ran_at`, `status` (`ok` | `error` | `skipped`),
  `changes_detected`, `alert_sent`, `alert_id`, `error_message`,
  `duration_ms`; written in a `finally` block so DB write never crashes
  the scheduler job
- **Real `/health` endpoint** — returns `status`, `scheduler_running`,
  `scheduled_jobs`, `last_poll_ran_at`, `last_poll_status`, `db_ok`;
  `status` is `"degraded"` when DB or scheduler is unhealthy; entire
  body wrapped in `try/except` so the endpoint never raises a 500
- **Poll History Streamlit panel** — new "Poll History" page in the
  dashboard; queries last 50 `PollRun` rows; displays full-width
  dataframe with human-readable column labels; shows Total runs, Alerts
  sent, and Error rate metrics below the table
- **`CONTRIBUTING.md`** — contribution guidelines
- **`docs/architecture.md`** — architecture reference document

### Changed

- Four test assertions updated to reflect new behavior:
  `test_llm_failure_returns_error` now asserts fallback alert is returned
  instead of an error dict; `test_marks_alert_as_sent` and
  `test_significant_change_produces_sent_alert` mock `send_alert` to
  avoid requiring `TELEGRAM_BOT_TOKEN` in CI; `test_health_returns_200`
  accepts both `"ok"` and `"degraded"` and checks for
  `"scheduler_running"` key instead of exact response match

---

## [0.1.0] — 2026-04-19

First complete MVP. All seven steps implemented, tested, and live-demonstrated.

### What it does

Monitors weather forecasts for user-defined events. Polls Open-Meteo every
N hours (default: 6), detects significant changes against the previous
snapshot, and delivers a GPT-4o-mini-generated alert narrative via Telegram.
Deterministic code makes all threshold and significance decisions; the LLM
is called exactly once per alert cycle, only to write the message.

Full pipeline demonstrated live: Open-Meteo (ECMWF IFS Best Match, 9 km) →
diff engine → significance evaluator → GPT-4o-mini → Telegram delivery.

### Added

**Core layer (`skygent/core/`)**
- `MonitoringProfile` — event configuration with per-variable thresholds,
  context field (`social_event` | `agriculture` | `energy` | `logistics`),
  and `event_duration_hours` documenting the hourly upgrade path
- `ForecastSnapshot` — immutable forecast capture with `horizon_days` and
  `model_used` (NWP model provenance per snapshot)
- `DiffAnalyzer` — deterministic delta computation between snapshots;
  `weather_code` excluded from numeric diff and evaluated categorically
- `SignificanceEvaluator` — threshold rules, WMO severity rank table,
  bidirectional weathercode alerts, horizon→confidence mapping

**Open-Meteo integration (`skygent/integrations/openmeteo.py`)**
- Async fetch with `timezone=UTC`, `start_date`/`end_date` params
- Captures `response["model"]` → `ForecastSnapshot.model_used`
  (defaults to `"best_match"` — Best Match endpoint does not expose
  which NWP models it selected)
- `JSONDecodeError` wrapped as `OpenMeteoError`
- WARNING logged when target date exceeds forecast window

**LangGraph agent (`skygent/agent/`)**
- Five-node graph: `fetch_forecast → analyze_diff → evaluate_significance
  → narrate → notify`
- Lazy LLM init — `OPENAI_API_KEY` not required at import time
- Stale state clearing — run N data never bleeds into run N+1
- Nodes never raise — all exceptions caught and returned as `{"error": ...}`
- LLM: OpenAI `gpt-4o-mini`, `max_tokens=400`

**Scheduler (`skygent/scheduler/jobs.py`)**
- `AsyncIOScheduler`, one job per profile, `max_instances=1`
- `next_run_time=now()` on registration → immediate first fetch
- `set_snapshot_store()` public setter for dependency injection
- Per-profile error isolation at startup

**FastAPI + database (`skygent/api/`)**
- `POST/GET/DELETE /api/v1/profiles`, `GET /api/v1/alerts`,
  `GET /api/v1/status`, `GET /health`
- `ProfileRow`, `SnapshotRow` (with `model_used` column), `AlertRow`
- `DBSnapshotStore` — DB-backed snapshot store, same interface as
  in-memory store; swap-in at startup via `set_snapshot_store()`
- `list_profiles(active_only=True)` filters by `event_datetime > now()`
  AND `is_active=True` — prevents stale boolean bug
- `deregister_snapshots()` — soft-delete with `deregistered_at` timestamp;
  re-registered profiles start a fresh diff baseline
- POST /profiles atomicity — scheduler failure rolls back DB write
- `ProfileCreate.context` typed as `Literal[...]` for accurate OpenAPI schema

**Telegram notifications (`skygent/integrations/telegram.py`)**
- HTML parse mode (not MarkdownV2 — safer for LLM-generated text)
- `TelegramError` wraps all failures including `JSONDecodeError`
- HTML-safe truncation at 4096 char limit
- `notify_node` routes on `profile.notification_channel`

**Streamlit dashboard (`ui/app.py`)**
- Register event with interactive map picker (streamlit-folium)
- Active profiles with deregister button
- Alert history with confidence badges and narrative display
- Configurable API URL in sidebar

**Documentation**
- `README.md` — project overview, NWP data sources table, confidence
  scoring basis, South America regional note, ensemble/ML roadmap
- `docs/design.md` — architecture, all design decisions, test coverage,
  portfolio-strengthening section, SQLite migration note
- `LICENSE` — MIT

### Test coverage

204 unit tests, 3 integration tests (deselected by default).

| File | Tests |
|---|---|
| `test_diff.py` | 20 |
| `test_significance.py` | 34 |
| `test_openmeteo.py` | 31 |
| `test_agent.py` | 37 |
| `test_scheduler.py` | 30 |
| `test_api.py` | 42 |
| `test_telegram.py` | 21 |

### Known limitations

- Daily aggregates only (not hourly event-window extraction) —
  upgrade path documented via `event_duration_hours`
- No high-resolution regional NWP model for South America —
  relies on ECMWF IFS HRES (9 km) and GFS (25 km)
- Confidence scoring is a horizon heuristic, not calibrated probabilities
- SQLite only — no Alembic migrations; schema changes require DB deletion
  in development or manual `ALTER TABLE` in production
- Telegram bot is outbound only — profile registration via dashboard or API

---

## [Unreleased] — feat/telegram-bot-conversation

### Added

**Telegram bot inbound handler (`skygent/integrations/telegram_bot.py`)**
- Polling-based inbound message handler (no public URL required)
- SQLite-backed conversation state machine — survives bot restarts
- Conversation flow: /start → location pin → name → date → time →
  context (keyboard) → duration (keyboard) → confirm → register
- Natural date parsing via `dateparser` (handles "Sep 15", "2026-09-15",
  "15/09/2026" and more)
- Inline keyboard buttons for structured choices (context, duration, confirm)
- Welcome forecast on registration: fetches real forecast, generates
  GPT-4o-mini narrative framed as introduction not alert
- Conversation expiry: stale state cleared after 24 hours
- /cancel command at any step

**Bot entry point (`skygent/bot.py`)**
- Long-polling loop (timeout=30s, offset tracking, retry on error)
- Runs as separate process alongside uvicorn
- Graceful shutdown on Ctrl+C

**Database (`skygent/api/database.py`)**
- `ConversationStateRow` table: chat_id, step, data (JSON), updated_at
- `get/save/clear_conversation_state()` CRUD helpers

**Core models (`skygent/core/models.py`)**
- `MonitoringProfile.telegram_chat_id: str | None` — routes alerts to
  the registering user's chat rather than the shared env var

**Agent (`skygent/agent/nodes.py`)**
- `notify_node` routes to `profile.telegram_chat_id` when set,
  falls back to `TELEGRAM_CHAT_ID` env var for API/dashboard registrations

### Design decisions

- **Polling over webhooks**: no public URL required for local development.
  Switch to webhooks for production deployment (one-line change).
- **SQLite state over in-memory**: state survives restarts and WatchFiles
  reloads during development. Existing DB stack reused at near-zero cost.
- **dateparser for natural input**: handles all common date formats without
  custom parsing logic.
- **Separate process from uvicorn**: polling loop is synchronous and blocking;
  keeping it separate from the async API process avoids event loop conflicts.
- **Per-user chat routing**: `telegram_chat_id` on `MonitoringProfile` enables
  multi-user deployment without a shared notification channel.