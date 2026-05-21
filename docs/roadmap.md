# Skygent — Product Roadmap

This document captures the planned evolution of Skygent beyond v0.3.0.
It is opinionated: features are ordered by value, not by complexity.
The guiding constraint at every stage is the architecture invariant:

> **Deterministic code owns decisions. LLMs own communication.**

---

## Current state (v0.3.0)

- Deterministic diff engine + significance evaluator
- ECMWF IFS forecasts via Open-Meteo (0–15 day horizon)
- LangGraph agent: fetch → diff → evaluate → narrate → notify
- Telegram bot: event registration + alert delivery
- APScheduler: 6-hour polling, unattended operation
- SQLite + WAL mode, persistent on Railway
- Structured logging (structlog), PollRun audit trail
- `/health` endpoint, Poll History dashboard
- **Live at:** `https://skygent-production.up.railway.app`
- **204 unit tests, zero LLM dependencies in test suite**

---

## Session 3 — Spanish localization + Forecast Q&A

*Priority: highest. These two features make Skygent genuinely usable
for real LATAM users, which is the foundation for everything else.*

### Spanish localization

Language is a `MonitoringProfile` field. The Telegram bot detects or
asks language preference on first interaction. The narrator's system
prompt switches language. Every bot message, confirmation, and alert
narrative is delivered in the user's language.

**Why now:** Skygent's target market is LATAM. Every demo to a potential
user or investor in Uruguay, Argentina, or Brazil happens in Spanish.
English-only is a barrier that costs nothing to remove.

**Scope:**
- Add `language: str` field to `MonitoringProfile` (default: `"en"`)
- Bot asks language preference at `/start` before any other question
- Narrator system prompt includes language instruction
- All bot response strings externalized to a simple dict per language
- Start with `en` and `es`. Architecture supports adding more.

### Forecast Q&A

User sends a free-text question about their monitored event. The LLM
receives a structured payload — not raw forecast data — and answers
grounded in what the engine has already computed.

**Why now:** the `poll_runs` audit table built in v0.2.0 is the context
source. The architecture is already designed for this. It's the most
differentiated feature Skygent has relative to generic weather apps.

**Scope:**
- New Telegram command: `/forecast` or free-text trigger
- Bot assembles payload: current snapshot + last N poll_runs + profile metadata
- LLM answers in user's language, constrained to structured data
- LLM never fetches or evaluates forecast data — narration only
- First version: user-initiated. Second: proactive after N quiet polls.

---

## Session 4 — Threshold customization

*Priority: high. Unlocks agricultural and logistics use cases.*

Guided threshold configuration via Telegram bot conversation — not a
free-form settings screen. The bot asks context-aware questions based
on event type and builds the threshold config from answers.

**Examples:**
- Social event outdoors → asks about rain probability tolerance, wind
- Agriculture → asks about frost threshold (heladas), rain during harvest
- Logistics → asks about wind limits for outdoor installation

**Key insight:** threshold customization is a registration-time decision,
not a settings panel. The bot conversation is the UX.

**Scope:**
- Extend `MonitoringProfile` threshold fields (already partially present)
- Add threshold customization step to bot registration flow
- Significance evaluator reads per-profile thresholds (already designed)
- Frost risk (`temperature_2m_min` ≤ 0°C) as first agricultural preset

---

## Session 5 — Alert quality + post-event feedback

*Priority: medium. Builds the calibration dataset needed for S2S.*

After a monitored event passes, Skygent asks: "Did the conditions
change the way we predicted?" Simple yes/no feedback per alert.

This closes the loop between signal and outcome — essential for
improving thresholds and building trust with agricultural users.

**Scope:**
- Post-event Telegram message: "Your event passed. Were our alerts useful?"
- Store feedback in `alert_feedback` table
- Dashboard panel: alert accuracy over time
- Use feedback to suggest threshold adjustments (manual first, automated later)

---

## Session 6 — S2S foundation (sub-seasonal to seasonal)

*Priority: medium-high for agriculture vertical. Build after real
agricultural users are using the current system.*

S2S operates at 2–6 week horizon using ensemble statistics, not
deterministic forecasts. The question shifts from "will it rain on
May 25" to "is the week of June 10 anomalously wet relative to
climatology."

**Why this is a startup idea:**
The competitive moat is the combination of deterministic architecture
(traceable decisions), S2S ensemble reasoning, and domain-specific
context (crop types, machinery windows, frost risk). No existing LATAM
agtech weather product does this with full auditability.

**Target use case:**
> "In June, what's the best week to hire machinery for grain harvest?"
→ Which week has lowest ensemble rain probability + wind below threshold,
weighted by forecast skill at that horizon.

**Data sources:**
- Open-Meteo seasonal models (ECMWF SEAS5, GFS Seasonal)
- ECMWF S2S database (requires separate access)
- Climatological baselines for anomaly detection

**Scope (when ready):**
- New `ForecastHorizon` enum: `deterministic` (0–15d) vs `seasonal` (2–6w)
- S2S fetcher alongside existing Open-Meteo client
- Ensemble spread as a first-class signal (not just mean forecast)
- New significance evaluator for probabilistic signals
- New narrator prompt for uncertainty communication
- Bot command: `/plan [month]` for week-level planning queries

**Prerequisite:** 3–5 real agricultural users with 1+ month of usage.
Their feedback defines what S2S questions actually matter.

---

## Ongoing / cross-cutting

These improvements apply across all sessions:

**Alert history digest** — monthly Telegram summary: N checks, M alerts,
top changed variables. Builds user trust, zero new architecture.

**Multi-event view** — profile-centric Streamlit dashboard for users
monitoring multiple events simultaneously.

**Webhook delivery** — alongside Telegram, support a webhook endpoint
for developer integrations. Opens B2B use case.

**Uptime monitoring** — Better Stack or UptimeRobot on `/health`,
alerting via Telegram on downtime.

---

## What is explicitly NOT on this roadmap

- PostgreSQL migration (unnecessary at current scale)
- Redis or external queue
- User authentication / account management
- React frontend rewrite
- Multi-model LLM routing
- Docker Compose local setup

These are revisited only when scale or specific user requirements
make them necessary. Not before.

---

## One-sentence startup pitch (S2S direction)

> "Skygent is a deterministic weather decision engine for LATAM
> agriculture — it tells farmers not just what the forecast says,
> but when the forecast changed in a way that requires action,
> with full auditability from signal to alert."

True today for 0–15 days. True for planting and harvest planning
windows once S2S is added.
