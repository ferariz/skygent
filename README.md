# Skygent — Decision Engine for Weather Change Detection

Skygent is a deterministic decision engine for detecting meaningful changes in numerical weather prediction systems.

It is based on a simple principle:

> **Significant change is a mathematical concept — not a heuristic, and not something to delegate to an LLM.**

Most weather applications continuously display forecast data.  
Skygent does something different: it decides **when a forecast actually changed in a way that matters**, and communicates that change clearly. This problem becomes critical in domains like agriculture, energy, and logistics, where decisions depend on detecting meaningful changes early.

---

## Why this exists

Modern numerical weather prediction (NWP) systems like ECMWF IFS produce extremely high-quality forecasts:
- ~9 km spatial resolution  
- global coverage  
- updated every 6 hours  

The problem is not the data.

The problem is **attention**.

Users don’t need more forecasts.  
They need to know:
- *Did something actually change?*
- *Is it meaningful?*
- *How confident should I be?*

Skygent is built to answer exactly those questions.

---

## Core design principle

Skygent follows a strict separation:

> **Deterministic code owns decisions. LLMs own communication.**

This leads to three non-negotiable rules:

1. **All significance logic is deterministic and testable**
2. **LLMs are never used to evaluate thresholds or make decisions**
3. **LLMs are invoked only when there is information worth communicating**

If you can write an `assert` for it, it does not belong in an LLM.

---

## How it works

Every N hours (default: 6), Skygent evaluates each monitored event:

```
fetch_forecast → analyze_diff → evaluate_significance → narrate → notify
```


Most runs exit early:
- No meaningful change → no alert → no LLM call

In practice:
- ~28 forecast polls per week  
- ~1–2 LLM calls per week per profile  

This is **event-driven AI**, not polling-based generation.

---

## What “significant change” means

A change is not defined by raw deltas alone.

Skygent evaluates:
- Magnitude of change (e.g. precipitation probability shift)
- Directionality (improvement vs deterioration)
- Context (event sensitivity)
- Forecast horizon

This is closer to **regime change detection** than simple thresholding.

---

## Confidence is not a guess

Skygent maps forecast horizon to a confidence label using known NWP skill limits:

| Horizon | Confidence |
|--------|------------|
| ≤ 3 days | High |
| 3–7 days | Medium |
| > 7 days | Low |

This reflects the known behavior of global NWP systems:
- deterministic skill degrades with lead time  
- uncertainty grows non-linearly  

These are **physics-informed heuristics**, not calibrated probabilities.

---

## Forecast data sources

Skygent uses [Open-Meteo](https://open-meteo.com/), which aggregates multiple NWP models:

| Model | Provider | Resolution | Forecast |
|------|--------|-----------|----------|
| ECMWF IFS HRES | ECMWF | 9 km | 15 days |
| GFS | NOAA | 12–25 km | 16 days |
| ICON | DWD | 2–11 km | 7.5 days |
| GEM | Environment Canada | 2.5–15 km | 10 days |

For South America (e.g. Montevideo), forecasts are primarily driven by:
- ECMWF IFS  
- GFS  

This implies:
- strong synoptic skill  
- limited convective resolution  

---

## Architecture

Skygent implements a hybrid system:

| Layer | Responsibility |
|------|----------------|
| Diff engine | Computes changes between forecast snapshots |
| Significance evaluator | Applies deterministic rules |
| State store | Tracks historical forecasts |
| LLM narrator | Converts structured alerts into natural language |
| Delivery | Sends alerts (Telegram, API, UI) |

The LLM is **never in the critical path of decision-making**.

---

## Example behavior

Instead of:

> “Forecast updated: chance of rain is now 40%”

Skygent produces:

> “Rain risk for your event increased from low to moderate (20% → 45%).  
Confidence is medium due to forecast horizon (~5 days).  
Conditions have deteriorated compared to the previous forecast.”

---

## Economics of the design

Naive agent:
- LLM call every poll → ~28 calls/week  

Skygent:
- LLM call only on change → ~1–2 calls/week  

Result:
- **~90%+ reduction in token usage**
- deterministic testability  
- full auditability  

---

## When this pattern breaks

This approach does not apply when:
- input is unstructured  
- decision boundaries are ambiguous  
- reasoning is open-ended  

In those cases:
- ReAct  
- Plan-and-Execute  
- LLM-first systems  

Skygent is intentionally **not that system**.

---

## Project status

**Live in production** — `https://skygent-production.up.railway.app`

| Step | Component | Status |
|---|---|---|
| 1 | Core models, diff engine, significance evaluator | ✅ Complete |
| 2 | Open-Meteo integration | ✅ Complete |
| 3 | LangGraph agent (graph, nodes, state) | ✅ Complete |
| 4 | APScheduler polling loop | ✅ Complete |
| 5 | FastAPI routes + DB persistence | ✅ Complete |
| 6 | Telegram notifications | ✅ Complete |
| 7 | Streamlit dashboard | ✅ Complete |
| 8 | Railway deployment | ✅ Live |

---

## Repository structure

```
skygent/
├── skygent/
│   ├── core/
│   │   ├── models.py         # MonitoringProfile, ForecastSnapshot, Alert
│   │   ├── diff.py           # DiffAnalyzer — delta computation between snapshots
│   │   └── significance.py   # SignificanceEvaluator — threshold rules, confidence scoring
│   ├── integrations/
│   │   ├── openmeteo.py      # Async Open-Meteo API client
│   │   └── telegram.py       # Telegram Bot API notification sender
│   ├── agent/
│   │   ├── state.py          # AgentState TypedDict
│   │   ├── nodes.py          # fetch, diff, significance, narrate, notify nodes
│   │   └── graph.py          # LangGraph graph + run_agent() entry point
│   ├── scheduler/
│   │   └── jobs.py           # APScheduler jobs, SnapshotStore
│   └── api/
│       ├── database.py       # SQLModel DB models + CRUD helpers
│       ├── routes.py         # FastAPI route handlers
│       └── main.py           # App entry point + lifespan
├── ui/
│   └── app.py                # Streamlit dashboard
├── tests/
│   ├── test_diff.py          # 20 tests
│   ├── test_significance.py  # 34 tests
│   ├── test_openmeteo.py     # 30 tests
│   ├── test_agent.py         # 37 tests
│   ├── test_scheduler.py     # 29 tests
│   ├── test_api.py           # 42 tests
│   └── test_telegram.py      # 21 tests
├── docs/
│   └── architecture.md       # Architecture and design decisions
├── requirements.txt
└── pytest.ini
```


---

## Setup

```bash
git clone https://github.com/ferariz/skygent.git
cd skygent

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set credentials:

```bash
export OPENAI_API_KEY=sk-...
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...
export SKYGENT_API_URL=http://localhost:8000
```

---

## Running the stack

```bash
# Production — single process (API + Telegram bot)
python -m skygent.launcher

# Development — API server
uvicorn skygent.api.main:app --port 8000 --reload

# Development — Dashboard (separate terminal)
streamlit run ui/app.py
```

Open `http://localhost:8501` to register events and browse alerts.

---

## Running the tests

```bash
# All unit tests (no API key, no network required)
pytest tests/ -v

# Live Open-Meteo API call
pytest -m integration tests/test_openmeteo.py -v

# Full agent run with real LLM (requires OPENAI_API_KEY)
pytest -m integration tests/test_agent.py -v

# Real Telegram delivery (requires TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
pytest -m integration tests/test_telegram.py -v
```

204 unit tests pass with zero external dependencies. 3 integration tests are deselected by default.

---

## Quick example

```python
import asyncio
from datetime import datetime, timezone
from skygent.core.models import MonitoringProfile
from skygent.agent.graph import run_agent

profile = MonitoringProfile(
    name="Ana & Juan's Wedding",
    location=(-34.9011, -56.1645),  # Montevideo, Uruguay
    event_datetime=datetime(2025, 9, 15, 17, 0, tzinfo=timezone.utc),
    monitoring_start=datetime(2025, 9, 1, tzinfo=timezone.utc),
)

# First run — fetches baseline snapshot from ECMWF IFS / GFS
state = asyncio.run(run_agent(profile, previous_snapshot=None))
print(state["current_snapshot"].data)
```

---

## Real alert example

**Previous forecast**
- Precipitation probability: 20%
- Confidence: Medium (5-day horizon)

**New forecast**
- Precipitation probability: 45%
- Confidence: Medium

**Detected change**
- +25pp increase in precipitation probability
- Threshold exceeded → alert triggered

**Generated alert**
> Rain risk for your event increased significantly (20% → 45%).  
> Confidence remains medium due to forecast horizon (~5 days).  
> Conditions have deteriorated compared to the previous forecast.

---

## Design documentation

Architecture decisions, tradeoffs, and the rationale behind non-obvious choices are documented in [`docs/architecture.md`](docs/architecture.md).

---

## Vertical extensibility

`MonitoringProfile` has a `context` field (`social_event` | `agriculture` | `energy` | `logistics`) and fully configurable per-variable thresholds. The same agent can monitor soil moisture thresholds for a farm or wind speed limits for a wind farm without code changes — only the profile configuration differs.

---

## License

MIT — see [LICENSE](LICENSE)

Note: This repository is open for learning and experimentation.
Commercial deployments may be subject to additional licensing in the future.
