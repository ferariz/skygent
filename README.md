# Skygent — AI Weather Monitoring Agent

Skygent watches weather forecasts for user-defined events and sends proactive alerts when conditions change significantly. Given a wedding, outdoor concert, or farm harvest window, it polls the forecast every few hours, detects meaningful changes, and delivers a natural-language summary that includes an honest estimate of forecast uncertainty.

Built as a portfolio project for a senior AI/ML engineering role and as the technical foundation for a future B2B pivot into agriculture, logistics, and energy.

---

## How it works

Every N hours (default: 6), the scheduler wakes up for each active event profile and runs a five-step pipeline:

```
fetch_forecast → analyze_diff → evaluate_significance → narrate → notify
```

Most runs exit after `evaluate_significance` with no alert — the LLM is called only when a threshold is crossed. On a typical 6-hour cycle this means the model is invoked perhaps once or twice per week per profile, not on every poll.

The core design principle is **hybrid intelligence**: deterministic code makes all threshold and significance decisions; the LLM is called exactly once per alert cycle, only to write the human-readable message.

---

## Forecast data sources

Skygent uses [Open-Meteo](https://open-meteo.com/) (free, no API key required), which blends output from multiple national weather services into a single endpoint. For any given location, the API automatically selects the best available model — a strategy called **Best Match**.

### Contributing NWP models

| Model | Provider | Spatial resolution | Temporal resolution | Forecast length | Update cycle |
|---|---|---|---|---|---|
| ECMWF IFS HRES | ECMWF | 9 km (O1280 Gaussian grid) | 1-hourly (0–90 h), 3-hourly (90–144 h), 6-hourly (>144 h) | 15 days | Every 6 h |
| ECMWF AIFS | ECMWF | ~28 km | 6-hourly | 15 days | Every 6 h |
| NCEP GFS | NOAA | 0.11°–0.25° (~12–25 km) | 1-hourly | 16 days | Every hour |
| DWD ICON | DWD (Germany) | 2–11 km | 1-hourly | 7.5 days | Every 3 h |
| GEM | Environment Canada | 2.5–15 km | 1-hourly | 10 days | Every 6 h |

ECMWF transitioned to open data on 1 October 2025 (CC-BY 4.0 licence), giving Open-Meteo access to the full-resolution IFS HRES output at 9 km without additional delay.

### What Skygent currently consumes

We request **daily aggregates** via the `daily=` parameter: `precipitation_probability_max`, `temperature_2m_max`, `wind_speed_10m_max`, `weather_code`. These are computed by Open-Meteo by aggregating over the underlying 1-hourly or 3-hourly NWP output for each UTC calendar day.

This is a deliberate MVP simplification. See [docs/design.md](docs/design.md) for the documented upgrade path to hourly event-window extraction using the `event_duration_hours` field already present on `MonitoringProfile`.

### Confidence scoring and NWP skill horizon

Skygent maps forecast horizon to a confidence label using established NWP skill thresholds:

| Horizon | Confidence | Basis |
|---|---|---|
| ≤ 3 days | High | Deterministic NWP models are reliable at this range |
| 3–7 days | Medium | Ensemble spread grows; synoptic-scale patterns are still captured but mesoscale detail degrades |
| > 7 days | Low | Beyond the reliable deterministic NWP horizon; large-scale anomalies may be captured but specific event forecasts carry high uncertainty |

These thresholds reflect the typical skill horizon of global NWP models like ECMWF IFS. They are not derived from Skygent's own verification and should be treated as heuristics, not calibrated probabilities.

### Regional note for South America

Open-Meteo does not currently include a high-resolution regional model for South America (unlike Europe, where ICON-D2 at 2 km or HARMONIE at 2 km are available, or North America, where HRRR at 3 km is available). For locations like Montevideo, Best Match relies primarily on **ECMWF IFS HRES (9 km)** and **NCEP GFS** as global models. The effective spatial resolution is therefore 9–25 km, which is adequate for synoptic-scale and daily aggregate variables but insufficient for resolving convective-scale events or terrain-induced phenomena.

### Roadmap: ensemble uncertainty and ML

The current confidence scoring is a deterministic rule (`horizon → label`). Two natural upgrades:

1. **Ensemble spread from Open-Meteo's Ensemble API** — returns output from 50 ECMWF ENS members plus NCEP GEFS, giving a quantitative spread estimate per variable. This would replace the horizon heuristic with an actual probabilistic uncertainty estimate for each snapshot.

2. **ML post-processing** — Model Output Statistics (MOS) or analog-based methods could be applied to the raw NWP output to produce calibrated probabilities, particularly for precipitation. This is a standard technique in operational meteorology that would fit naturally into the `analyze_diff` → `evaluate_significance` pipeline.

---

## Stack

| Component | Role |
|---|---|
| LangGraph + GPT-4o-mini (OpenAI) | Agent graph + LLM narrator |
| Open-Meteo API | NWP forecast data — ECMWF IFS, GFS, and others |
| FastAPI + SQLModel + SQLite | HTTP backend + persistence |
| APScheduler | Polling scheduler |
| Telegram Bot API | Alert delivery |
| Streamlit | Dashboard |
| Pydantic v2 | Data contracts throughout |

---

## Project status

| Step | Component | Status |
|---|---|---|
| 1 | Core models, diff engine, significance evaluator | ✅ Complete |
| 2 | Open-Meteo integration | ✅ Complete |
| 3 | LangGraph agent (graph, nodes, state) | ✅ Complete |
| 4 | APScheduler polling loop | ✅ Complete |
| 5 | FastAPI routes + DB persistence | ✅ Complete |
| 6 | Telegram notifications | ✅ Complete |
| 7 | Streamlit dashboard | ✅ Complete |

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
│   └── design.md             # Architecture and design decisions
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
```

---

## Running the stack

```bash
# Terminal 1 — API server
uvicorn skygent.api.main:app --port 8000 --reload

# Terminal 2 — Dashboard
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

## Design documentation

Architecture decisions, tradeoffs, and the rationale behind non-obvious choices are documented in [`docs/design.md`](docs/design.md).

---

## Vertical extensibility

`MonitoringProfile` has a `context` field (`social_event` | `agriculture` | `energy` | `logistics`) and fully configurable per-variable thresholds. The same agent can monitor soil moisture thresholds for a farm or wind speed limits for a wind farm without code changes — only the profile configuration differs.

---

## License

MIT — see [LICENSE](LICENSE)