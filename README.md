# Skygent — AI Weather Monitoring Agent

Skygent watches weather forecasts for user-defined events and sends proactive alerts when conditions change significantly. Given a wedding, outdoor concert, or farm harvest window, it polls the forecast every few hours, detects meaningful changes, and delivers a natural-language summary that includes an honest estimate of forecast uncertainty.

Built as a portfolio project for a senior AI/ML engineering role and as the technical foundation for a future B2B pivot into agriculture, logistics, and energy.

---

## How it works

Every N hours (default: 6), the scheduler wakes up for each active event profile and runs a five-step pipeline:

```
fetch_forecast → analyze_diff → evaluate_significance → narrate → notify
```

Most runs exit after `evaluate_significance` with no alert — the LLM is called only when a threshold is crossed. On a typical 6-hour cycle this means Claude is invoked perhaps once or twice per week per profile, not on every poll.

The core design principle is **hybrid intelligence**: deterministic code makes all threshold and significance decisions; the LLM is called exactly once per alert cycle, only to write the human-readable message.

---

## Stack

| Component | Role |
|---|---|
| LangGraph + Claude Sonnet | Agent graph + LLM narrator |
| Open-Meteo API | Free weather data, no auth required |
| FastAPI + SQLModel | HTTP backend *(Step 5 — in progress)* |
| APScheduler | Polling scheduler |
| Telegram Bot API | Alert delivery *(Step 6 — in progress)* |
| Streamlit | Dashboard *(Step 7 — in progress)* |
| Pydantic v2 | Data contracts throughout |

---

## Project status

| Step | Component | Status |
|---|---|---|
| 1 | Core models, diff engine, significance evaluator | ✅ Complete |
| 2 | Open-Meteo integration | ✅ Complete |
| 3 | LangGraph agent (graph, nodes, state) | ✅ Complete |
| 4 | APScheduler polling loop | ✅ Complete |
| 5 | FastAPI routes + DB persistence | 🔄 Next |
| 6 | Telegram notifications | ⬜ Planned |
| 7 | Streamlit dashboard | ⬜ Planned |

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
│   │   └── openmeteo.py      # Async Open-Meteo API client
│   ├── agent/
│   │   ├── state.py          # AgentState TypedDict
│   │   ├── nodes.py          # fetch, diff, significance, narrate, notify nodes
│   │   └── graph.py          # LangGraph graph + run_agent() entry point
│   └── scheduler/
│       └── jobs.py           # APScheduler jobs, SnapshotStore
├── tests/
│   ├── test_diff.py          # 20 tests
│   ├── test_significance.py  # 34 tests
│   ├── test_openmeteo.py     # 30 tests
│   ├── test_agent.py         # 37 tests
│   └── test_scheduler.py     # 29 tests
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

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running the tests

```bash
# All unit tests (no API key, no network required)
pytest tests/ -v

# Live Open-Meteo API call
pytest -m integration tests/test_openmeteo.py -v

# Full agent run with real LLM (requires ANTHROPIC_API_KEY)
pytest -m integration tests/test_agent.py -v
```

148 unit tests pass with zero external dependencies. 2 integration tests are deselected by default.

---

## Quick example

```python
import asyncio
from datetime import datetime, timezone
from skygent.core.models import MonitoringProfile
from skygent.agent.graph import run_agent

profile = MonitoringProfile(
    name="Ana & Juan's Wedding",
    location=(-34.9011, -56.1645),  # Montevideo
    event_datetime=datetime(2025, 9, 15, 17, 0, tzinfo=timezone.utc),
    monitoring_start=datetime(2025, 9, 1, tzinfo=timezone.utc),
)

# First run — fetches baseline snapshot
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
