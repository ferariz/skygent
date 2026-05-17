# Contributing to Skygent

Thank you for your interest in contributing. This document covers the
architectural invariant you must understand before opening a PR, how to
set up the project locally, and the conventions we follow.

---

## The one rule that governs everything

Skygent is built on a single non-negotiable principle:

> **Deterministic code owns decisions. LLMs own communication.**

In practice this means:

- The significance evaluator decides whether a forecast change is meaningful.
  It uses explicit thresholds, computable deltas, and auditable logic.
- The LLM narrator receives a structured alert the engine has already built
  and writes a human-readable message from it.
- The LLM never sees raw forecast data and decides what matters.
  The LLM never sets or evaluates thresholds.
  The LLM never determines whether an alert should fire.

A useful test before opening any PR: **if you can write an `assert` for it,
it does not belong in an LLM call.**

This constraint is not a limitation — it is the reason the system is testable,
auditable, and reliable. All 204 unit tests pass with zero LLM dependencies.
If your change requires mocking an LLM to test a decision, the decision is
in the wrong layer.

---

## Local setup

**Prerequisites:** Python 3.11+, a virtual environment manager.

```bash
git clone https://github.com/ferariz/skygent.git
cd skygent

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Environment variables** (only needed for integration tests and live runs):

```bash
export OPENAI_API_KEY=sk-...
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...
export ENV=development          # enables colored console logging
```

**Running the stack locally:**

```bash
# Terminal 1 — API server + scheduler
uvicorn skygent.api.main:app --port 8000 --reload

# Terminal 2 — Telegram bot
python -m skygent.bot

# Terminal 3 — Dashboard
streamlit run ui/app.py
```

Open `http://localhost:8501` for the dashboard and `http://localhost:8000/health`
to confirm the scheduler is running.

---

## Running the tests

```bash
# Full unit test suite — no API key, no network, no LLM required
pytest tests/ -v

# Live Open-Meteo API call (requires network)
pytest -m integration tests/test_openmeteo.py -v

# Full agent run with real LLM (requires OPENAI_API_KEY)
pytest -m integration tests/test_agent.py -v

# Real Telegram delivery (requires TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
pytest -m integration tests/test_telegram.py -v
```

**The rule:** all 204 unit tests must pass before any PR is merged.
Integration tests are deselected by default and are never required in CI —
they exist for manual validation before production deployments.

---

## Branch strategy

```
main                — always deployable, tagged at each milestone
feat/<name>         — new features (e.g. feat/forecast-qa)
fix/<name>          — bug fixes
chore/<name>        — infrastructure, dependencies, documentation
```

Branch from `main`. Open a PR against `main`. `main` is tagged at each
meaningful milestone using semantic versioning (`v0.1.0`, `v0.2.0`, ...).

---

## Commit message convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(scheduler): add poll_runs audit table with finally-block write
fix(nodes): correct field name in Open-Meteo response parsing
chore(logging): replace stdlib logger with structlog across all modules
docs(architecture): document significance evaluator design decisions
```

Scope is the module or subsystem affected: `db`, `scheduler`, `nodes`,
`logging`, `resilience`, `observability`, `dashboard`, `bot`, `api`.

---

## What not to contribute (right now)

The following are explicitly out of scope for this stage of the project.
PRs in these areas will be closed without review:

- PostgreSQL migration — SQLite with WAL mode is correct for the current scale
- Redis or any external queue — APScheduler with SQLite job store is sufficient
- User authentication or account management
- React or any frontend rewrite of the Streamlit dashboard
- Multi-model LLM routing or provider abstraction
- Docker Compose multi-container local setup

If you believe one of these is genuinely necessary, open an issue first to
discuss the tradeoff before writing any code.

---

## Adding a new delivery channel

The `notify_node` in `skygent/agent/nodes.py` routes on
`profile.notification_channel`. Adding a new channel (email, Slack, webhook)
means:

1. Implementing the delivery function in `skygent/integrations/`
2. Adding a branch in `notify_node`
3. Writing unit tests that mock the delivery function — not the node logic

The node interface and graph are unchanged. The LLM narration layer is
unchanged. Only the delivery implementation is new.

---

## Questions

Open an issue with the `question` label. Design discussions belong in issues
before code is written.
