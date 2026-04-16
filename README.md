# Skygent — AI Weather Monitoring Agent

Agentic weather monitoring system that tracks forecast changes for user-defined events and proactively notifies users when significant changes occur.

## Stack
- LangGraph + Claude (Anthropic) — agent framework and LLM narrator
- Open-Meteo — free weather API, no auth required
- FastAPI + SQLite — backend
- APScheduler — polling scheduler
- Telegram — notifications
- Streamlit — dashboard

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
