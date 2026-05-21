"""
skygent/integrations/telegram_bot.py — Inbound Telegram bot handler
=====================================================================

This module implements the registration conversation flow for the Skygent
Telegram bot. Users interact with the bot to register a new event for
weather monitoring without needing to use the Streamlit dashboard or API.

Design decisions
----------------
1. Polling over webhooks: polling (getUpdates long-polling) requires no
   public URL and works immediately in local development. The bot calls
   getUpdates every few seconds and processes incoming messages. Webhooks
   are more efficient at scale but require a publicly reachable HTTPS
   endpoint. For MVP local deployment, polling is the right choice. The
   switch to webhooks when deploying to Railway/Render is a one-line change
   (replace the polling loop with a FastAPI endpoint that Telegram calls).

2. SQLite-backed conversation state over in-memory: conversation state
   survives bot restarts and WatchFiles-triggered reloads during development.
   The existing SQLModel/SQLite stack is already in place — adding one table
   costs almost nothing. In-memory state would be lost on every file save
   during development, making the bot untestable in practice.

3. State machine with explicit steps: the conversation is modelled as a
   finite state machine with named steps (IDLE, ASK_DATE, ASK_TIME, etc.).
   Each incoming message is routed to the handler for the current step.
   This makes the flow easy to extend (add a step, add a handler) and easy
   to test (each handler is a pure function of message text + current data).

4. dateparser for natural date input: users should be able to type
   "September 15", "15/09/2026", or "2026-09-15" and have it work.
   dateparser handles all of these without custom parsing logic.
   We always coerce to UTC to stay consistent with the rest of the stack.

5. Telegram keyboard buttons for structured choices: context and duration
   are constrained choices — we present them as inline keyboard buttons
   rather than free-text input. This eliminates an entire class of parsing
   errors and makes the conversation feel like a proper mobile app.

6. telegram_chat_id on MonitoringProfile: when a profile is registered via
   the bot, its telegram_chat_id is set to the user's chat ID. The notify_node
   routes alerts to this chat directly, enabling per-user delivery without
   a shared TELEGRAM_CHAT_ID env var. Profiles registered via the dashboard
   or API still use the env var.

7. Welcome forecast on registration: immediately after the profile is
   registered, the bot fetches the current forecast and sends a GPT-4o-mini
   narrative framed as an introduction — "here is what the weather looks like
   right now for your event" — rather than an alert. A different system prompt
   produces a warmer, more informational tone than the change-detection alerts.

8. Stale conversation expiry: if a conversation has been inactive for more
   than CONVERSATION_EXPIRY_HOURS, the state is cleared and the user is asked
   to start over. This prevents confusing state from a conversation abandoned
   days ago being resumed unexpectedly.

9. /cancel command at any step: users can always type /cancel to abort the
   current registration and start fresh. This is the escape hatch for users
   who made a mistake earlier in the flow.

10. One module, no external bot library: python-telegram-bot and aiogram
    are excellent libraries but add significant complexity (event loops,
    dispatcher patterns, middleware). Since we only need a small subset of
    the Telegram Bot API (getUpdates, sendMessage, answerCallbackQuery), a
    direct httpx implementation keeps dependencies minimal and the code
    auditable.

Conversation flow
-----------------
IDLE
  /start or any message → explain what the bot does, ask for location pin

ASK_LOCATION
  location message → save lat/lon, ask for event name

ASK_NAME
  text → save name, ask for event date

ASK_DATE
  text → parse with dateparser, ask for event time (UTC)

ASK_TIME
  text → parse time, ask for context (keyboard)

ASK_CONTEXT
  callback_query → save context, ask for duration (keyboard)

ASK_DURATION
  callback_query → save duration, show summary and ask to confirm

CONFIRM
  "Yes" callback → register profile via API, fetch forecast, send welcome
  "No" callback  → cancel, back to IDLE

Running the bot
---------------
    python -m skygent.bot

Or as part of a supervisor / process manager alongside uvicorn.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any

import dateparser
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from skygent.api.database import (
    clear_conversation_state,
    get_conversation_state,
    get_profiles_by_chat_id,
    get_recent_poll_runs,
    get_session_sync,
    load_latest_snapshot,
    save_conversation_state,
)
from skygent.core.models import MonitoringProfile, ForecastSnapshot
from skygent.integrations.openmeteo import fetch_forecast
from skygent.integrations.telegram import TelegramError, _escape

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TELEGRAM_API_BASE = "https://api.telegram.org"
POLL_TIMEOUT = 30            # long-poll timeout in seconds
CONVERSATION_EXPIRY_HOURS = 24  # clear stale state after this many hours


# ---------------------------------------------------------------------------
# Conversation steps
# ---------------------------------------------------------------------------

class Step:
    """Named constants for conversation state machine steps."""
    IDLE         = "IDLE"
    ASK_LANGUAGE = "ASK_LANGUAGE"
    ASK_LOCATION = "ASK_LOCATION"
    ASK_NAME     = "ASK_NAME"
    ASK_DATE     = "ASK_DATE"
    ASK_TIME     = "ASK_TIME"
    ASK_CONTEXT  = "ASK_CONTEXT"
    ASK_DURATION = "ASK_DURATION"
    CONFIRM      = "CONFIRM"


# ---------------------------------------------------------------------------
# Strings — all user-facing registration flow text, bilingual
# ---------------------------------------------------------------------------

STRINGS: dict[str, dict[str, str]] = {
    'en': {
        'welcome': (
            "<b>👋 Welcome to Skygent!</b>\n\n"
            "I monitor the weather forecast for your events and alert you "
            "when conditions change significantly.\n\n"
            "First, choose your preferred language:"
        ),
        'ask_location': (
            "To get started, <b>send me your event location</b> as a "
            "Telegram location pin 📍\n\n"
            "Tap the 📎 paperclip → Location, then drop a pin on the map."
        ),
        'location_missing': (
            "I need a location pin 📍\n\n"
            "Tap the 📎 paperclip → Location, then drop a pin on the map."
        ),
        'location_received': "📍 Got it — <b>{lat}, {lon}</b>\n\nWhat should I call this event?\n<i>e.g. Ana & Juan's Wedding, Harvest Day, Wind Farm Inspection</i>",
        'ask_name_missing': "Please send the event name as a text message.",
        'name_confirmed': "Great — <b>{name}</b> ✓\n\nWhat date is the event?\n<i>e.g. September 15, 2026 / 2026-09-15 / 15/09/2026</i>",
        'ask_date_missing': "Please send the event date as a text message.",
        'date_unparseable': (
            "I couldn't understand that date 🤔\n\n"
            "Please try a format like:\n"
            "• <b>September 15, 2026</b>\n"
            "• <b>2026-09-15</b>\n"
            "• <b>15/09/2026</b>"
        ),
        'date_in_past': "That date is in the past. Please send a future event date.",
        'date_confirmed': "📅 <b>{date}</b> ✓\n\nWhat time does the event start? <i>(UTC)</i>\n<i>e.g. 17:00 / 5pm / 14:30</i>\n\nAll times are in UTC. Montevideo is UTC-3, so 5pm local = 20:00 UTC.",
        'ask_time_missing': "Please send the event time as a text message.",
        'time_unparseable': (
            "I couldn't understand that time 🤔\n\n"
            "Please try a format like:\n"
            "• <b>17:00</b>\n"
            "• <b>5pm</b>\n"
            "• <b>14:30</b>"
        ),
        'time_confirmed': "⏰ <b>{time}</b> ✓\n\nWhat type of event is this?",
        'ask_duration': "How long does the event last?",
        'confirm_header': "<b>Please confirm your event:</b>",
        'confirm_body': (
            "📍 <b>Name:</b> {name}\n"
            "🗺 <b>Location:</b> {lat}, {lon}\n"
            "📅 <b>Date:</b> {date}\n"
            "⏱ <b>Duration:</b> {duration} hours\n"
            "🏷 <b>Context:</b> {context}\n\n"
            "Skygent will check the forecast every <b>{interval}h</b> and alert you when conditions change significantly."
        ),
        'cancelled': "Registration cancelled. Send /start whenever you're ready to try again.",
        'cancel_any': "Cancelled. Send /start to begin again.",
        'registering': "⏳ Registering your event and fetching the first forecast...",
        'registration_failed': (
            "❌ Something went wrong registering your event. "
            "Please try again or contact support."
        ),
        'use_buttons': "Please use the buttons above to continue, or send /cancel to start over.",
        'no_active_events': "You have no active events being monitored. Send /start to register one.",
        'beyond_forecast_window': (
            "✅ <b>{name}</b> is now being monitored!\n\n"
            "📅 Your event is <b>{days:.0f} days away</b> — "
            "weather forecasts are only available up to 16 days out, "
            "so I don't have data yet.\n\n"
            "I'll check every <b>{interval}h</b> "
            "and send you the first forecast as soon as it becomes available "
            "(roughly {wait:.0f} days from now). "
            "I'll also alert you whenever the forecast changes significantly."
        ),
        'welcome_fallback': (
            "✅ <b>{name}</b> is now being monitored!\n\n"
            "I'll send you updates when the forecast changes significantly."
        ),
    },
    'es': {
        'welcome': (
            "<b>👋 ¡Bienvenido a Skygent!</b>\n\n"
            "Monitoreo el pronóstico del tiempo para tus eventos y te aviso "
            "cuando las condiciones cambien significativamente.\n\n"
            "Primero, elige tu idioma preferido:"
        ),
        'ask_location': (
            "Para comenzar, <b>envíame la ubicación de tu evento</b> como un "
            "pin de ubicación de Telegram 📍\n\n"
            "Toca el 📎 clip → Ubicación y coloca un pin en el mapa."
        ),
        'location_missing': (
            "Necesito un pin de ubicación 📍\n\n"
            "Toca el 📎 clip → Ubicación y coloca un pin en el mapa."
        ),
        'location_received': "📍 Listo — <b>{lat}, {lon}</b>\n\n¿Cómo se llama este evento?\n<i>ej. Boda de Ana y Juan, Día de cosecha, Inspección del parque eólico</i>",
        'ask_name_missing': "Por favor envía el nombre del evento como mensaje de texto.",
        'name_confirmed': "Genial — <b>{name}</b> ✓\n\n¿Qué fecha es el evento?\n<i>ej. 15 de septiembre de 2026 / 2026-09-15 / 15/09/2026</i>",
        'ask_date_missing': "Por favor envía la fecha del evento como mensaje de texto.",
        'date_unparseable': (
            "No pude entender esa fecha 🤔\n\n"
            "Por favor intenta un formato como:\n"
            "• <b>15 de septiembre de 2026</b>\n"
            "• <b>2026-09-15</b>\n"
            "• <b>15/09/2026</b>"
        ),
        'date_in_past': "Esa fecha es en el pasado. Por favor envía una fecha futura.",
        'date_confirmed': "📅 <b>{date}</b> ✓\n\n¿A qué hora comienza el evento? <i>(UTC)</i>\n<i>ej. 17:00 / 5pm / 14:30</i>\n\nTodas las horas son UTC. Montevideo es UTC-3, así que las 5pm locales = 20:00 UTC.",
        'ask_time_missing': "Por favor envía la hora del evento como mensaje de texto.",
        'time_unparseable': (
            "No pude entender esa hora 🤔\n\n"
            "Por favor intenta un formato como:\n"
            "• <b>17:00</b>\n"
            "• <b>5pm</b>\n"
            "• <b>14:30</b>"
        ),
        'time_confirmed': "⏰ <b>{time}</b> ✓\n\n¿Qué tipo de evento es este?",
        'ask_duration': "¿Cuánto dura el evento?",
        'confirm_header': "<b>Por favor confirma tu evento:</b>",
        'confirm_body': (
            "📍 <b>Nombre:</b> {name}\n"
            "🗺 <b>Ubicación:</b> {lat}, {lon}\n"
            "📅 <b>Fecha:</b> {date}\n"
            "⏱ <b>Duración:</b> {duration} horas\n"
            "🏷 <b>Contexto:</b> {context}\n\n"
            "Skygent verificará el pronóstico cada <b>{interval}h</b> y te avisará cuando las condiciones cambien significativamente."
        ),
        'cancelled': "Registro cancelado. Envía /start cuando estés listo para intentarlo de nuevo.",
        'cancel_any': "Cancelado. Envía /start para empezar de nuevo.",
        'registering': "⏳ Registrando tu evento y obteniendo el primer pronóstico...",
        'registration_failed': (
            "❌ Algo salió mal al registrar tu evento. "
            "Por favor intenta de nuevo o contacta al soporte."
        ),
        'use_buttons': "Por favor usa los botones de arriba para continuar, o envía /cancel para empezar de nuevo.",
        'no_active_events': "No tienes eventos activos siendo monitoreados. Envía /start para registrar uno.",
        'beyond_forecast_window': (
            "✅ <b>{name}</b> está siendo monitoreado.\n\n"
            "📅 Tu evento está a <b>{days:.0f} días</b> — "
            "los pronósticos del tiempo solo están disponibles hasta 16 días, "
            "así que aún no tengo datos.\n\n"
            "Verificaré cada <b>{interval}h</b> "
            "y te enviaré el primer pronóstico tan pronto como esté disponible "
            "(aproximadamente en {wait:.0f} días). "
            "También te avisaré cuando el pronóstico cambie significativamente."
        ),
        'welcome_fallback': (
            "✅ <b>{name}</b> está siendo monitoreado.\n\n"
            "Te enviaré actualizaciones cuando el pronóstico cambie significativamente."
        ),
    },
}


def t(key: str, lang: str) -> str:
    """
    Return the string for `key` in the given language.
    Falls back to 'en' if the language dict is missing the key.
    This makes partial translations safe.
    """
    lang_strings = STRINGS.get(lang, STRINGS['en'])
    if key in lang_strings:
        return lang_strings[key]
    # Fall back to English rather than raising KeyError
    return STRINGS['en'].get(key, key)


# ---------------------------------------------------------------------------
# Low-level Telegram API helpers
# ---------------------------------------------------------------------------

def _token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise TelegramError("TELEGRAM_BOT_TOKEN not set")
    return token


def _url(method: str) -> str:
    return f"{TELEGRAM_API_BASE}/bot{_token()}/{method}"


def send_message(
    chat_id: str,
    text: str,
    *,
    parse_mode: str = "HTML",
    reply_markup: dict | None = None,
) -> None:
    """Send a text message to a Telegram chat (synchronous)."""
    payload: dict[str, Any] = {
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": parse_mode,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    with httpx.Client(timeout=10.0) as client:
        try:
            resp = client.post(_url("sendMessage"), json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise TelegramError(
                f"sendMessage HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise TelegramError(f"sendMessage network error: {exc}") from exc

        try:
            data = resp.json()
        except Exception as exc:
            raise TelegramError(
                f"sendMessage non-JSON response: {resp.text[:200]}"
            ) from exc

        if not data.get("ok"):
            raise TelegramError(
                f"sendMessage API error: {data.get('description', 'unknown')}"
            )


def answer_callback_query(callback_query_id: str) -> None:
    """
    Acknowledge a callback query (removes the loading indicator).
    Failures are logged but not raised — a stuck spinner is a minor UX
    issue and should not abort the handler that called us.
    """
    with httpx.Client(timeout=10.0) as client:
        try:
            resp = client.post(
                _url("answerCallbackQuery"),
                json={"callback_query_id": callback_query_id},
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                logger.warning(
                    "answerCallbackQuery non-ok: %s",
                    data.get("description", "unknown"),
                )
        except Exception as exc:
            logger.warning("answerCallbackQuery failed: %s", exc)


def get_updates(offset: int | None = None) -> list[dict]:
    """
    Long-poll getUpdates. Returns a list of update objects.
    offset = last_update_id + 1 to acknowledge processed updates.
    """
    params: dict[str, Any] = {"timeout": POLL_TIMEOUT}
    if offset is not None:
        params["offset"] = offset

    with httpx.Client(timeout=POLL_TIMEOUT + 5.0) as client:
        try:
            resp = client.get(_url("getUpdates"), params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("getUpdates HTTP %d: %s",
                         exc.response.status_code, exc.response.text[:200])
            return []
        except httpx.RequestError as exc:
            logger.error("getUpdates network error: %s", exc)
            return []

        try:
            data = resp.json()
        except Exception as exc:
            logger.error("getUpdates non-JSON response: %s", exc)
            return []

        if not data.get("ok"):
            logger.error("getUpdates API error: %s", data.get("description"))
            return []

        return data.get("result", [])


# ---------------------------------------------------------------------------
# Keyboard helpers
# ---------------------------------------------------------------------------

def _inline_keyboard(buttons: list[tuple[str, str]]) -> dict:
    """
    Build an inline keyboard reply_markup.
    buttons: list of (label, callback_data) tuples.
    Each button on its own row for clarity.
    """
    return {
        "inline_keyboard": [
            [{"text": label, "callback_data": data}]
            for label, data in buttons
        ]
    }


# ---------------------------------------------------------------------------
# Conversation state helpers
# ---------------------------------------------------------------------------

def _load_state(chat_id: str) -> tuple[str, dict]:
    """
    Load conversation step and data for a chat.
    Returns (Step.IDLE, {}) if no state exists or state is expired.

    All row attributes are read inside the session context to avoid
    SQLAlchemy 'not bound to a Session' errors on lazy attribute access
    after the session closes.
    """
    with get_session_sync() as session:
        row = get_conversation_state(session, chat_id)
        if row is None:
            return Step.IDLE, {}

        # Read all attributes while session is open
        step = row.step
        data_str = row.data
        updated_at = row.updated_at

    # Check expiry outside session — we have the values we need
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - updated_at
    if age > timedelta(hours=CONVERSATION_EXPIRY_HOURS):
        logger.info("bot: clearing expired conversation for chat %s", chat_id)
        with get_session_sync() as session:
            clear_conversation_state(session, chat_id)
        return Step.IDLE, {}

    try:
        data = json.loads(data_str)
    except Exception:
        data = {}

    # Guard: if step is unknown (e.g. from a mid-flow deploy), reset to IDLE
    VALID_STEPS = {
        Step.IDLE, Step.ASK_LANGUAGE, Step.ASK_LOCATION, Step.ASK_NAME,
        Step.ASK_DATE, Step.ASK_TIME, Step.ASK_CONTEXT, Step.ASK_DURATION,
        Step.CONFIRM,
    }
    if step not in VALID_STEPS:
        logger.warning(
            'bot: unknown step %s for chat %s — resetting to IDLE', step, chat_id
        )
        with get_session_sync() as session:
            clear_conversation_state(session, chat_id)
        return Step.IDLE, {}

    return step, data


def _save_state(chat_id: str, step: str, data: dict) -> None:
    with get_session_sync() as session:
        save_conversation_state(session, chat_id, step, data)


def _clear_state(chat_id: str) -> None:
    with get_session_sync() as session:
        clear_conversation_state(session, chat_id)


# ---------------------------------------------------------------------------
# Welcome message — initial forecast narrative
# ---------------------------------------------------------------------------

_WELCOME_BASE_SYSTEM_PROMPT = """\
You are Skygent, a friendly AI weather monitoring assistant.

A user has just registered an event for weather monitoring. Write a warm,
informative welcome message that:
1. Confirms the event details (name, date, location description)
2. Describes what the weather currently looks like for that date based on
   the forecast data provided
3. Explains what Skygent will do — check the forecast every N hours and
   notify them if conditions change significantly
4. Sets expectations: at this range the forecast has medium-to-low confidence
   and will be refined as the event approaches

Tone: friendly, clear, reassuring. Plain prose only — no markdown, no bullet
points. Under 200 words. Do not mention specific model names or API details.
"""


def _build_welcome_system_prompt(language: str) -> str:
    """Build the welcome narrative system prompt with optional Spanish instruction."""
    if language == 'es':
        return _WELCOME_BASE_SYSTEM_PROMPT + 'Respond entirely in Spanish. Do not use English.\n'
    return _WELCOME_BASE_SYSTEM_PROMPT


QA_SYSTEM_PROMPT = """\
You are a weather forecast assistant for Skygent, an AI weather monitoring service.

Rules:
- Only describe data that is present in the payload provided. Never invent values.
- Respond in the user's language (check profile.language: 'es' = Spanish, 'en' = English).
- Answer conversationally in plain prose. No markdown, no bullet points.
- Keep your response under 200 words.
- If asked something that cannot be answered from the payload data, say so honestly.
- Do not include a subject line or greeting — start directly with the information.
"""


# Module-level LLM instance — created on first use (lazy init so
# OPENAI_API_KEY is not required at import time, only when welcome runs).
_llm_instance: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(model="gpt-4o-mini", max_tokens=400)
    return _llm_instance


def _generate_welcome_narrative(
    profile: MonitoringProfile,
    snapshot: ForecastSnapshot,
) -> str:
    """
    Generate a welcome message narrative using GPT-4o-mini.
    Returns a plain-text narrative string.
    """
    from skygent.core.significance import horizon_to_confidence

    confidence = horizon_to_confidence(snapshot.horizon_days)

    payload = {
        "event_name":            profile.name,
        "event_datetime_utc":    profile.event_datetime.isoformat(),
        "horizon_days":          round(snapshot.horizon_days, 1),
        "confidence":            confidence,
        "check_interval_hours":  profile.check_interval_hours,
        "context":               profile.context,
        "current_forecast": {
            k: v for k, v in snapshot.data.items()
            if v is not None
        },
    }

    llm = _get_llm()
    messages = [
        SystemMessage(content=_build_welcome_system_prompt(profile.language)),
        HumanMessage(content=json.dumps(payload, indent=2)),
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as exc:
        logger.error("bot: welcome narrative LLM error: %s", exc)
        return (
            f"Welcome! Skygent is now monitoring the weather for "
            f"{profile.name}. I'll check every {profile.check_interval_hours} "
            f"hours and notify you if conditions change significantly."
        )


def format_welcome_message(
    profile: MonitoringProfile,
    snapshot: ForecastSnapshot,
    narrative: str,
) -> str:
    """Format the welcome Telegram HTML message."""
    data = snapshot.data
    rain = data.get("precipitation_probability_max")
    temp = data.get("temperature_2m_max")
    wind = data.get("wind_speed_10m_max")

    lines = [
        f"<b>⛅ Skygent is watching: {_escape(profile.name)}</b>",
        "",
        _escape(narrative),
        "",
        "─────────────────",
        f"📅 Event: <b>{profile.event_datetime.strftime('%b %d, %Y %H:%M')} UTC</b>",
        f"📍 Location: <b>{profile.location[0]:.4f}, {profile.location[1]:.4f}</b>",
    ]

    if rain is not None:
        lines.append(f"🌧 Current rain probability: <b>{rain:.0f}%</b>")
    if temp is not None:
        lines.append(f"🌡 Max temperature: <b>{temp:.1f}°C</b>")
    if wind is not None:
        lines.append(f"💨 Max wind speed: <b>{wind:.1f} km/h</b>")

    lines += [
        "",
        f"🔄 I'll check every <b>{profile.check_interval_hours}h</b> and "
        f"message you if anything significant changes.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration via API
# ---------------------------------------------------------------------------

def _register_profile_via_api(data: dict, chat_id: str) -> dict | None:
    """
    POST the profile to the FastAPI backend.
    Returns the created profile JSON or None on failure.
    """
    api_url = os.environ.get("SKYGENT_API_URL", "http://localhost:8000")

    payload = {
        "name":                data["name"],
        "latitude":            data["lat"],
        "longitude":           data["lon"],
        "event_datetime":      data["event_datetime"],
        "check_interval_hours": int(data.get("check_interval_hours", 6)),
        "event_duration_hours": int(data.get("duration_hours", 4)),
        "context":             data.get("context", "social_event"),
        "notes":               f"Registered via Telegram bot (chat_id={chat_id})",
        "language":            data.get("language", "en"),
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{api_url}/api/v1/profiles", json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.error("bot: profile registration failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Step handlers
# ---------------------------------------------------------------------------

def handle_idle(chat_id: str, update: dict) -> None:
    """Any message in IDLE state → ask for language preference first."""
    send_message(
        chat_id,
        t('welcome', 'en'),
        reply_markup=_inline_keyboard([
            ("🇬🇧 English", "lang_en"),
            ("🇪🇸 Español", "lang_es"),
        ]),
    )
    _save_state(chat_id, Step.ASK_LANGUAGE, {})


def handle_ask_language(chat_id: str, callback_query: dict) -> None:
    """Receive language selection from inline keyboard, transition to ASK_LOCATION."""
    answer_callback_query(callback_query["id"])
    lang_data = callback_query.get("data", "lang_en")
    lang = "es" if lang_data == "lang_es" else "en"

    data = {"language": lang}
    _save_state(chat_id, Step.ASK_LOCATION, data)

    send_message(chat_id, t('ask_location', lang))


def handle_ask_location(chat_id: str, update: dict) -> None:
    """Expect a location message. Extract lat/lon and ask for event name."""
    msg = update.get("message", {})
    location = msg.get("location")
    _, data = _load_state(chat_id)
    lang = data.get('language', 'en')

    if not location:
        send_message(chat_id, t('location_missing', lang))
        return

    lat = location["latitude"]
    lon = location["longitude"]
    data.update({"lat": lat, "lon": lon})
    _save_state(chat_id, Step.ASK_NAME, data)

    send_message(
        chat_id,
        t('location_received', lang).format(lat=f"{lat:.4f}", lon=f"{lon:.4f}"),
    )


def handle_ask_name(chat_id: str, update: dict) -> None:
    """Expect free text event name."""
    msg = update.get("message", {})
    text = (msg.get("text") or "").strip()
    _, data = _load_state(chat_id)
    lang = data.get('language', 'en')

    if not text:
        send_message(chat_id, t('ask_name_missing', lang))
        return

    data["name"] = text
    _save_state(chat_id, Step.ASK_DATE, data)

    send_message(
        chat_id,
        t('name_confirmed', lang).format(name=_escape(text)),
    )


def handle_ask_date(chat_id: str, update: dict) -> None:
    """Parse natural date input using dateparser."""
    msg = update.get("message", {})
    text = (msg.get("text") or "").strip()
    _, data = _load_state(chat_id)
    lang = data.get('language', 'en')

    if not text:
        send_message(chat_id, t('ask_date_missing', lang))
        return

    # dateparser with future-dates preferred, UTC output
    parsed = dateparser.parse(
        text,
        settings={
            "PREFER_DATES_FROM": "future",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": "UTC",
            "TO_TIMEZONE": "UTC",
        },
    )

    if parsed is None:
        send_message(chat_id, t('date_unparseable', lang))
        return

    if parsed <= datetime.now(timezone.utc):
        send_message(chat_id, t('date_in_past', lang))
        return

    # Store date only — time collected next
    data["event_date"] = parsed.strftime("%Y-%m-%d")
    _save_state(chat_id, Step.ASK_TIME, data)

    send_message(
        chat_id,
        t('date_confirmed', lang).format(date=parsed.strftime('%B %d, %Y')),
    )


def handle_ask_time(chat_id: str, update: dict) -> None:
    """Parse time input and combine with previously stored date."""
    msg = update.get("message", {})
    text = (msg.get("text") or "").strip()
    _, data = _load_state(chat_id)
    lang = data.get('language', 'en')

    if not text:
        send_message(chat_id, t('ask_time_missing', lang))
        return

    date_str = data.get("event_date", "")

    # Parse time by prepending the known date so dateparser has full context
    parsed = dateparser.parse(
        f"{date_str} {text}",
        settings={
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": "UTC",
            "TO_TIMEZONE": "UTC",
        },
    )

    if parsed is None:
        send_message(chat_id, t('time_unparseable', lang))
        return

    data["event_datetime"] = parsed.isoformat()
    _save_state(chat_id, Step.ASK_CONTEXT, data)

    send_message(
        chat_id,
        t('time_confirmed', lang).format(time=parsed.strftime('%H:%M UTC')),
        reply_markup=_inline_keyboard([
            ("💒 Social event (wedding, party, concert)", "social_event"),
            ("🌾 Agriculture (harvest, planting, livestock)", "agriculture"),
            ("⚡ Energy (wind farm, solar, grid)", "energy"),
            ("🚛 Logistics (transport, construction, outdoor work)", "logistics"),
        ]),
    )


def handle_ask_context(chat_id: str, callback_query: dict) -> None:
    """Receive context selection from inline keyboard."""
    answer_callback_query(callback_query["id"])
    context = callback_query.get("data", "social_event")

    _, data = _load_state(chat_id)
    lang = data.get('language', 'en')
    data["context"] = context
    _save_state(chat_id, Step.ASK_DURATION, data)

    context_labels = {
        "social_event": "💒 Social event",
        "agriculture":  "🌾 Agriculture",
        "energy":       "⚡ Energy",
        "logistics":    "🚛 Logistics",
    }

    send_message(
        chat_id,
        f"{context_labels.get(context, context)} ✓\n\n"
        + t('ask_duration', lang),
        reply_markup=_inline_keyboard([
            ("2 hours",  "2"),
            ("4 hours",  "4"),
            ("6 hours",  "6"),
            ("8 hours",  "8"),
            ("Full day", "12"),
        ]),
    )


def handle_ask_duration(chat_id: str, callback_query: dict) -> None:
    """Receive duration selection and show confirmation summary."""
    answer_callback_query(callback_query["id"])
    duration = callback_query.get("data", "4")

    _, data = _load_state(chat_id)
    lang = data.get('language', 'en')
    data["duration_hours"] = int(duration)
    _save_state(chat_id, Step.CONFIRM, data)

    event_dt = datetime.fromisoformat(data["event_datetime"])

    summary = (
        t('confirm_header', lang) + "\n\n"
        + t('confirm_body', lang).format(
            name=_escape(data.get('name', '')),
            lat=f"{data.get('lat', 0):.4f}",
            lon=f"{data.get('lon', 0):.4f}",
            date=event_dt.strftime('%B %d, %Y %H:%M UTC'),
            duration=duration,
            context=data.get('context', 'social_event'),
            interval=data.get('check_interval_hours', 6),
        )
    )

    send_message(
        chat_id,
        summary,
        reply_markup=_inline_keyboard([
            ("✅ Yes, start monitoring!", "confirm_yes"),
            ("❌ No, cancel",            "confirm_no"),
        ]),
    )


def _patch_telegram_chat_id(profile_id: str, chat_id: str) -> None:
    """
    Directly update the profile's telegram_chat_id in the DB after
    bot registration. The POST /profiles endpoint does not accept this
    field (it is internal), so we patch it via the DB layer.

    This ensures notify_node routes future alerts to the correct user
    rather than the shared TELEGRAM_CHAT_ID env var.

    Uses a retry loop because the API and bot share the same SQLite DB.
    The API commits the profile row just before this is called — WAL mode
    helps but a brief retry guards against any remaining sync delay.
    """
    from skygent.api.database import ProfileRow, get_session_sync
    import json
    import time

    for attempt in range(3):
        with get_session_sync() as session:
            row = session.get(ProfileRow, profile_id)
            if row is not None:
                profile_data = json.loads(row.data)
                profile_data["telegram_chat_id"] = chat_id
                row.data = json.dumps(profile_data)
                session.add(row)
                logger.info(
                    "bot: patched telegram_chat_id=%s on profile %s",
                    chat_id, profile_id,
                )
                return
        # Row not visible yet — brief wait then retry
        logger.debug("bot: profile %s not found on attempt %d, retrying", profile_id, attempt + 1)
        time.sleep(0.5)

    logger.warning(
        "bot: could not patch telegram_chat_id for profile %s after 3 attempts",
        profile_id,
    )


def handle_confirm(chat_id: str, callback_query: dict) -> None:
    """Register the profile or cancel based on user choice."""
    answer_callback_query(callback_query["id"])
    choice = callback_query.get("data", "confirm_no")

    _, data = _load_state(chat_id)
    lang = data.get('language', 'en')

    if choice == "confirm_no":
        _clear_state(chat_id)
        send_message(chat_id, t('cancelled', lang))
        return

    # Register the profile
    send_message(chat_id, t('registering', lang))

    profile_json = _register_profile_via_api(data, chat_id)

    if profile_json is None:
        send_message(chat_id, t('registration_failed', lang))
        return

    # The API creates the profile without telegram_chat_id (the field is
    # not in ProfileCreate). We patch the profile row in the DB directly
    # so future alerts route to this user's chat.
    # This is the correct MVP approach — a PATCH /profiles/{id} endpoint
    # would be cleaner for a multi-user production deployment.
    _patch_telegram_chat_id(profile_json["id"], chat_id)
    _clear_state(chat_id)

    # Build a MonitoringProfile for the welcome forecast
    try:
        profile = MonitoringProfile(
            id=profile_json["id"],
            name=profile_json["name"],
            location=tuple(profile_json["location"]),
            event_datetime=datetime.fromisoformat(profile_json["event_datetime"]),
            check_interval_hours=profile_json["check_interval_hours"],
            event_duration_hours=profile_json["event_duration_hours"],
            context=profile_json["context"],
            telegram_chat_id=chat_id,
            language=data.get('language', 'en'),
        )

        import asyncio
        from skygent.integrations.openmeteo import OpenMeteoError

        horizon_days = (
            profile.event_datetime - datetime.now(timezone.utc)
        ).total_seconds() / 86400

        if horizon_days > 16:
            # Event is beyond the Open-Meteo forecast window — no data yet.
            # Send a friendly explanation instead of attempting a doomed fetch.
            send_message(
                chat_id,
                t('beyond_forecast_window', lang).format(
                    name=_escape(profile.name),
                    days=horizon_days,
                    interval=profile.check_interval_hours,
                    wait=max(0, horizon_days - 16),
                ),
            )
        else:
            # Event is within forecast window — fetch and narrate
            snapshot = asyncio.run(fetch_forecast(profile))
            narrative = _generate_welcome_narrative(profile, snapshot)
            welcome = format_welcome_message(profile, snapshot, narrative)
            send_message(chat_id, welcome)

    except Exception as exc:
        logger.error("bot: welcome forecast failed: %s", exc)
        send_message(
            chat_id,
            t('welcome_fallback', lang).format(name=_escape(profile_json['name'])),
        )


# ---------------------------------------------------------------------------
# Forecast Q&A command
# ---------------------------------------------------------------------------

def handle_forecast_query(chat_id: str, update: dict) -> None:
    """
    Handle /forecast — answer a forecast Q&A grounded in current snapshot
    and last 10 poll runs. Picks the profile with the soonest event_datetime
    for this chat. Uses llm.invoke (sync) since the bot is synchronous.
    """
    _, state_data = _load_state(chat_id)
    lang = state_data.get('language', 'en')

    with get_session_sync() as session:
        profiles = get_profiles_by_chat_id(session, chat_id)

    if not profiles:
        send_message(chat_id, t('no_active_events', lang))
        return

    # Pick the profile with the soonest event_datetime
    profile = min(profiles, key=lambda p: p.event_datetime)

    with get_session_sync() as session:
        snapshot = load_latest_snapshot(session, profile.id)
        poll_runs = get_recent_poll_runs(session, limit=10, profile_id=profile.id)

    payload = {
        'profile': {
            'name': profile.name,
            'event_datetime': profile.event_datetime.isoformat(),
            'context': profile.context,
            'language': profile.language,
        },
        'current_snapshot': {
            'data': snapshot.data if snapshot else None,
            'horizon_days': round(snapshot.horizon_days, 1) if snapshot else None,
            'fetched_at': snapshot.fetched_at.isoformat() if snapshot else None,
        },
        'poll_history': [
            {
                'ran_at': r.ran_at.isoformat(),
                'status': r.status,
                'changes_detected': r.changes_detected,
                'alert_sent': r.alert_sent,
            }
            for r in poll_runs
        ],
    }

    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload, default=str)),
        ]
        response = llm.invoke(messages)
        send_message(chat_id, response.content.strip())
    except Exception as exc:
        logger.error("bot: forecast Q&A LLM error: %s", exc)
        send_message(
            chat_id,
            "Sorry, I couldn't retrieve forecast information right now. Please try again later.",
        )


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def dispatch(update: dict) -> None:
    """
    Route a single Telegram update to the appropriate step handler.

    Handles three update types:
    - message with location: location pin from user
    - message with text: text input from user
    - callback_query: inline keyboard button press
    """
    # Callback query (inline keyboard button)
    if "callback_query" in update:
        cq = update["callback_query"]
        chat_id = str(cq["message"]["chat"]["id"])
        step, _ = _load_state(chat_id)
        data = cq.get("data", "")

        # /cancel from any state
        if data == "cancel":
            _clear_state(chat_id)
            answer_callback_query(cq["id"])
            send_message(chat_id, "Cancelled. Send /start to begin again.")
            return

        if step == Step.ASK_LANGUAGE:
            handle_ask_language(chat_id, cq)
        elif step == Step.ASK_CONTEXT:
            handle_ask_context(chat_id, cq)
        elif step == Step.ASK_DURATION:
            handle_ask_duration(chat_id, cq)
        elif step == Step.CONFIRM:
            handle_confirm(chat_id, cq)
        else:
            answer_callback_query(cq["id"])
        return

    # Regular message
    msg = update.get("message", {})
    if not msg:
        return

    chat_id = str(msg["chat"]["id"])
    text = (msg.get("text") or "").strip()

    # /cancel command at any step
    if text == "/cancel":
        _, data = _load_state(chat_id)
        lang = data.get('language', 'en')
        _clear_state(chat_id)
        send_message(chat_id, t('cancel_any', lang))
        return

    step, _ = _load_state(chat_id)

    # /start resets to IDLE regardless of current step
    if text == "/start":
        _clear_state(chat_id)
        handle_idle(chat_id, update)
        return

    # /forecast command — answer forecast Q&A for this user
    if text == "/forecast":
        handle_forecast_query(chat_id, update)
        return

    if step == Step.IDLE:
        handle_idle(chat_id, update)
    elif step == Step.ASK_LANGUAGE:
        # Text received during ASK_LANGUAGE — nudge user to press a button
        _, data = _load_state(chat_id)
        lang = data.get('language', 'en')
        send_message(chat_id, t('use_buttons', lang))
    elif step == Step.ASK_LOCATION:
        handle_ask_location(chat_id, update)
    elif step == Step.ASK_NAME:
        handle_ask_name(chat_id, update)
    elif step == Step.ASK_DATE:
        handle_ask_date(chat_id, update)
    elif step == Step.ASK_TIME:
        handle_ask_time(chat_id, update)
    else:
        # Unexpected text during a keyboard-input step
        _, data = _load_state(chat_id)
        lang = data.get('language', 'en')
        send_message(chat_id, t('use_buttons', lang))