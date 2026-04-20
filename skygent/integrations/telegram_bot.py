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
    get_session_sync,
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
    ASK_LOCATION = "ASK_LOCATION"
    ASK_NAME     = "ASK_NAME"
    ASK_DATE     = "ASK_DATE"
    ASK_TIME     = "ASK_TIME"
    ASK_CONTEXT  = "ASK_CONTEXT"
    ASK_DURATION = "ASK_DURATION"
    CONFIRM      = "CONFIRM"


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

_WELCOME_SYSTEM_PROMPT = """\
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
        SystemMessage(content=_WELCOME_SYSTEM_PROMPT),
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
    """Any message in IDLE state → explain the bot and ask for a location."""
    text = (
        "<b>👋 Welcome to Skygent!</b>\n\n"
        "I monitor the weather forecast for your events and alert you "
        "when conditions change significantly.\n\n"
        "To get started, <b>send me your event location</b> as a "
        "Telegram location pin 📍\n\n"
        "Tap the 📎 paperclip → Location, then drop a pin on the map."
    )
    send_message(chat_id, text)
    _save_state(chat_id, Step.ASK_LOCATION, {})


def handle_ask_location(chat_id: str, update: dict) -> None:
    """Expect a location message. Extract lat/lon and ask for event name."""
    msg = update.get("message", {})
    location = msg.get("location")

    if not location:
        send_message(
            chat_id,
            "I need a location pin 📍\n\n"
            "Tap the 📎 paperclip → Location, then drop a pin on the map.",
        )
        return

    lat = location["latitude"]
    lon = location["longitude"]
    _, data = _load_state(chat_id)
    data.update({"lat": lat, "lon": lon})
    _save_state(chat_id, Step.ASK_NAME, data)

    send_message(
        chat_id,
        f"📍 Got it — <b>{lat:.4f}, {lon:.4f}</b>\n\n"
        "What should I call this event?\n"
        "<i>e.g. Ana & Juan's Wedding, Harvest Day, Wind Farm Inspection</i>",
    )


def handle_ask_name(chat_id: str, update: dict) -> None:
    """Expect free text event name."""
    msg = update.get("message", {})
    text = (msg.get("text") or "").strip()

    if not text:
        send_message(chat_id, "Please send the event name as a text message.")
        return

    _, data = _load_state(chat_id)
    data["name"] = text
    _save_state(chat_id, Step.ASK_DATE, data)

    send_message(
        chat_id,
        f"Great — <b>{_escape(text)}</b> ✓\n\n"
        "What date is the event?\n"
        "<i>e.g. September 15, 2026 / 2026-09-15 / 15/09/2026</i>",
    )


def handle_ask_date(chat_id: str, update: dict) -> None:
    """Parse natural date input using dateparser."""
    msg = update.get("message", {})
    text = (msg.get("text") or "").strip()

    if not text:
        send_message(chat_id, "Please send the event date as a text message.")
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
        send_message(
            chat_id,
            "I couldn't understand that date 🤔\n\n"
            "Please try a format like:\n"
            "• <b>September 15, 2026</b>\n"
            "• <b>2026-09-15</b>\n"
            "• <b>15/09/2026</b>",
        )
        return

    if parsed <= datetime.now(timezone.utc):
        send_message(
            chat_id,
            "That date is in the past. Please send a future event date.",
        )
        return

    _, data = _load_state(chat_id)
    # Store date only — time collected next
    data["event_date"] = parsed.strftime("%Y-%m-%d")
    _save_state(chat_id, Step.ASK_TIME, data)

    send_message(
        chat_id,
        f"📅 <b>{parsed.strftime('%B %d, %Y')}</b> ✓\n\n"
        "What time does the event start? <i>(UTC)</i>\n"
        "<i>e.g. 17:00 / 5pm / 14:30</i>\n\n"
        "All times are in UTC. Montevideo is UTC-3, so 5pm local = 20:00 UTC.",
    )


def handle_ask_time(chat_id: str, update: dict) -> None:
    """Parse time input and combine with previously stored date."""
    msg = update.get("message", {})
    text = (msg.get("text") or "").strip()

    if not text:
        send_message(chat_id, "Please send the event time as a text message.")
        return

    _, data = _load_state(chat_id)
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
        send_message(
            chat_id,
            "I couldn't understand that time 🤔\n\n"
            "Please try a format like:\n"
            "• <b>17:00</b>\n"
            "• <b>5pm</b>\n"
            "• <b>14:30</b>",
        )
        return

    data["event_datetime"] = parsed.isoformat()
    _save_state(chat_id, Step.ASK_CONTEXT, data)

    send_message(
        chat_id,
        f"⏰ <b>{parsed.strftime('%H:%M UTC')}</b> ✓\n\n"
        "What type of event is this?",
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
        "How long does the event last?",
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
    data["duration_hours"] = int(duration)
    _save_state(chat_id, Step.CONFIRM, data)

    event_dt = datetime.fromisoformat(data["event_datetime"])

    summary = (
        f"<b>Please confirm your event:</b>\n\n"
        f"📍 <b>Name:</b> {_escape(data.get('name', ''))}\n"
        f"🗺 <b>Location:</b> {data.get('lat', 0):.4f}, {data.get('lon', 0):.4f}\n"
        f"📅 <b>Date:</b> {event_dt.strftime('%B %d, %Y %H:%M UTC')}\n"
        f"⏱ <b>Duration:</b> {duration} hours\n"
        f"🏷 <b>Context:</b> {data.get('context', 'social_event')}\n\n"
        f"Skygent will check the forecast every "
        f"<b>{data.get('check_interval_hours', 6)}h</b> and alert you "
        f"when conditions change significantly."
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

    if choice == "confirm_no":
        _clear_state(chat_id)
        send_message(
            chat_id,
            "Registration cancelled. Send /start whenever you're ready to try again.",
        )
        return

    # Register the profile
    _, data = _load_state(chat_id)

    send_message(chat_id, "⏳ Registering your event and fetching the first forecast...")

    profile_json = _register_profile_via_api(data, chat_id)

    if profile_json is None:
        send_message(
            chat_id,
            "❌ Something went wrong registering your event. "
            "Please try again or contact support.",
        )
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
                f"✅ <b>{_escape(profile.name)}</b> is now being monitored!\n\n"
                f"📅 Your event is <b>{horizon_days:.0f} days away</b> — "
                f"weather forecasts are only available up to 16 days out, "
                f"so I don't have data yet.\n\n"
                f"I'll check every <b>{profile.check_interval_hours}h</b> "
                f"and send you the first forecast as soon as it becomes available "
                f"(roughly {max(0, horizon_days - 16):.0f} days from now). "
                f"I'll also alert you whenever the forecast changes significantly.",
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
            f"✅ <b>{_escape(profile_json['name'])}</b> is now being monitored!\n\n"
            "I'll send you updates when the forecast changes significantly.",
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

        if step == Step.ASK_CONTEXT:
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
        _clear_state(chat_id)
        send_message(chat_id, "Cancelled. Send /start to begin again.")
        return

    step, _ = _load_state(chat_id)

    # /start resets to IDLE regardless of current step
    if text == "/start":
        _clear_state(chat_id)
        handle_idle(chat_id, update)
        return

    if step == Step.IDLE:
        handle_idle(chat_id, update)
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
        send_message(
            chat_id,
            "Please use the buttons above to continue, or send /cancel to start over.",
        )