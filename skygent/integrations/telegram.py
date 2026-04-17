"""
skygent/integrations/telegram.py — Telegram Bot API notification sender
========================================================================

Design decisions
----------------
1. httpx over python-telegram-bot: the official library adds significant
   complexity (Update handlers, dispatcher, webhook setup) for a use case
   that is purely outbound — we only ever send messages, never receive them.
   A direct POST to the sendMessage endpoint is 10 lines and zero extra
   dependencies.

2. Async send function with sync wrapper: matches the pattern established
   in openmeteo.py. The notify_node calls the async version; scripts and
   tests can use the sync wrapper.

3. TelegramError wraps all failures: callers catch one exception type, not
   httpx internals. The original exception is preserved as __cause__.

4. Message formatting: Telegram supports MarkdownV2 and HTML. We use HTML
   because it is more forgiving — MarkdownV2 requires escaping dozens of
   special characters, which conflicts with LLM-generated narrative text
   that may contain any punctuation. HTML only requires escaping <, >, &.

5. Message structure: the alert message is structured in three parts:
   - Header: event name + confidence badge
   - Body: the LLM narrative (already under 200 words by prompt design)
   - Footer: horizon, triggered variables, next check interval
   This gives the user the narrative first (the important part) with
   structured metadata below for quick scanning.

6. Character limit: Telegram messages are capped at 4096 characters.
   Alert narratives are capped at 200 words by the narrator prompt, so
   we should never approach this limit — but we truncate defensively.

7. Environment variables: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are
   read at call time (not at import time) so the module can be imported
   in tests without the variables being set.
"""

from __future__ import annotations

import html
import logging
import os

import httpx

from skygent.core.models import Alert, MonitoringProfile

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
MAX_MESSAGE_LENGTH = 4096


# ---------------------------------------------------------------------------
# Domain exception
# ---------------------------------------------------------------------------

class TelegramError(Exception):
    """
    Raised when the Telegram Bot API returns an error or the request fails.
    Wraps the original exception as __cause__.
    """


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------

# Confidence label → emoji + human description
_CONFIDENCE_BADGE = {
    "high":   "🟢 High confidence",
    "medium": "🟡 Medium confidence",
    "low":    "🔴 Low confidence",
}

# Open-Meteo variable name → display label
_VARIABLE_LABEL = {
    "precipitation_probability_max": "Rain probability",
    "temperature_2m_max":            "Max temperature",
    "wind_speed_10m_max":            "Wind speed",
    "weather_code":                  "Weather conditions",
}


def _escape(text: str) -> str:
    """Escape HTML special characters for Telegram HTML parse mode."""
    return html.escape(str(text), quote=False)


def format_alert_message(alert: Alert, profile: MonitoringProfile) -> str:
    """
    Format an Alert into a Telegram HTML message.

    Structure:
        📍 <Event Name>
        🟡 Medium confidence (3–7 days out)

        <LLM narrative>

        ─────────────────
        📅 5.2 days to event
        ⚡ Triggered by: Rain probability, Wind speed
        🔄 Next check in 6 hours
    """
    confidence_badge = _CONFIDENCE_BADGE.get(
        alert.confidence, f"Confidence: {alert.confidence}"
    )

    # Build trigger display names from numeric changes only.
    # weather_code is excluded from numeric diff so it never appears in
    # alert.changes. A pure weather_code rank trigger produces an empty
    # changes dict — the "Weather conditions" fallback covers that case.
    trigger_labels = []
    for var in alert.changes.keys():
        trigger_labels.append(_VARIABLE_LABEL.get(var, var.replace("_", " ").title()))
    if not trigger_labels:
        trigger_labels = ["Weather conditions"]
    triggers_str = ", ".join(trigger_labels)

    narrative = alert.narrative or "(no narrative generated)"

    lines = [
        f"<b>📍 {_escape(profile.name)}</b>",
        f"{_escape(confidence_badge)}",
        "",
        _escape(narrative),
        "",
        "─────────────────",
        f"📅 <b>{alert.horizon_days:.1f} days</b> to event",
        f"⚡ Triggered by: {_escape(triggers_str)}",
        f"🔄 Next check in <b>{profile.check_interval_hours}h</b>",
    ]

    message = "\n".join(lines)

    # Defensive truncation. Strip back to before any unclosed HTML tag
    # to avoid Telegram rejecting the message with a parse error.
    if len(message) > MAX_MESSAGE_LENGTH:
        truncated = message[: MAX_MESSAGE_LENGTH - 4]
        last_open = truncated.rfind("<")
        last_close = truncated.rfind(">")
        if last_open > last_close:
            truncated = truncated[:last_open]
        message = truncated.rstrip() + "\n..."
        logger.warning("Alert message truncated to %d chars", len(message))

    return message


# ---------------------------------------------------------------------------
# Send function
# ---------------------------------------------------------------------------

async def send_alert(
    alert: Alert,
    profile: MonitoringProfile,
    *,
    bot_token: str | None = None,
    chat_id: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> None:
    """
    Send an alert notification via the Telegram Bot API.

    Parameters
    ----------
    alert:      the Alert to deliver
    profile:    the MonitoringProfile the alert belongs to
    bot_token:  Telegram bot token. Defaults to TELEGRAM_BOT_TOKEN env var.
    chat_id:    Telegram chat ID to send to. Defaults to TELEGRAM_CHAT_ID env var.
    client:     optional httpx.AsyncClient (for testing / connection reuse).

    Raises
    ------
    TelegramError on missing credentials, HTTP errors, or API-level errors.
    """
    token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    cid = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

    if not token:
        raise TelegramError(
            "TELEGRAM_BOT_TOKEN not set. "
            "Export it as an environment variable or pass bot_token explicitly."
        )
    if not cid:
        raise TelegramError(
            "TELEGRAM_CHAT_ID not set. "
            "Export it as an environment variable or pass chat_id explicitly."
        )

    message = format_alert_message(alert, profile)
    url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"
    payload = {
        "chat_id":    cid,
        "text":       message,
        "parse_mode": "HTML",
    }

    logger.info(
        "telegram: sending alert %s to chat %s for '%s'",
        alert.id, cid, profile.name,
    )

    async def _do_send(c: httpx.AsyncClient) -> None:
        try:
            response = await c.post(url, json=payload, timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise TelegramError(
                f"Telegram API returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise TelegramError(f"Network error sending to Telegram: {exc}") from exc

        try:
            data = response.json()
        except Exception as exc:
            raise TelegramError(
                f"Telegram returned non-JSON response: {response.text[:200]}"
            ) from exc

        if not data.get("ok"):
            raise TelegramError(
                f"Telegram API error: {data.get('description', 'unknown error')}"
            )

    if client is not None:
        await _do_send(client)
    else:
        async with httpx.AsyncClient() as c:
            await _do_send(c)

    logger.info("telegram: alert %s delivered successfully", alert.id)


# ---------------------------------------------------------------------------
# Sync convenience wrapper
# ---------------------------------------------------------------------------

def send_alert_sync(
    alert: Alert,
    profile: MonitoringProfile,
    *,
    bot_token: str | None = None,
    chat_id: str | None = None,
) -> None:
    """
    Synchronous wrapper around send_alert for scripts and CLI use.
    Do NOT call from inside a running event loop.
    """
    import asyncio
    asyncio.run(send_alert(alert, profile, bot_token=bot_token, chat_id=chat_id))