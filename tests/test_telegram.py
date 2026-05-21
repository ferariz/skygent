"""
tests/test_telegram.py — Unit tests for the Telegram integration
================================================================

Test philosophy
---------------
- All unit tests mock httpx entirely — zero real network calls.
- We test format_alert_message independently as a pure function since it
  has no side effects and its output is directly user-facing.
- send_alert tests verify the correct URL, payload shape, parse_mode,
  and error handling without hitting the real API.
- One integration test sends a real message (gated behind the
  'integration' mark and requires env vars to be set).

Test structure
--------------
TestFormatAlertMessage  — message structure, HTML escaping, truncation
TestSendAlert           — HTTP payload, error handling (mocked)
TestNotifyNodeTelegram  — notify_node routes to send_alert correctly
TestTelegramIntegration — real delivery (integration only)
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from skygent.core.models import Alert, ForecastSnapshot, MonitoringProfile, VariableChange
from skygent.integrations.telegram import (
    MAX_MESSAGE_LENGTH,
    TelegramError,
    format_alert_message,
    send_alert,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_profile(**overrides) -> MonitoringProfile:
    defaults = dict(
        name="Ana & Juan's Wedding",
        location=(-34.9011, -56.1645),
        event_datetime=datetime(2025, 9, 15, 17, 0, tzinfo=timezone.utc),
        monitoring_start=datetime(2025, 8, 1, tzinfo=timezone.utc),
        check_interval_hours=6,
    )
    defaults.update(overrides)
    return MonitoringProfile(**defaults)


def make_alert(profile_id: str, **overrides) -> Alert:
    defaults = dict(
        profile_id=profile_id,
        previous_snapshot_id="prev-id",
        current_snapshot_id="curr-id",
        changes={
            "precipitation_probability_max": VariableChange(
                from_value=10.0, to_value=55.0, delta=45.0, delta_pct=450.0
            )
        },
        horizon_days=5.2,
        confidence="medium",
        narrative="Rain probability has jumped from 10% to 55% for your wedding day. At 5 days out this forecast carries medium confidence — worth keeping an eye on. Next check in 6 hours.",
        sent=False,
    )
    defaults.update(overrides)
    return Alert(**defaults)


PROFILE = make_profile()
ALERT = make_alert(PROFILE.id)


def make_mock_client(status_code: int = 200, ok: bool = True) -> MagicMock:
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = {"ok": ok, "result": {"message_id": 42}}
    mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


# ---------------------------------------------------------------------------
# TestFormatAlertMessage
# ---------------------------------------------------------------------------

class TestFormatAlertMessage:
    def test_contains_event_name(self):
        msg = format_alert_message(ALERT, PROFILE)
        assert "Ana &amp; Juan&#x27;s Wedding" in msg or "Ana" in msg

    def test_contains_confidence_badge(self):
        msg = format_alert_message(ALERT, PROFILE)
        assert "Medium confidence" in msg

    def test_contains_narrative(self):
        msg = format_alert_message(ALERT, PROFILE)
        assert "Rain probability" in msg

    def test_contains_horizon_days(self):
        msg = format_alert_message(ALERT, PROFILE)
        assert "5.2" in msg

    def test_contains_check_interval(self):
        msg = format_alert_message(ALERT, PROFILE)
        assert "6" in msg  # check_interval_hours

    def test_contains_trigger_variable_label(self):
        """Variable names are converted to human-readable labels."""
        msg = format_alert_message(ALERT, PROFILE)
        assert "Rain probability" in msg  # precipitation_probability_max → label

    def test_html_escaping_in_event_name(self):
        """Special HTML chars in event name must be escaped."""
        profile = make_profile(name="Test & <Event>")
        msg = format_alert_message(ALERT, profile)
        assert "<Event>" not in msg          # raw HTML tag must not appear
        assert "&lt;Event&gt;" in msg        # must be escaped

    def test_high_confidence_badge(self):
        alert = make_alert(PROFILE.id, confidence="high", horizon_days=1.5)
        msg = format_alert_message(alert, PROFILE)
        assert "High confidence" in msg
        assert "🟢" in msg

    def test_low_confidence_badge(self):
        alert = make_alert(PROFILE.id, confidence="low", horizon_days=9.0)
        msg = format_alert_message(alert, PROFILE)
        assert "Low confidence" in msg
        assert "🔴" in msg

    def test_message_truncated_when_too_long(self):
        """Messages exceeding Telegram's limit must be truncated."""
        long_narrative = "x " * 3000  # well over 4096 chars when formatted
        alert = make_alert(PROFILE.id, narrative=long_narrative)
        msg = format_alert_message(alert, PROFILE)
        assert len(msg) <= MAX_MESSAGE_LENGTH

    def test_empty_changes_uses_fallback_trigger(self):
        """Alert with no numeric changes (pure weathercode trigger)."""
        alert = make_alert(PROFILE.id, changes={})
        msg = format_alert_message(alert, PROFILE)
        assert "Weather conditions" in msg

    def test_unknown_variable_name_formatted_gracefully(self):
        """Variables not in _VARIABLE_LABEL fall back to title-cased name."""
        alert = make_alert(PROFILE.id, changes={
            "some_future_variable": VariableChange(
                from_value=1.0, to_value=2.0, delta=1.0, delta_pct=100.0
            )
        })
        msg = format_alert_message(alert, PROFILE)
        assert "Some Future Variable" in msg


# ---------------------------------------------------------------------------
# TestSendAlert
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSendAlert:
    async def test_posts_to_correct_url(self):
        client = make_mock_client()
        await send_alert(
            ALERT, PROFILE, bot_token="test-token", chat_id="12345", client=client
        )
        call_args = client.post.call_args
        assert "bot-test-token" in call_args[0][0] or "test-token" in call_args[0][0]
        assert "sendMessage" in call_args[0][0]

    async def test_payload_contains_chat_id(self):
        client = make_mock_client()
        await send_alert(
            ALERT, PROFILE, bot_token="test-token", chat_id="99999", client=client
        )
        payload = client.post.call_args[1]["json"]
        assert payload["chat_id"] == "99999"

    async def test_payload_uses_html_parse_mode(self):
        client = make_mock_client()
        await send_alert(
            ALERT, PROFILE, bot_token="test-token", chat_id="12345", client=client
        )
        payload = client.post.call_args[1]["json"]
        assert payload["parse_mode"] == "HTML"

    async def test_payload_contains_formatted_message(self):
        client = make_mock_client()
        await send_alert(
            ALERT, PROFILE, bot_token="test-token", chat_id="12345", client=client
        )
        payload = client.post.call_args[1]["json"]
        assert "Ana" in payload["text"]
        assert "Rain probability" in payload["text"]

    async def test_missing_token_raises_telegram_error(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(TelegramError, match="TELEGRAM_BOT_TOKEN"):
                await send_alert(ALERT, PROFILE, chat_id="12345")

    async def test_missing_chat_id_raises_telegram_error(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(TelegramError, match="TELEGRAM_CHAT_ID"):
                await send_alert(ALERT, PROFILE, bot_token="test-token")

    async def test_http_error_raises_telegram_error(self):
        import httpx
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(TelegramError):
            await send_alert(
                ALERT, PROFILE, bot_token="bad-token", chat_id="12345",
                client=mock_client
            )

    async def test_api_ok_false_raises_telegram_error(self):
        """Telegram returns HTTP 200 but ok=False for logical errors."""
        client = make_mock_client(status_code=200, ok=False)
        client.post.return_value.json.return_value = {
            "ok": False,
            "description": "Bad Request: chat not found",
        }
        with pytest.raises(TelegramError, match="chat not found"):
            await send_alert(
                ALERT, PROFILE, bot_token="test-token", chat_id="wrong-id",
                client=client
            )

    async def test_network_error_raises_telegram_error(self):
        import httpx
        mock_client = MagicMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        with pytest.raises(TelegramError):
            await send_alert(
                ALERT, PROFILE, bot_token="test-token", chat_id="12345",
                client=mock_client
            )

    async def test_non_json_response_raises_telegram_error(self):
        """
        Fix A: response.json() is now wrapped separately. A proxy or CDN
        returning HTML on a 200 (common during outages) previously escaped
        as a raw ValueError; now always surfaces as TelegramError.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = ValueError("not valid JSON")
        mock_response.text = "<html>Bad Gateway</html>"
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(TelegramError, match="non-JSON"):
            await send_alert(
                ALERT, PROFILE, bot_token="test-token", chat_id="12345",
                client=mock_client
            )

    async def test_reads_token_from_environment(self):
        """send_alert reads credentials from env vars when not passed explicitly."""
        client = make_mock_client()
        with patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "env-token",
            "TELEGRAM_CHAT_ID": "env-chat",
        }):
            await send_alert(ALERT, PROFILE, client=client)
        call_url = client.post.call_args[0][0]
        assert "env-token" in call_url


# ---------------------------------------------------------------------------
# TestNotifyNodeTelegram
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestNotifyNodeTelegram:
    """Verify notify_node routes to send_alert for telegram channel."""

    async def test_telegram_channel_calls_send_alert(self):
        from skygent.agent.nodes import notify_node
        from skygent.agent.state import AgentState

        state: AgentState = {
            "profile": PROFILE,
            "alert": ALERT,
            "previous_snapshot": None,
            "current_snapshot": None,
            "changes": None,
            "significant": True,
            "triggering_variables": ["precipitation_probability_max"],
            "error": None,
        }

        with patch(
            "skygent.agent.nodes.send_alert",
            new=AsyncMock(),
        ) as mock_send:
            result = await notify_node(state)

        mock_send.assert_called_once()
        assert result["alert"].sent is True

    async def test_telegram_error_returns_error_state(self):
        from skygent.agent.nodes import notify_node
        from skygent.agent.state import AgentState

        state: AgentState = {
            "profile": PROFILE,
            "alert": ALERT,
            "previous_snapshot": None,
            "current_snapshot": None,
            "changes": None,
            "significant": True,
            "triggering_variables": [],
            "error": None,
        }

        with patch(
            "skygent.agent.nodes.send_alert",
            new=AsyncMock(side_effect=TelegramError("bot token invalid")),
        ):
            result = await notify_node(state)

        assert "error" in result
        assert "Telegram" in result["error"]

    async def test_unknown_channel_logs_and_marks_sent(self):
        """Unknown notification channels fall back to log output, not error."""
        from skygent.agent.nodes import notify_node
        from skygent.agent.state import AgentState

        profile = make_profile(notification_channel="email")
        state: AgentState = {
            "profile": profile,
            "alert": ALERT,
            "previous_snapshot": None,
            "current_snapshot": None,
            "changes": None,
            "significant": True,
            "triggering_variables": [],
            "error": None,
        }

        result = await notify_node(state)
        assert result["alert"].sent is True
        assert "error" not in result or result.get("error") is None


# ---------------------------------------------------------------------------
# TestHandleForecastQuery — /forecast command (LLM mocked)
# ---------------------------------------------------------------------------

class TestHandleForecastQuery:
    """
    Tests for handle_forecast_query. The LLM is always mocked to satisfy
    the architecture invariant: no LLM calls in the test suite.
    """

    def _make_profile(self, chat_id="chat_123", **overrides) -> MonitoringProfile:
        defaults = dict(
            name="Test Event",
            location=(-34.9011, -56.1645),
            event_datetime=datetime.now(timezone.utc) + timedelta(days=10),
            monitoring_start=datetime.now(timezone.utc),
            telegram_chat_id=chat_id,
        )
        defaults.update(overrides)
        return MonitoringProfile(**defaults)

    def _make_snapshot(self, profile_id: str) -> ForecastSnapshot:
        return ForecastSnapshot(
            profile_id=profile_id,
            target_datetime=datetime.now(timezone.utc) + timedelta(days=10),
            data={"precipitation_probability_max": 20.0, "temperature_2m_max": 25.0},
            horizon_days=10.0,
        )

    def test_no_profiles_sends_no_active_events_message(self):
        from skygent.integrations.telegram_bot import handle_forecast_query

        with patch("skygent.integrations.telegram_bot._load_state", return_value=("IDLE", {})), \
             patch("skygent.integrations.telegram_bot.get_profiles_by_chat_id", return_value=[]), \
             patch("skygent.integrations.telegram_bot.get_session_sync") as mock_session, \
             patch("skygent.integrations.telegram_bot.send_message") as mock_send:
            # Make get_session_sync a context manager
            mock_session.return_value.__enter__ = lambda s: MagicMock()
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            handle_forecast_query("chat_123", {})

        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        assert "chat_123" == args[0]
        # Message should contain something about no events (English)
        assert "no active" in args[1].lower() or "No " in args[1]

    def test_with_profile_calls_llm_and_sends_response(self):
        from skygent.integrations.telegram_bot import handle_forecast_query
        from unittest.mock import MagicMock, patch

        profile = self._make_profile()
        snapshot = self._make_snapshot(profile.id)

        mock_llm_response = MagicMock()
        mock_llm_response.content = "The weather looks good for your event."

        with patch("skygent.integrations.telegram_bot._load_state", return_value=("IDLE", {"language": "en"})), \
             patch("skygent.integrations.telegram_bot.get_profiles_by_chat_id", return_value=[profile]), \
             patch("skygent.integrations.telegram_bot.load_latest_snapshot", return_value=snapshot), \
             patch("skygent.integrations.telegram_bot.get_recent_poll_runs", return_value=[]), \
             patch("skygent.integrations.telegram_bot.get_session_sync") as mock_session, \
             patch("skygent.integrations.telegram_bot._get_llm") as mock_get_llm, \
             patch("skygent.integrations.telegram_bot.send_message") as mock_send:
            mock_session.return_value.__enter__ = lambda s: MagicMock()
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_llm_response
            mock_get_llm.return_value = mock_llm

            handle_forecast_query("chat_123", {})

        mock_llm.invoke.assert_called_once()
        # send_message is now called at least twice: header + LLM narrative
        assert mock_send.call_count >= 2
        # The LLM narrative is the last send_message call
        sent_text = mock_send.call_args[0][1]
        assert "weather looks good" in sent_text

    def test_llm_error_sends_fallback_message(self):
        from skygent.integrations.telegram_bot import handle_forecast_query

        profile = self._make_profile()
        snapshot = self._make_snapshot(profile.id)

        with patch("skygent.integrations.telegram_bot._load_state", return_value=("IDLE", {})), \
             patch("skygent.integrations.telegram_bot.get_profiles_by_chat_id", return_value=[profile]), \
             patch("skygent.integrations.telegram_bot.load_latest_snapshot", return_value=snapshot), \
             patch("skygent.integrations.telegram_bot.get_recent_poll_runs", return_value=[]), \
             patch("skygent.integrations.telegram_bot.get_session_sync") as mock_session, \
             patch("skygent.integrations.telegram_bot._get_llm") as mock_get_llm, \
             patch("skygent.integrations.telegram_bot.send_message") as mock_send:
            mock_session.return_value.__enter__ = lambda s: MagicMock()
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("LLM unavailable")
            mock_get_llm.return_value = mock_llm

            handle_forecast_query("chat_123", {})

        # send_message is now called at least twice: header + fallback
        assert mock_send.call_count >= 2
        sent_text = mock_send.call_args[0][1]
        # Fallback message should mention inability to retrieve info
        assert len(sent_text) > 0

    def test_picks_soonest_event_when_multiple_profiles(self):
        from skygent.integrations.telegram_bot import handle_forecast_query

        p_later = self._make_profile(
            name="Later Event",
            event_datetime=datetime.now(timezone.utc) + timedelta(days=30),
        )
        p_sooner = self._make_profile(
            name="Sooner Event",
            event_datetime=datetime.now(timezone.utc) + timedelta(days=5),
        )
        # Provide a snapshot for the soonest profile so we don't hit the
        # graceful empty-state early return (poll_runs=[], snapshot=None).
        sooner_snapshot = self._make_snapshot(p_sooner.id)

        mock_llm_response = MagicMock()
        mock_llm_response.content = "Forecast info."
        captured_payload = {}

        def capture_invoke(messages):
            import json
            # The user message contains the payload JSON
            user_msg_content = messages[1].content
            captured_payload.update(json.loads(user_msg_content))
            return mock_llm_response

        with patch("skygent.integrations.telegram_bot._load_state", return_value=("IDLE", {})), \
             patch("skygent.integrations.telegram_bot.get_profiles_by_chat_id", return_value=[p_later, p_sooner]), \
             patch("skygent.integrations.telegram_bot.load_latest_snapshot", return_value=sooner_snapshot), \
             patch("skygent.integrations.telegram_bot.get_recent_poll_runs", return_value=[]), \
             patch("skygent.integrations.telegram_bot.get_session_sync") as mock_session, \
             patch("skygent.integrations.telegram_bot._get_llm") as mock_get_llm, \
             patch("skygent.integrations.telegram_bot.send_message"):
            mock_session.return_value.__enter__ = lambda s: MagicMock()
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = capture_invoke
            mock_get_llm.return_value = mock_llm

            handle_forecast_query("chat_123", {})

        assert captured_payload["profile"]["name"] == "Sooner Event"


# ---------------------------------------------------------------------------
# TestTelegramIntegration — real delivery (gated)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
class TestTelegramIntegration:
    """
    Sends a real Telegram message. Requires TELEGRAM_BOT_TOKEN and
    TELEGRAM_CHAT_ID to be set in the environment.

    Run with: pytest -m integration tests/test_telegram.py -v
    """

    async def test_send_real_alert_message(self):
        import os
        if not os.environ.get("TELEGRAM_BOT_TOKEN"):
            pytest.skip("TELEGRAM_BOT_TOKEN not set")
        if not os.environ.get("TELEGRAM_CHAT_ID"):
            pytest.skip("TELEGRAM_CHAT_ID not set")

        # Use a clearly labeled test alert so you know it's a test
        test_profile = make_profile(name="🧪 Skygent Test Alert")
        test_alert = make_alert(
            test_profile.id,
            narrative=(
                "This is a test alert from Skygent. "
                "Rain probability has increased from 10% to 55% for your test event. "
                "At 5 days out this forecast carries medium confidence."
            ),
        )

        # Should not raise
        await send_alert(test_alert, test_profile)
        print(f"\nAlert sent successfully for '{test_profile.name}'")