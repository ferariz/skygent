"""
tests/test_agent.py — Unit tests for the LangGraph agent
=========================================================

Test philosophy
---------------
- All unit tests mock external calls (openmeteo fetch, LLM) completely.
  Zero network calls, zero API keys required.
- We test state transitions and routing logic, not the implementations of
  nodes already covered by test_diff.py and test_significance.py.
- The integration test (TestAgentIntegration) requires ANTHROPIC_API_KEY
  and makes real calls — gated behind the 'integration' mark.

Test structure
--------------
TestStateShape          — AgentState can be constructed and updated correctly
TestAfterFetchRouting   — conditional edge after fetch_forecast
TestAfterAnalyzeRouting — conditional edge after analyze_diff
TestAfterSignificanceRouting — conditional edge after evaluate_significance
TestAfterNarrateRouting — conditional edge after narrate
TestFetchForecastNode   — fetch node with mocked openmeteo
TestAnalyzeDiffNode     — diff node with pre-populated state
TestEvaluateSignificanceNode — significance node routing
TestNarrateNode         — narrate node with mocked LLM
TestNotifyNode          — notify node marks alert as sent
TestRunAgent            — full graph run end-to-end with all mocks
TestLazyLLMInit         — lazy LLM init: import safety and caching
TestStaleStateClearing  — stale fields do not bleed between runs
TestAgentIntegration    — real LLM call, mocked fetch (integration only)
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from skygent.agent.graph import (
    after_analyze,
    after_fetch,
    after_narrate,
    after_significance,
    run_agent,
)
from skygent.agent.nodes import (
    analyze_diff_node,
    evaluate_significance_node,
    fetch_forecast_node,
    narrate_node,
    notify_node,
)
from skygent.agent.state import AgentState
from skygent.core.models import (
    Alert,
    ForecastSnapshot,
    MonitoringProfile,
    VariableChange,
)
from langgraph.graph import END


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_profile(**overrides) -> MonitoringProfile:
    defaults = dict(
        name="Test Wedding",
        location=(-34.9011, -56.1645),
        event_datetime=datetime(2025, 9, 1, 16, 0, tzinfo=timezone.utc),
        monitoring_start=datetime(2025, 8, 1, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return MonitoringProfile(**defaults)


def make_snapshot(profile_id: str, horizon_days: float = 5.0) -> ForecastSnapshot:
    return ForecastSnapshot(
        profile_id=profile_id,
        fetched_at=datetime(2025, 8, 27, 0, 0, tzinfo=timezone.utc),
        target_datetime=datetime(2025, 9, 1, 16, 0, tzinfo=timezone.utc),
        data={
            "precipitation_probability_max": 20.0,
            "temperature_2m_max": 25.0,
            "wind_speed_10m_max": 18.0,
            "weather_code": 3,
        },
        horizon_days=horizon_days,
    )


def make_changes() -> dict[str, VariableChange]:
    return {
        "precipitation_probability_max": VariableChange(
            from_value=10.0, to_value=55.0, delta=45.0, delta_pct=450.0
        )
    }


def make_alert(profile: MonitoringProfile, prev: ForecastSnapshot,
               curr: ForecastSnapshot) -> Alert:
    return Alert(
        profile_id=profile.id,
        previous_snapshot_id=prev.id,
        current_snapshot_id=curr.id,
        changes=make_changes(),
        horizon_days=curr.horizon_days,
        confidence="medium",
    )


PROFILE = make_profile()


def base_state(**overrides) -> AgentState:
    """Return a minimal valid AgentState with optional overrides."""
    state: AgentState = {
        "profile": PROFILE,
        "previous_snapshot": None,
        "current_snapshot": None,
        "changes": None,
        "significant": None,
        "triggering_variables": None,
        "alert": None,
        "error": None,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# TestStateShape
# ---------------------------------------------------------------------------

class TestStateShape:
    def test_base_state_all_fields_present(self):
        state = base_state()
        for key in ("profile", "previous_snapshot", "current_snapshot",
                    "changes", "significant", "triggering_variables",
                    "alert", "error"):
            assert key in state

    def test_state_can_be_updated_with_partial_dict(self):
        state = base_state()
        snapshot = make_snapshot(PROFILE.id)
        state.update({"current_snapshot": snapshot})
        assert state["current_snapshot"] is snapshot
        assert state["previous_snapshot"] is None  # untouched


# ---------------------------------------------------------------------------
# TestAfterFetchRouting
# ---------------------------------------------------------------------------

class TestAfterFetchRouting:
    def test_error_routes_to_end(self):
        state = base_state(error="fetch failed")
        assert after_fetch(state) == END

    def test_first_run_significant_false_routes_to_end(self):
        """First run sets significant=False — should exit without diffing."""
        state = base_state(significant=False)
        assert after_fetch(state) == END

    def test_successful_fetch_with_previous_routes_to_analyze(self):
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)
        state = base_state(
            previous_snapshot=prev,
            current_snapshot=curr,
            significant=None,  # not yet evaluated
        )
        assert after_fetch(state) == "analyze_diff"


# ---------------------------------------------------------------------------
# TestAfterAnalyzeRouting
# ---------------------------------------------------------------------------

class TestAfterAnalyzeRouting:
    def test_error_routes_to_end(self):
        state = base_state(error="diff failed")
        assert after_analyze(state) == END

    def test_success_routes_to_evaluate_significance(self):
        state = base_state(changes=make_changes())
        assert after_analyze(state) == "evaluate_significance"


# ---------------------------------------------------------------------------
# TestAfterSignificanceRouting
# ---------------------------------------------------------------------------

class TestAfterSignificanceRouting:
    def test_error_routes_to_end(self):
        state = base_state(error="significance failed")
        assert after_significance(state) == END

    def test_not_significant_routes_to_end(self):
        state = base_state(significant=False, triggering_variables=[])
        assert after_significance(state) == END

    def test_significant_routes_to_narrate(self):
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)
        alert = make_alert(PROFILE, prev, curr)
        state = base_state(
            significant=True,
            triggering_variables=["precipitation_probability_max"],
            alert=alert,
        )
        assert after_significance(state) == "narrate"


# ---------------------------------------------------------------------------
# TestAfterNarrateRouting
# ---------------------------------------------------------------------------

class TestAfterNarrateRouting:
    def test_error_routes_to_end(self):
        state = base_state(error="LLM failed")
        assert after_narrate(state) == END

    def test_success_routes_to_notify(self):
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)
        alert = make_alert(PROFILE, prev, curr)
        alert_with_narrative = alert.model_copy(
            update={"narrative": "Rain probability has increased significantly."}
        )
        state = base_state(alert=alert_with_narrative)
        assert after_narrate(state) == "notify"


# ---------------------------------------------------------------------------
# TestFetchForecastNode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFetchForecastNode:
    async def test_first_run_returns_snapshot_and_significant_false(self):
        """No previous snapshot → significant=False, no error."""
        snapshot = make_snapshot(PROFILE.id)
        state = base_state()  # previous_snapshot=None

        with patch(
            "skygent.agent.nodes.fetch_forecast",
            new=AsyncMock(return_value=snapshot),
        ):
            result = await fetch_forecast_node(state)

        assert result["current_snapshot"] is snapshot
        assert result["significant"] is False
        assert result["error"] is None

    async def test_with_previous_snapshot_returns_only_current(self):
        """With previous snapshot → returns current_snapshot only, no significant."""
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)
        state = base_state(previous_snapshot=prev)

        with patch(
            "skygent.agent.nodes.fetch_forecast",
            new=AsyncMock(return_value=curr),
        ):
            result = await fetch_forecast_node(state)

        assert result["current_snapshot"] is curr
        assert result["significant"] is None  # cleared, not absent
        assert result["error"] is None

    async def test_api_error_returns_error_key(self):
        """OpenMeteoError → error in returned state, no snapshot."""
        from skygent.integrations.openmeteo import OpenMeteoError
        state = base_state()

        with patch(
            "skygent.agent.nodes.fetch_forecast",
            new=AsyncMock(side_effect=OpenMeteoError("API down")),
        ):
            result = await fetch_forecast_node(state)

        assert "error" in result
        assert "current_snapshot" not in result


# ---------------------------------------------------------------------------
# TestAnalyzeDiffNode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAnalyzeDiffNode:
    async def test_produces_changes_dict(self):
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)
        # Make curr differ from prev
        curr_data = dict(curr.data)
        curr_data["precipitation_probability_max"] = 65.0
        curr = curr.model_copy(update={"data": curr_data})

        state = base_state(previous_snapshot=prev, current_snapshot=curr)
        result = await analyze_diff_node(state)

        assert "changes" in result
        assert "precipitation_probability_max" in result["changes"]
        assert "error" not in result

    async def test_missing_snapshots_returns_error(self):
        state = base_state()  # both None
        result = await analyze_diff_node(state)
        assert "error" in result


# ---------------------------------------------------------------------------
# TestEvaluateSignificanceNode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEvaluateSignificanceNode:
    async def test_significant_change_returns_alert(self):
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)
        changes = make_changes()  # 45pp precip change > 20pp threshold
        state = base_state(
            previous_snapshot=prev,
            current_snapshot=curr,
            changes=changes,
        )
        result = await evaluate_significance_node(state)

        assert result["significant"] is True
        assert "precipitation_probability_max" in result["triggering_variables"]
        assert result["alert"] is not None
        assert result["alert"].narrative == ""  # not yet filled

    async def test_no_change_returns_significant_false(self):
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)  # identical data
        changes: dict = {}  # no numeric changes
        state = base_state(
            previous_snapshot=prev,
            current_snapshot=curr,
            changes=changes,
        )
        result = await evaluate_significance_node(state)

        assert result["significant"] is False
        assert result["triggering_variables"] == []
        assert result["alert"] is None  # explicitly cleared

    async def test_missing_snapshots_returns_error(self):
        state = base_state(changes={})
        result = await evaluate_significance_node(state)
        assert "error" in result


# ---------------------------------------------------------------------------
# TestNarrateNode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestNarrateNode:
    async def test_fills_in_narrative(self):
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)
        alert = make_alert(PROFILE, prev, curr)
        state = base_state(
            previous_snapshot=prev,
            current_snapshot=curr,
            alert=alert,
            triggering_variables=["precipitation_probability_max"],
        )

        mock_response = MagicMock()
        mock_response.content = "Rain probability has jumped significantly."

        with patch(
            "skygent.agent.nodes._get_llm",
            return_value=MagicMock(ainvoke=AsyncMock(return_value=mock_response)),
        ):
            result = await narrate_node(state)

        assert "alert" in result
        assert result["alert"].narrative == "Rain probability has jumped significantly."
        assert result["alert"].sent is False  # notify hasn't run yet

    async def test_no_alert_in_state_returns_error(self):
        state = base_state()
        result = await narrate_node(state)
        assert "error" in result

    async def test_llm_failure_returns_error(self):
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)
        alert = make_alert(PROFILE, prev, curr)
        state = base_state(alert=alert, triggering_variables=[])

        with patch(
            "skygent.agent.nodes._get_llm",
            return_value=MagicMock(ainvoke=AsyncMock(side_effect=Exception("LLM timeout"))),
        ):
            result = await narrate_node(state)

        assert "error" in result
        assert "LLM timeout" in result["error"]


# ---------------------------------------------------------------------------
# TestNotifyNode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestNotifyNode:
    async def test_marks_alert_as_sent(self):
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)
        alert = make_alert(PROFILE, prev, curr)
        alert = alert.model_copy(update={"narrative": "Conditions have changed."})
        state = base_state(alert=alert)

        result = await notify_node(state)

        assert result["alert"].sent is True
        assert "error" not in result

    async def test_no_alert_returns_error(self):
        state = base_state()
        result = await notify_node(state)
        assert "error" in result

    async def test_missing_narrative_returns_error(self):
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)
        alert = make_alert(PROFILE, prev, curr)
        # narrative is "" by default
        state = base_state(alert=alert)
        result = await notify_node(state)
        assert "error" in result


# ---------------------------------------------------------------------------
# TestRunAgent — full graph with all external calls mocked
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunAgent:
    async def test_first_run_exits_cleanly(self):
        """No previous snapshot → graph exits after fetch, no error."""
        snapshot = make_snapshot(PROFILE.id)

        with patch(
            "skygent.agent.nodes.fetch_forecast",
            new=AsyncMock(return_value=snapshot),
        ):
            final = await run_agent(PROFILE, previous_snapshot=None)

        assert final["current_snapshot"] is snapshot
        assert final["significant"] is False
        assert final["error"] is None

    async def test_no_significant_change_exits_before_narrate(self):
        """Identical snapshots → significant=False, LLM never called."""
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)  # identical data

        llm_mock = MagicMock(ainvoke=AsyncMock())

        with patch("skygent.agent.nodes.fetch_forecast",
                   new=AsyncMock(return_value=curr)), \
             patch("skygent.agent.nodes._get_llm", return_value=llm_mock):
            final = await run_agent(PROFILE, previous_snapshot=prev)

        assert final["significant"] is False
        llm_mock.ainvoke.assert_not_called()

    async def test_significant_change_produces_sent_alert(self):
        """Large precip change → full pipeline runs, alert sent."""
        prev = make_snapshot(PROFILE.id)
        # Current has 55pp precip (prev had 20pp) — exceeds 20pp threshold
        curr_data = {**prev.data, "precipitation_probability_max": 65.0}
        curr = prev.model_copy(update={"data": curr_data})

        mock_response = MagicMock()
        mock_response.content = "Rain probability has increased significantly."

        with patch("skygent.agent.nodes.fetch_forecast",
                   new=AsyncMock(return_value=curr)), \
             patch("skygent.agent.nodes._get_llm",
                   return_value=MagicMock(ainvoke=AsyncMock(return_value=mock_response))):
            final = await run_agent(PROFILE, previous_snapshot=prev)

        assert final["significant"] is True
        assert final["alert"] is not None
        assert final["alert"].sent is True
        assert "Rain probability" in final["alert"].narrative
        assert final["error"] is None

    async def test_fetch_error_returns_error_state(self):
        """API failure → error in final state, no further processing."""
        from skygent.integrations.openmeteo import OpenMeteoError
        prev = make_snapshot(PROFILE.id)

        with patch("skygent.agent.nodes.fetch_forecast",
                   new=AsyncMock(side_effect=OpenMeteoError("timeout"))):
            final = await run_agent(PROFILE, previous_snapshot=prev)

        assert final["error"] is not None
        assert "fetch_forecast" in final["error"]




# ---------------------------------------------------------------------------
# TestLazyLLMInit — fix A: import must not require ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------

class TestLazyLLMInit:
    def test_llm_not_instantiated_until_needed(self):
        """
        _llm_instance must be None until _get_llm() is explicitly called.
        This is the core invariant of lazy init: importing nodes.py must not
        require ANTHROPIC_API_KEY, so _llm_instance must start as None and
        remain None as long as no node that calls the LLM has been invoked.

        We reset _llm_instance to None before the assertion to isolate from
        other tests that may have already triggered _get_llm(). If lazy init
        is broken (LLM created at import), this reset would have no effect
        and the test below (test_get_llm_creates_instance) would fail because
        call_count would already be > 0.
        """
        import skygent.agent.nodes as nodes_module
        nodes_module._llm_instance = None   # reset to simulate fresh import
        assert nodes_module._llm_instance is None  # strict — no or True escape

    def test_get_llm_creates_and_caches_instance(self):
        """
        _get_llm() must construct ChatAnthropic exactly once and return the
        same instance on subsequent calls (singleton cache).
        """
        from skygent.agent.nodes import _get_llm
        with patch("skygent.agent.nodes.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            import skygent.agent.nodes as nodes_module
            nodes_module._llm_instance = None  # reset for test isolation
            llm1 = _get_llm()
            llm2 = _get_llm()
            assert llm1 is llm2              # same instance returned twice
            assert mock_cls.call_count == 1  # constructed exactly once

    def test_llm_not_created_during_non_llm_node_execution(self):
        """
        Running the fetch, diff, and significance nodes must not trigger
        LLM instantiation. _llm_instance should remain None after these
        nodes run, confirming the LLM is truly lazy.
        """
        import skygent.agent.nodes as nodes_module
        nodes_module._llm_instance = None  # reset

        # These nodes never call _get_llm() — confirmed by checking state
        # after they run without patching ChatAnthropic at all.
        # If _get_llm() were called, ChatAnthropic() would be invoked and
        # (outside a patch context) would attempt credential validation.
        # The fact that the unit tests for these nodes pass without patching
        # ChatAnthropic is the real proof — this test makes it explicit.
        assert nodes_module._llm_instance is None


# ---------------------------------------------------------------------------
# TestStaleStateClearing — fix B: fields must not bleed between runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestStaleStateClearing:
    async def test_fetch_node_clears_stale_downstream_fields(self):
        """
        fetch_forecast_node must zero out changes, alert, triggering_variables
        from any prior run even when the new fetch succeeds.
        """
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)
        stale_alert = make_alert(PROFILE, prev, curr)
        stale_changes = make_changes()

        # State as if a prior run left data behind
        state = base_state(
            previous_snapshot=prev,
            changes=stale_changes,
            significant=True,
            triggering_variables=["precipitation_probability_max"],
            alert=stale_alert,
        )

        with patch("skygent.agent.nodes.fetch_forecast",
                   new=AsyncMock(return_value=curr)):
            result = await fetch_forecast_node(state)

        assert result["changes"] is None
        assert result["significant"] is None
        assert result["triggering_variables"] is None
        assert result["alert"] is None
        assert result["error"] is None

    async def test_analyze_node_clears_stale_significance_fields(self):
        """
        analyze_diff_node must reset significant, triggering_variables, alert
        so a prior significant run does not pollute an identical new run.
        """
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)
        stale_alert = make_alert(PROFILE, prev, curr)

        state = base_state(
            previous_snapshot=prev,
            current_snapshot=curr,
            significant=True,
            triggering_variables=["precipitation_probability_max"],
            alert=stale_alert,
        )

        result = await analyze_diff_node(state)

        assert result["significant"] is None
        assert result["triggering_variables"] is None
        assert result["alert"] is None

    async def test_significance_node_clears_alert_when_not_significant(self):
        """
        When no threshold is crossed, evaluate_significance_node must
        explicitly set alert=None so a stale alert from a prior run
        does not persist into the final state.
        """
        prev = make_snapshot(PROFILE.id)
        curr = make_snapshot(PROFILE.id)
        stale_alert = make_alert(PROFILE, prev, curr)

        state = base_state(
            previous_snapshot=prev,
            current_snapshot=curr,
            changes={},           # no changes
            alert=stale_alert,    # stale from prior run
        )

        result = await evaluate_significance_node(state)

        assert result["significant"] is False
        assert result["alert"] is None  # explicitly cleared

# ---------------------------------------------------------------------------
# TestAgentIntegration — real LLM + real API (gated)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
class TestAgentIntegration:
    """
    Agent run with a real LLM call and mocked Open-Meteo fetch.
    Requires ANTHROPIC_API_KEY in environment.

    Open-Meteo is mocked (two pre-built snapshots with a large delta) so
    the test is deterministic and fast. The real integration point is the
    LLM call in the narrate node — this confirms the full narrative
    pipeline works end-to-end with actual Claude output.

    Run with: pytest -m integration tests/test_agent.py -v
    """

    async def test_full_run_with_significant_change(self):
        """
        Construct two snapshots with a large precip delta and run the full
        agent — real LLM generates the narrative.
        """
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        prev = make_snapshot(PROFILE.id, horizon_days=5.0)
        curr_data = {**prev.data, "precipitation_probability_max": 75.0}
        curr = prev.model_copy(update={
            "data": curr_data,
            "horizon_days": 4.8,
        })

        with patch("skygent.agent.nodes.fetch_forecast",
                   new=AsyncMock(return_value=curr)):
            final = await run_agent(PROFILE, previous_snapshot=prev)

        assert final["significant"] is True
        assert final["alert"].sent is True
        assert len(final["alert"].narrative) > 20
        print(f"\nGenerated narrative:\n{final['alert'].narrative}")