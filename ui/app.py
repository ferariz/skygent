"""
ui/app.py — Skygent Streamlit Dashboard
========================================

Minimal functional dashboard covering three things:
1. Register a new event for monitoring
2. View active profiles and scheduler status
3. Browse generated alerts

Design decisions
----------------
1. Talks to the FastAPI backend via HTTP, not by importing skygent modules
   directly. This keeps the UI layer decoupled — the dashboard could run
   on a different machine from the backend. All data goes through the API.

2. st.session_state for the API base URL: allows the user to point the
   dashboard at a different backend (e.g. a deployed Railway instance)
   without restarting Streamlit.

3. No authentication for MVP: matches the API's own no-auth stance.
   The dashboard is for local/single-user use.

4. Auto-refresh via st.rerun() + a manual Refresh button: Streamlit does
   not have a built-in polling mechanism. We offer a manual refresh button
   rather than a timer-based rerun to avoid hammering the API during
   development. Known tradeoff: the Settings sidebar reruns on every
   keystroke while editing the URL — inherent Streamlit behavior, acceptable
   for a local MVP.

5. Error display with st.error() rather than exceptions: a failed API call
   shows a user-friendly message, not a Python traceback.

6. datetime.fromisoformat() is used to parse API timestamps. Python 3.11+
   handles the Z suffix correctly; earlier versions require .replace("Z", "+00:00").
   Skygent targets Python 3.11+ so this is safe, but documented here for
   awareness if the dashboard is ever run on an older interpreter.

7. Free-text fields (notes) are rendered with st.text(), not st.markdown(),
   to avoid user-supplied text accidentally injecting markdown formatting.

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import requests
import streamlit as st
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Skygent",
    page_icon="⛅",
    layout="wide",
)

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL


def api(path: str) -> str:
    return f"{st.session_state.api_url}{path}"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get(path: str) -> dict | list | None:
    """GET from the API. Returns parsed JSON or None on error."""
    try:
        resp = requests.get(api(path), timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to Skygent API at **{st.session_state.api_url}**. "
            "Is the server running?  \n`uvicorn skygent.api.main:app --port 8000`"
        )
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text[:200]}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def post(path: str, payload: dict) -> dict | None:
    """POST to the API. Returns parsed JSON or None on error."""
    try:
        resp = requests.post(api(path), json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        # Prefer the FastAPI "detail" field; fall back to raw text if
        # the error body is not JSON (e.g. a proxy returning HTML).
        try:
            detail = e.response.json().get("detail", e.response.text[:200])
        except Exception:
            detail = e.response.text[:200]
        st.error(f"API error: {detail}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {st.session_state.api_url}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def delete(path: str) -> bool:
    """DELETE from the API. Returns True on success."""
    try:
        resp = requests.delete(api(path), timeout=5)
        resp.raise_for_status()
        return True
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to Skygent API at **{st.session_state.api_url}**. "
            "Is the server running?"
        )
        return False
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text[:200]}")
        return False
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return False


# ---------------------------------------------------------------------------
# Confidence badge helper
# ---------------------------------------------------------------------------

_CONFIDENCE_COLOR = {"high": "🟢", "medium": "🟡", "low": "🔴"}


def confidence_badge(c: str) -> str:
    return f"{_CONFIDENCE_COLOR.get(c, '⚪')} {c.capitalize()}"


# ---------------------------------------------------------------------------
# Page: Register Event
# ---------------------------------------------------------------------------

def page_register():
    st.header("Register new event")
    st.caption("Add an event to monitor. The first forecast fetch runs immediately.")

    with st.form("register_form"):
        name = st.text_input("Event name", placeholder="Ana & Juan's Wedding")

        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0,
                                  value=-34.9011, format="%.4f")
            event_date = st.date_input(
                "Event date",
                value=datetime.now(timezone.utc).date() + timedelta(days=30),
            )
            check_interval = st.slider("Check interval (hours)", 1, 24, 6)
        with col2:
            lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0,
                                  value=-56.1645, format="%.4f")
            event_time = st.time_input("Event time (UTC)", value=datetime.strptime("17:00", "%H:%M").time())
            duration = st.slider("Event duration (hours)", 1, 12, 4)

        context = st.selectbox(
            "Context",
            ["social_event", "agriculture", "energy", "logistics"],
            help="Tailors the alert narrative tone",
        )
        notes = st.text_area(
            "Notes (optional)",
            placeholder="Outdoor ceremony, no tent backup",
            height=80,
        )

        submitted = st.form_submit_button("Register event", type="primary")

    if submitted:
        if not name:
            st.warning("Event name is required.")
            return

        event_dt = datetime.combine(event_date, event_time).replace(tzinfo=timezone.utc)

        if event_dt <= datetime.now(timezone.utc):
            st.warning("Event datetime must be in the future.")
            return

        payload = {
            "name": name,
            "latitude": lat,
            "longitude": lon,
            "event_datetime": event_dt.isoformat(),
            "check_interval_hours": check_interval,
            "event_duration_hours": duration,
            "context": context,
            "notes": notes,
        }

        result = post("/api/v1/profiles", payload)
        if result:
            st.success(
                f"✅ **{result['name']}** registered! "
                f"First forecast fetch queued immediately.  \n"
                f"Profile ID: `{result['id']}`"
            )


# ---------------------------------------------------------------------------
# Page: Active Profiles
# ---------------------------------------------------------------------------

def page_profiles():
    st.header("Active profiles")

    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()

    status = get("/api/v1/status")
    if status:
        m1, m2, m3 = st.columns(3)
        m1.metric("Active profiles", status["active_profiles"])
        m2.metric("Scheduled jobs", status["scheduled_jobs"])
        m3.metric(
            "API status",
            "🟢 Online" if status["status"] == "ok" else "🔴 Error",
        )

    st.divider()

    profiles = get("/api/v1/profiles?active_only=true")
    if profiles is None:
        return

    if not profiles:
        st.info("No active profiles. Register an event to start monitoring.")
        return

    for p in profiles:
        event_dt = datetime.fromisoformat(p["event_datetime"])
        days_left = (event_dt - datetime.now(timezone.utc)).days

        with st.expander(f"📍 {p['name']}  —  {days_left}d to event", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Location**  \n{p['location'][0]:.4f}, {p['location'][1]:.4f}")
            c2.markdown(f"**Event date**  \n{event_dt.strftime('%b %d, %Y %H:%M')} UTC")
            c3.markdown(f"**Check interval**  \n{p['check_interval_hours']}h")

            c4, c5 = st.columns(2)
            c4.markdown(f"**Context**  \n{p['context']}")
            c5.markdown(f"**Event duration**  \n{p['event_duration_hours']}h")

            if p.get("notes"):
                st.markdown("**Notes:**")
                st.text(p["notes"])

            st.markdown(f"**Profile ID:** `{p['id']}`")

            if st.button(f"🗑️ Deregister", key=f"del_{p['id']}"):
                if delete(f"/api/v1/profiles/{p['id']}"):
                    st.success(f"'{p['name']}' deregistered.")
                    st.rerun()


# ---------------------------------------------------------------------------
# Page: Alerts
# ---------------------------------------------------------------------------

def page_alerts():
    st.header("Alert history")

    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        profile_filter = st.text_input("Filter by profile ID (optional)", "")
    with col2:
        limit = st.selectbox("Show", [20, 50, 100, 200], index=0)
    with col3:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh"):
            st.rerun()

    path = f"/api/v1/alerts?limit={limit}"
    if profile_filter.strip():
        path += f"&profile_id={profile_filter.strip()}"

    alerts = get(path)
    if alerts is None:
        return

    if not alerts:
        st.info("No alerts yet. Alerts appear here when forecast thresholds are crossed.")
        return

    st.caption(f"{len(alerts)} alert(s) shown")

    for alert in alerts:
        detected = datetime.fromisoformat(alert["detected_at"])
        badge = confidence_badge(alert["confidence"])
        sent_icon = "✅" if alert["sent"] else "⏳"

        with st.expander(
            f"{sent_icon} {detected.strftime('%b %d %H:%M')} UTC  —  "
            f"{badge}  —  {alert['triggering_summary']}",
            expanded=False,
        ):
            st.markdown(f"**Profile ID:** `{alert['profile_id']}`")
            st.markdown(f"**Alert ID:** `{alert['id']}`")

            c1, c2 = st.columns(2)
            c1.markdown(f"**Horizon:** {alert['horizon_days']:.1f} days")
            c2.markdown(f"**Sent:** {'Yes' if alert['sent'] else 'No'}")

            st.markdown("**Narrative:**")
            st.info(alert["narrative"] or "_No narrative generated_")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("⛅ Skygent")
st.sidebar.caption("AI Weather Monitoring Agent")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Register event", "Active profiles", "Alert history"],
    label_visibility="collapsed",
)

st.sidebar.divider()
with st.sidebar.expander("⚙️ Settings"):
    new_url = st.text_input("API URL", value=st.session_state.api_url)
    if new_url != st.session_state.api_url:
        st.session_state.api_url = new_url
        st.rerun()

# ---------------------------------------------------------------------------
# Render selected page
# ---------------------------------------------------------------------------

if page == "Register event":
    page_register()
elif page == "Active profiles":
    page_profiles()
elif page == "Alert history":
    page_alerts()