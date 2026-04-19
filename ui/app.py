"""
ui/app.py — Skygent Streamlit Dashboard
========================================

Minimal functional dashboard covering three things:
1. Register a new event for monitoring (with interactive map picker)
2. View active profiles and scheduler status
3. Browse generated alerts

Design decisions
----------------
1. Talks to the FastAPI backend via HTTP only — decoupled, configurable URL.
2. Map picker via streamlit-folium: user clicks to drop a pin, coordinates
   populate automatically. No Google Maps API key required.
3. st.session_state for map coordinates: persists the clicked location
   across Streamlit reruns without losing the pin.
4. st.text_area() for notes input (multiline), st.text() for notes display:
   input uses text_area for usability; display uses st.text() to prevent
   user-supplied markdown from injecting formatting in the profiles page.
5. Manual refresh buttons instead of timer-based rerun: avoids hammering
   the API during development.

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import requests
import streamlit as st
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

from streamlit_folium import st_folium
import folium

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_LAT = -34.9011   # Montevideo
DEFAULT_LON = -56.1645

st.set_page_config(
    page_title="Skygent",
    page_icon="⛅",
    layout="wide",
)

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL
if "map_lat" not in st.session_state:
    st.session_state.map_lat = DEFAULT_LAT
if "map_lon" not in st.session_state:
    st.session_state.map_lon = DEFAULT_LON


def api(path: str) -> str:
    return f"{st.session_state.api_url}{path}"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get(path: str) -> dict | list | None:
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
    try:
        resp = requests.post(api(path), json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
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
# Helpers
# ---------------------------------------------------------------------------

_CONFIDENCE_COLOR = {"high": "🟢", "medium": "🟡", "low": "🔴"}

def confidence_badge(c: str) -> str:
    return f"{_CONFIDENCE_COLOR.get(c, '⚪')} {c.capitalize()}"


# ---------------------------------------------------------------------------
# Page: Register Event
# ---------------------------------------------------------------------------

def page_register():
    st.header("Register new event")
    st.caption("Click the map to place your event location, then fill in the details below.")

    # ── Map picker ────────────────────────────────────────────────────────────
    m = folium.Map(
        location=[st.session_state.map_lat, st.session_state.map_lon],
        zoom_start=10,
        tiles="CartoDB positron",
    )

    # Show current pin if already set
    # Font Awesome icons may not render in all Leaflet/Folium environments.
    # Fall back to the default marker if icon construction fails.
    try:
        marker_icon = folium.Icon(color="blue", icon="map-pin", prefix="fa")
    except Exception:
        marker_icon = folium.Icon(color="blue")

    folium.Marker(
        location=[st.session_state.map_lat, st.session_state.map_lon],
        tooltip="Event location",
        icon=marker_icon,
    ).add_to(m)

    map_result = st_folium(
        m,
        width="100%",
        height=350,
        returned_objects=["last_clicked"],
        key="register_map",
    )

    # Update coordinates when user clicks the map
    if map_result and map_result.get("last_clicked"):
        clicked = map_result["last_clicked"]
        st.session_state.map_lat = round(clicked["lat"], 6)
        st.session_state.map_lon = round(clicked["lng"], 6)
        st.rerun()

    # Show selected coordinates
    col_lat, col_lon = st.columns(2)
    col_lat.metric("Latitude", f"{st.session_state.map_lat:.4f}")
    col_lon.metric("Longitude", f"{st.session_state.map_lon:.4f}")

    st.caption("Click anywhere on the map to move the pin. Zoom in for precision.")
    st.divider()

    # ── Event details form ────────────────────────────────────────────────────
    with st.form("register_form"):
        name = st.text_input("Event name", placeholder="Ana & Juan's Wedding")

        col1, col2 = st.columns(2)
        with col1:
            event_date = st.date_input(
                "Event date",
                value=datetime.now(timezone.utc).date() + timedelta(days=30),
            )
            check_interval = st.slider("Check interval (hours)", 1, 24, 6)
        with col2:
            event_time = st.time_input(
                "Event time (UTC)",
                value=datetime.strptime("17:00", "%H:%M").time(),
            )
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
            "latitude": st.session_state.map_lat,
            "longitude": st.session_state.map_lon,
            "event_datetime": event_dt.isoformat(),
            "check_interval_hours": check_interval,
            "event_duration_hours": duration,
            "context": context,
            "notes": notes,
        }

        result = post("/api/v1/profiles", payload)
        if result:
            st.success(
                f"✅ **{result['name']}** registered at "
                f"({result['location'][0]:.4f}, {result['location'][1]:.4f}).  \n"
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

        with st.expander(
            f"📍 {p['name']}  —  {days_left}d to event",
            expanded=False,
        ):
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
        path += f"&profile_id={quote(profile_filter.strip())}"

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