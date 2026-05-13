"""BME Performance — Individual player dashboard.

Same drill-down view as the club's player view, but the athlete is fixed at
startup. The athlete cannot be changed from the UI.

In a real product, the binding to a single athlete must be enforced by
authentication (one account = one Atleta_ID). For this prototype, the binding
is via the BME_PLAYER_ID environment variable, defaulting to "P01".

Run:
    py -m streamlit run player_app.py
    # or, for a specific player:
    $env:BME_PLAYER_ID = "P07"; py -m streamlit run player_app.py
"""
import os
from datetime import datetime
import streamlit as st

from developmentday2 import (
    TEAM_CSV_PATH,
    compute_all_metrics_team,
    load_or_generate_team_dataset,
)
import bme_ui as ui

st.set_page_config(page_title="BME Performance", layout="centered",
                   initial_sidebar_state="collapsed")
ui.apply_global_css()

# ── Auth boundary (prototype) ────────────────────────────────────────────────
PLAYER_ID = os.environ.get("BME_PLAYER_ID", "P01")


# ── Data — load + filter immediately to the bound athlete ───────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def _load_for(player_id):
    raw = load_or_generate_team_dataset(TEAM_CSV_PATH, n_athletes=25, samples=100)
    if player_id not in set(raw["Atleta_ID"].unique()):
        return None
    raw = raw[raw["Atleta_ID"] == player_id].reset_index(drop=True)
    return compute_all_metrics_team(raw)


with st.spinner(""):
    player_df = _load_for(PLAYER_ID)

if player_df is None:
    st.error(f"No data found for player {PLAYER_ID}.")
    st.stop()


# ── Sidebar — day picker + dark mode toggle ──────────────────────────────────
MAX_DAY = int(player_df["Dia"].max())
with st.sidebar:
    st.markdown(
        '<p style="font-size:0.75rem;color:#5F6368;letter-spacing:.16em;'
        'text-transform:uppercase;margin:0.5rem 0 0.2rem">BME Performance</p>'
        '<p style="font-size:1.2rem;font-weight:700;margin:0 0 1rem">My account</p>',
        unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        day = st.slider("Day", 1, MAX_DAY, MAX_DAY, label_visibility="collapsed")
    with col2:
        if st.button("Today"):
            day = MAX_DAY
    st.markdown(f'<p style="font-size:0.85rem;color:#5F6368;margin:0.5rem 0">'
                f'{datetime.now().strftime("%b %d, %Y")}</p>',
                unsafe_allow_html=True)

    st.divider()

    if st.checkbox("🌙 Dark mode", value=st.session_state.get("dark_mode", False)):
        st.session_state.dark_mode = True
    else:
        st.session_state.dark_mode = False


# ── Render ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown(
        f'<h2 style="margin:0 0 0.25rem;font-size:1.3rem">My performance</h2>'
        f'<p class="bme-hero-caption" style="margin:0 0 1rem">Day {day} of {MAX_DAY}</p>',
        unsafe_allow_html=True)
with col2:
    pdf_data = ui.export_player_view_pdf(
        player_df=player_df,
        day=day,
        header_label=f"My performance · Day {day}",
        subheader=f"Day {day} of {MAX_DAY}"
    )
    st.download_button(
        label="📥 PDF",
        data=pdf_data,
        file_name=f"bme_performance_day{day}.pdf",
        mime="application/pdf",
        key=f"export_player_{day}"
    )

ui.render_player_view(
    player_df=player_df,
    day=day,
    header_label="",
    subheader=None,
)
