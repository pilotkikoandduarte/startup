"""BME Performance - Club dashboard.

Two views, sidebar-toggled:
- Team overview: one hero number (median readiness), squad list with status pills,
                 a horizontal bar chart so the coach can scan the squad at a glance.
- Individual player: drill-down rendered by bme_ui.render_player_view.
"""
import html as _html
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from developmentday2 import (
    TEAM_CSV_PATH,
    compute_all_metrics_team,
    load_or_generate_team_dataset,
)
import bme_ui as ui

st.set_page_config(page_title="BME Performance - Club", layout="wide",
                   initial_sidebar_state="expanded")
ui.apply_global_css()


# -- Data --
@st.cache_data(show_spinner=False, ttl=3600)
def _load_team():
    raw = load_or_generate_team_dataset(TEAM_CSV_PATH, n_athletes=25, samples=100)
    return compute_all_metrics_team(raw)

@st.cache_data(show_spinner=False)
def _get_roster(df):
    return sorted(df["Atleta_ID"].unique().tolist())

@st.cache_data(show_spinner=False)
def _get_max_day(df):
    return int(df["Dia"].max())

with st.spinner(""):
    DF = _load_team()

ROSTER = _get_roster(DF)
MAX_DAY = _get_max_day(DF)

ROSTER = _get_roster(DF)
MAX_DAY = _get_max_day(DF)


# -- Sidebar --
with st.sidebar:
    st.markdown(
        '<p style="font-size:0.75rem;color:#5F6368;letter-spacing:.16em;'
        'text-transform:uppercase;margin:0.5rem 0 0.2rem">BME Performance</p>'
        '<p style="font-size:1.3rem;font-weight:700;margin:0 0 1rem">Club</p>',
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

    if st.checkbox("Moon Dark mode", value=st.session_state.get("dark_mode", False)):
        st.session_state.dark_mode = True
    else:
        st.session_state.dark_mode = False

    view = st.radio("View", ["Team overview", "Individual player"])

    selected_player = None
    if view == "Individual player":
        selected_player = st.selectbox("Player", ROSTER, index=0)

    status_filter = None
    readiness_filter = None
    if view == "Team overview":
        st.divider()
        st.markdown("**Filters**")
        status_filter = st.multiselect("Status", ["OK", "Light", "Rest"], default=["OK", "Light", "Rest"])
        readiness_filter = st.slider("Min readiness", 0, 100, 0)

    st.markdown(f'<p style="font-size:0.85rem;color:#5F6368;margin-top:1rem">'
                f'Squad: {len(ROSTER)} players - Day {day} of {MAX_DAY}</p>',
                unsafe_allow_html=True)


# -- Helpers --
def _today_snapshot(df, day):
    """One row per athlete for the given day (or the last day each athlete has)."""
    snap = df[df["Dia"] == day].copy()
    if snap.empty:
        snap = df.groupby("Atleta_ID", as_index=False).last()
    return snap.reset_index(drop=True)


def _overall_status(row):
    """One-word overall status per player. Worst-wins across pillars."""
    if row["Readiness_Score"] < 50 or row["ACWR"] > 1.5 or row["Injury_Risk_Score"] >= 60:
        return ("Rest", ui.RISK)
    if row["Readiness_Score"] < 67 or row["ACWR"] > 1.3 or row["Injury_Risk_Score"] >= 35:
        return ("Light", ui.WARN)
    return ("OK", ui.OK)


# -- Team overview --
def render_team_overview(day, status_filter=None, readiness_filter=None):
    snap = _today_snapshot(DF, day)

    # Compute team-level aggregates
    statuses = snap.apply(_overall_status, axis=1)
    snap["__status_label"] = [s[0] for s in statuses]
    snap["__status_color"] = [s[1] for s in statuses]

    # Apply filters
    if status_filter:
        snap = snap[snap["__status_label"].isin(status_filter)].reset_index(drop=True)
    if readiness_filter and readiness_filter > 0:
        snap = snap[snap["Readiness_Score"] >= readiness_filter].reset_index(drop=True)

    median_readiness = float(snap["Readiness_Score"].median()) if len(snap) > 0 else 0
    needs_attention = int((snap["__status_label"] != "OK").sum())
    rest_count = int((snap["__status_label"] == "Rest").sum())

    # Header + PDF export
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(
            f'<h2 style="margin:0 0 0.25rem;font-size:1.3rem">Team - Day {day}</h2>'
            f'<p class="bme-hero-caption" style="margin:0 0 1.5rem">'
            f'{len(snap)} players in the squad</p>',
            unsafe_allow_html=True)
    with col2:
        pdf_data = ui.export_team_overview_pdf(snap, day, MAX_DAY, median_readiness, needs_attention, rest_count)
        st.download_button(
            label="PDF",
            data=pdf_data,
            file_name=f"bme_team_day{day}.pdf",
            mime="application/pdf",
            key=f"export_team_{day}"
        )

    # Hero - one primary number for the team
    if rest_count >= 3:
        team_label, team_color = "High strain across squad", ui.RISK
    elif needs_attention >= 5:
        team_label, team_color = "Several players need attention", ui.WARN
    elif needs_attention >= 1:
        team_label, team_color = "Mostly fit", ui.OK
    else:
        team_label, team_color = "Squad fully fit", ui.OK

    attention_phrase = (
        "All players cleared for full training." if needs_attention == 0
        else f"{needs_attention} player{'s' if needs_attention != 1 else ''} need attention "
             f"({rest_count} flagged for rest)."
    )

    st.markdown(
        f'<div>'
        f'<div class="bme-hero-caption" style="margin-bottom:0.3rem">Median squad readiness</div>'
        f'<div><span class="bme-hero-num">{median_readiness:.0f}</span>'
        f'<span class="bme-hero-unit">/ 100</span></div>'
        f'<div class="bme-hero-status" style="color:{team_color}">{team_label}</div>'
        f'<div class="bme-hero-caption">{attention_phrase}</div>'
        f'</div>', unsafe_allow_html=True)

    # Squad bar chart - every player on one axis, sorted so risks float to top
    st.markdown('<div class="bme-section">Squad readiness</div>', unsafe_allow_html=True)
    theme = ui.get_theme()
    chart_df = snap.sort_values("Readiness_Score", ascending=True).reset_index(drop=True)
    bar_colors = chart_df["__status_color"].tolist()
    fig = go.Figure(go.Bar(
        x=chart_df["Readiness_Score"], y=chart_df["Atleta_ID"],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=chart_df["Readiness_Score"].round(0).astype(int).astype(str),
        textposition="outside", textfont=dict(color=theme["TEXT"], size=11),
        hovertemplate="%{y}: %{x:.0f} / 100<extra></extra>",
    ))
    fig.add_vline(x=67, line_dash="dash", line_color=theme["OK"], line_width=1, opacity=0.3)
    fig.add_vline(x=50, line_dash="dash", line_color=theme["RISK"], line_width=1, opacity=0.3)
    fig.update_layout(
        paper_bgcolor=theme["BG"], plot_bgcolor=theme["BG"],
        font=dict(family=ui.FONT, color=theme["TEXT_MUTED"], size=11),
        margin=dict(l=8, r=30, t=8, b=8),
        height=max(360, 18 * len(chart_df)),
        xaxis=dict(range=[0, 110], gridcolor=theme["BORDER"], zeroline=False, showline=False,
                   tickfont=dict(size=11, color=theme["TEXT_MUTED"]), title=None),
        yaxis=dict(gridcolor=theme["BG"], zeroline=False, showline=False,
                   tickfont=dict(size=11, color=theme["TEXT"]), title=None),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Squad table - sortable
    st.markdown('<div class="bme-section">Squad table</div>', unsafe_allow_html=True)
    table = pd.DataFrame({
        "Player":     snap["Atleta_ID"],
        "Readiness":  snap["Readiness_Score"].round(0).astype(int),
        "Recovery %": snap["Recovery_Score"].round(0).astype(int),
        "ACWR":       snap["ACWR"].round(2),
        "Injury risk": snap["Injury_Risk_Score"].round(0).astype(int),
        "Sleep h":    snap["Horas_Sono"].round(1),
        "Status":     snap["__status_label"],
    }).sort_values("Readiness", ascending=True).reset_index(drop=True)

    def _style(s):
        return [f"color: {theme['RISK']}; font-weight: 600" if v == "Rest"
                else f"color: {theme['WARN']}; font-weight: 600" if v == "Light"
                else f"color: {theme['OK']}; font-weight: 600" for v in s]

    st.dataframe(
        table.style.apply(_style, subset=["Status"]),
        use_container_width=True, hide_index=True,
    )

    st.caption("To inspect a player: switch the sidebar to **Individual player** and pick from the list.")


# -- Individual player view --
def render_player(day, player_id):
    player_df = DF[DF["Atleta_ID"] == player_id].reset_index(drop=True)

    # Header + PDF export
    col1, col2 = st.columns([4, 1])
    with col1:
        player_display = _html.escape(player_id)
        header_html = f'<h2 style="margin:0 0 0.25rem;font-size:1.3rem">{player_display} - Day {day}</h2>'
        caption_html = '<p class="bme-hero-caption" style="margin:0 0 1rem">Switch back to Team overview in the sidebar to compare with the squad.</p>'
        st.markdown(header_html + caption_html, unsafe_allow_html=True)
    with col2:
        pdf_data = ui.export_player_view_pdf(
            player_df=player_df,
            day=day,
            header_label=f"{player_id} - Day {day}",
            subheader=f"Day {day} of {MAX_DAY}"
        )
        st.download_button(
            label="PDF",
            data=pdf_data,
            file_name=f"bme_player_{player_id}_day{day}.pdf",
            mime="application/pdf",
            key=f"export_player_{player_id}_{day}"
        )

    # Render metrics (without duplicate header)
    ui.render_player_view(
        player_df=player_df,
        day=day,
        header_label="",
        subheader=None,
    )


# -- Route --
if view == "Team overview":
    render_team_overview(day, status_filter, readiness_filter)
else:
    render_player(day, selected_player)
