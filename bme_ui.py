"""Shared UI module for BME Performance dashboards.

Both app.py (Club) and player_app.py (Individual) import from here so the
player drill-down view is rendered identically in both products.
"""
import html as _html
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from fpdf import FPDF
import io

# ── Design tokens (light/dark modes) ────────────────────────────────────────
LIGHT_THEME = {
    "BG": "#FFFFFF",
    "SURFACE": "#F8F9FA",
    "BORDER": "#E0E0E5",
    "TEXT": "#1F1F23",
    "TEXT_MUTED": "#5F6368",
    "OK": "#137333",
    "WARN": "#B86E00",
    "RISK": "#B3261E",
    "ACCENT": "#1A56DB",
}

DARK_THEME = {
    "BG": "#121212",
    "SURFACE": "#1E1E1E",
    "BORDER": "#2A2A2A",
    "TEXT": "#E8E8ED",
    "TEXT_MUTED": "#9E9EA3",
    "OK": "#81C995",
    "WARN": "#FFB74D",
    "RISK": "#EF9A9A",
    "ACCENT": "#64B5F6",
}

FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

def get_theme():
    """Return current theme colors based on session state."""
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    return DARK_THEME if st.session_state.dark_mode else LIGHT_THEME

BG = LIGHT_THEME["BG"]
SURFACE = LIGHT_THEME["SURFACE"]
BORDER = LIGHT_THEME["BORDER"]
TEXT = LIGHT_THEME["TEXT"]
TEXT_MUTED = LIGHT_THEME["TEXT_MUTED"]
OK = LIGHT_THEME["OK"]
WARN = LIGHT_THEME["WARN"]
RISK = LIGHT_THEME["RISK"]
ACCENT = LIGHT_THEME["ACCENT"]

# ── Threshold configuration ────────────────────────────────────────────────
THRESHOLDS = {
    "readiness_ok": 67,
    "readiness_light": 50,
    "acwr_danger": 1.5,
    "acwr_caution": 1.3,
    "acwr_safe_min": 0.8,
    "injury_high": 60,
    "injury_moderate": 35,
    "recovery_good": 67,
    "recovery_fair": 50,
    "sleep_debt_catch": 3,
    "sleep_debt_watch": 1,
    "hrv_baseline_good": 0,
    "hrv_baseline_warn": -10,
    "asymmetry_threshold": 10,
}


def apply_global_css():
    """Render theme-aware CSS for the whole app. Call once at the top of each entry point."""
    theme = get_theme()
    st.markdown(f"""<style>
    html, body, [data-testid="stApp"] {{
        background: {theme["BG"]} !important;
        color: {theme["TEXT"]} !important;
        font-family: {FONT} !important;
    }}
    [data-testid="stSidebar"] {{
        background: {theme["SURFACE"]} !important;
        border-right: 1px solid {theme["BORDER"]} !important;
    }}
    [data-testid="stSidebar"] * {{ font-family: {FONT} !important; color: {theme["TEXT"]}; }}
    .block-container {{ padding: 2.5rem 3rem 4rem !important; max-width: 1180px !important; }}
    @media (max-width: 768px) {{ .block-container {{ padding: 1.5rem 1rem 2rem !important; }} }}
    #MainMenu, footer, header, .stDeployButton {{ display: none !important; }}
    h1, h2, h3, h4 {{ color: {theme["TEXT"]}; font-weight: 600; letter-spacing: -0.01em; }}
    .stMarkdown, .stMarkdown p, .stCaption {{ color: {theme["TEXT"]}; }}
    .stRadio label, .stSelectbox label, .stSlider label {{ color: {theme["TEXT"]} !important; font-weight: 500; }}

    /* Hero section — compact for Phase 2 */
    .bme-hero-container {{ background: {theme["SURFACE"]}; border-radius: 8px; padding: 1.5rem; margin-bottom: 2.5rem; }}
    .bme-hero-num     {{ font-size: 5rem; font-weight: 700; line-height: 1; margin: 0;
                          font-feature-settings: "tnum"; letter-spacing: -0.03em; color: {theme["TEXT"]}; }}
    .bme-hero-unit    {{ font-size: 1.1rem; color: {theme["TEXT_MUTED"]}; font-weight: 400; margin-left: 0.4rem; }}
    .bme-hero-status  {{ font-size: 1.2rem; font-weight: 700; }}
    .bme-hero-caption {{ font-size: 0.95rem; color: {theme["TEXT_MUTED"]}; line-height: 1.5; }}

    /* Section titles */
    .bme-section      {{ font-size: 1.05rem; font-weight: 700; color: {theme["TEXT"]};
                          margin: 3rem 0 1.2rem 0; padding-top: 1.5rem; border-top: 2px solid {theme["BORDER"]}; }}
    .bme-section:first-child {{ border-top: none; padding-top: 0; margin-top: 0; }}

    /* Metric rows with more breathing */
    .bme-metric-row   {{ display: flex; justify-content: space-between; align-items: baseline;
                          padding: 1.1rem 0; border-bottom: 1px solid {theme["BORDER"]}; }}
    .bme-metric-row:last-child {{ border-bottom: none; }}
    .bme-metric-left  {{ display: flex; flex-direction: column; flex: 1; }}
    .bme-metric-label {{ font-size: 0.98rem; color: {theme["TEXT"]}; font-weight: 600; }}
    .bme-metric-note  {{ font-size: 0.85rem; color: {theme["TEXT_MUTED"]}; margin-top: 0.2rem; }}
    .bme-metric-right {{ display: flex; align-items: center; gap: 0.8rem; margin-left: 1rem; flex-shrink: 0; }}
    .bme-metric-val   {{ font-size: 1.25rem; color: {theme["TEXT"]}; font-feature-settings: "tnum"; font-weight: 600; }}

    /* Status pills — bigger and more prominent */
    .bme-pill         {{ font-size: 0.82rem; font-weight: 700; padding: 0.35rem 0.75rem;
                          border-radius: 6px; display: inline-block; }}

    /* Legend box */
    .bme-legend       {{ background: {theme["SURFACE"]}; border-left: 4px solid {theme["ACCENT"]};
                          padding: 1rem; margin: 1.5rem 0; border-radius: 4px; font-size: 0.9rem; color: {theme["TEXT_MUTED"]}; }}
    .bme-legend-item  {{ margin: 0.5rem 0; }}
    .bme-legend-label {{ font-weight: 600; display: inline-block; min-width: 60px; }}
    </style>""", unsafe_allow_html=True)


# ── Status helpers (always return label AND colour) ─────────────────────────
def readiness_status(v):
    if v >= THRESHOLDS["readiness_ok"]: return ("OK to train", OK)
    if v >= THRESHOLDS["readiness_light"]: return ("Train light", WARN)
    return ("Rest", RISK)

def acwr_status(v):
    if v > THRESHOLDS["acwr_danger"]: return ("Danger", RISK)
    if v > THRESHOLDS["acwr_caution"]: return ("Caution", WARN)
    if v >= THRESHOLDS["acwr_safe_min"]: return ("Safe", OK)
    return ("Under-loaded", WARN)

def injury_status(v):
    if v >= THRESHOLDS["injury_high"]: return ("High", RISK)
    if v >= THRESHOLDS["injury_moderate"]: return ("Moderate", WARN)
    return ("Low", OK)

def recovery_status(v):
    if v >= THRESHOLDS["recovery_good"]: return ("Good", OK)
    if v >= THRESHOLDS["recovery_fair"]: return ("Fair", WARN)
    return ("Poor", RISK)

def sleep_debt_status(v):
    if v > THRESHOLDS["sleep_debt_catch"]: return ("Catch up", RISK)
    if v > THRESHOLDS["sleep_debt_watch"]: return ("Watch", WARN)
    return ("OK", OK)

def hrv_vs_baseline(hrv, baseline):
    diff = hrv - baseline
    if diff >= THRESHOLDS["hrv_baseline_good"]:
        return (f"{diff:+.0f} ms above baseline", OK)
    if diff >= THRESHOLDS["hrv_baseline_warn"]:
        return (f"{diff:+.0f} ms below baseline", WARN)
    return (f"{diff:+.0f} ms below baseline", RISK)


def _pill(label, color):
    return (f'<span class="bme-pill" '
            f'style="background:{color}14;color:{color};border:1px solid {color}40">'
            f'{_html.escape(label)}</span>')


def metric_row(label, value, note=None, status=None, color=None):
    """One vertical row: 'Label (note) ........ value   [status pill]'."""
    pill = _pill(status, color) if status and color else ""
    note_html = f'<div class="bme-metric-note">{_html.escape(note)}</div>' if note else ""
    return (f'<div class="bme-metric-row">'
            f'<div class="bme-metric-left">'
            f'<div class="bme-metric-label">{_html.escape(label)}</div>{note_html}</div>'
            f'<div class="bme-metric-right">'
            f'<span class="bme-metric-val">{value}</span>{pill}</div>'
            f'</div>')


def readiness_trend_chart(df, current_day=None, height=380):
    """Clean line chart with Readiness over time. No decorative fills."""
    theme = get_theme()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Dia"], y=df["Readiness_Score"],
        mode="lines+markers",
        line=dict(color=theme["ACCENT"], width=3),
        marker=dict(size=6, color=theme["ACCENT"]),
        showlegend=False, hovertemplate="Day %{x}: %{y:.0f}<extra></extra>",
    ))
    if current_day is not None:
        fig.add_vline(x=current_day, line_dash="dot", line_color=theme["TEXT_MUTED"], line_width=1.5, opacity=0.6)
    fig.add_hline(y=67, line_dash="dash", line_color=theme["OK"], line_width=1, opacity=0.3)
    fig.add_hline(y=50, line_dash="dash", line_color=theme["RISK"], line_width=1, opacity=0.3)
    fig.update_layout(
        paper_bgcolor=theme["BG"], plot_bgcolor=theme["BG"],
        font=dict(family=FONT, color=theme["TEXT_MUTED"], size=12),
        margin=dict(l=8, r=8, t=8, b=8), height=height,
        xaxis=dict(gridcolor=theme["BORDER"], zeroline=False, showline=False, title=None,
                   tickfont=dict(size=12, color=theme["TEXT_MUTED"])),
        yaxis=dict(range=[0, 100], gridcolor=theme["BORDER"], zeroline=False, showline=False, title=None,
                   tickfont=dict(size=12, color=theme["TEXT_MUTED"])),
    )
    return fig


def _row_for_day(player_df, day):
    rows = player_df[player_df["Dia"] == day]
    return rows.iloc[0] if not rows.empty else player_df.iloc[-1]


def render_player_view(player_df, day, header_label, subheader=None):
    """Render the full single-player drill-down: hero + metric rows + chart.

    player_df: DataFrame filtered to ONE athlete, with all metrics already computed.
    day: int — currently selected day
    header_label: str — what to put as the page header (e.g. "P03" or "My Performance")
    subheader: optional small caption under the header
    """
    if player_df.empty:
        st.warning("No data available for this player.")
        return

    row = _row_for_day(player_df, day)
    history = player_df[player_df["Dia"] <= day]

    def f(col, default=0.0):
        v = row.get(col, default)
        try: return default if pd.isna(v) else float(v)
        except Exception: return default

    readiness = f("Readiness_Score", 50)
    recovery  = f("Recovery_Score", 50)
    acwr      = f("ACWR", 1.0)
    inj       = f("Injury_Risk_Score", 0)
    hrv       = f("HRV_ms", 70)
    hrv_b     = f("HRV_Media_Base", hrv)
    sono      = f("Horas_Sono", 7)
    sonop     = f("Sono_Profundo_h", 1.5)
    debt      = f("Sleep_Debt_Acumulado_h", 0)
    assim     = f("Assimetria_Percent", 0)

    r_label, r_color = readiness_status(readiness)
    a_label, a_color = acwr_status(acwr)

    # Header (only render if header_label is not empty)
    if header_label:
        st.markdown(
            f'<h2 style="margin:0 0 0.25rem;font-size:1.3rem">{_html.escape(header_label)}</h2>'
            + (f'<p class="bme-hero-caption" style="margin:0 0 1rem">{_html.escape(subheader)}</p>' if subheader else ''),
            unsafe_allow_html=True)

    # Hero — one primary number + status (more compact for Phase 2)
    if acwr > 1.5:
        explanation = "High injury risk from overtraining."
    elif readiness < 50:
        explanation = "Need more recovery before training."
    elif readiness < 67:
        explanation = "Reduce intensity today."
    else:
        explanation = "Ready for full intensity training."

    st.markdown(
        f'<div style="margin-bottom:2rem">'
        f'<div class="bme-hero-caption" style="margin-bottom:0.3rem">Readiness today</div>'
        f'<div style="margin-bottom:0.6rem"><span class="bme-hero-num">{readiness:.0f}</span>'
        f'<span class="bme-hero-unit">/ 100</span></div>'
        f'<div class="bme-hero-status" style="color:{r_color};margin-bottom:0.3rem">{r_label}</div>'
        f'<div class="bme-hero-caption">{_html.escape(explanation)}</div>'
        f'</div>', unsafe_allow_html=True)

    # Chart — now primary (moved up, larger, more prominent)
    st.markdown('<div class="bme-section">Readiness trend</div>', unsafe_allow_html=True)
    chart_df = history.tail(30) if len(history) > 30 else history
    st.plotly_chart(readiness_trend_chart(chart_df, current_day=day),
                    use_container_width=True, config={"displayModeBar": False})

    # Supporting metrics (now secondary, below chart)
    rec_label, rec_color = recovery_status(recovery)
    inj_label, inj_color = injury_status(inj)
    debt_label, debt_color = sleep_debt_status(debt)
    hrv_note, hrv_color = hrv_vs_baseline(hrv, hrv_b)

    theme = get_theme()
    rows_html = "".join([
        metric_row("Recovery score", f"{recovery:.0f} %",
                   note="HRV + sleep + load composite",
                   status=rec_label, color=rec_color),
        metric_row("ACWR", f"{acwr:.2f}",
                   note=f"7-day vs 28-day load · safe range 0.8–1.3",
                   status=a_label, color=a_color),
        metric_row("Heart-rate variability", f"{hrv:.0f} ms",
                   note=hrv_note,
                   status=("Above baseline" if hrv >= hrv_b else "Below baseline"),
                   color=hrv_color),
        metric_row("Sleep last night", f"{sono:.1f} h",
                   note=f"{sonop:.1f} h deep ({sonop/max(sono, 0.1)*100:.0f} % of total)"),
        metric_row("Sleep debt (7-day)", f"{debt:+.1f} h",
                   note="Cumulative deficit vs sleep need",
                   status=debt_label, color=debt_color),
        metric_row("Injury risk", f"{inj:.0f} / 100",
                   note="ACWR + asymmetry + fatigue + temperature",
                   status=inj_label, color=inj_color),
        metric_row("Muscular asymmetry", f"{assim:.1f} %",
                   note="Above 10 % flags biomechanical concern",
                   status=("Above threshold" if assim > 10 else "Within range"),
                   color=(theme["RISK"] if assim > 10 else theme["OK"])),
    ])
    st.markdown('<div class="bme-section">Today\'s key metrics</div>', unsafe_allow_html=True)
    st.markdown(f'<div>{rows_html}</div>', unsafe_allow_html=True)


def export_team_overview_pdf(snap, day, max_day, median_readiness, needs_attention, rest_count):
    """Generate PDF export of team overview."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "BME Performance - Team Overview", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Day {day} of {max_day} | {datetime.now().strftime('%b %d, %Y')}", ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Squad Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Total squad: {len(snap)} players", ln=True)
    pdf.cell(0, 6, f"Median readiness: {median_readiness:.0f} / 100", ln=True)
    pdf.cell(0, 6, f"Players needing attention: {needs_attention}", ln=True)
    pdf.cell(0, 6, f"Players flagged for rest: {rest_count}", ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Squad List", ln=True)
    pdf.set_font("Helvetica", "", 9)

    col_width = pdf.w / 5 - 2
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(col_width, 6, "Player", border=1)
    pdf.cell(col_width, 6, "Readiness", border=1)
    pdf.cell(col_width, 6, "Recovery", border=1)
    pdf.cell(col_width, 6, "Injury Risk", border=1)
    pdf.cell(col_width, 6, "Status", border=1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for _, row in snap.iterrows():
        pdf.cell(col_width, 6, str(row["Atleta_ID"])[:10], border=1)
        pdf.cell(col_width, 6, f"{row['Readiness_Score']:.0f}", border=1)
        pdf.cell(col_width, 6, f"{row['Recovery_Score']:.0f}%", border=1)
        pdf.cell(col_width, 6, f"{row['Injury_Risk_Score']:.0f}", border=1)
        status = row["__status_label"] if "__status_label" in row else "-"
        pdf.cell(col_width, 6, str(status)[:10], border=1)
        pdf.ln()

    return pdf.output(dest='S')


def export_player_view_pdf(player_df, day, header_label, subheader=None):
    """Generate PDF export of player view."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, header_label, ln=True)
    if subheader:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, subheader, ln=True)

    if player_df.empty:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, "No data available.", ln=True)
        return pdf.output(dest='S')

    row = player_df[player_df["Dia"] == day]
    if row.empty:
        row = player_df.iloc[-1:]
    row = row.iloc[0]

    def f(col, default=0.0):
        v = row.get(col, default)
        try: return default if pd.isna(v) else float(v)
        except Exception: return default

    readiness = f("Readiness_Score", 50)
    recovery = f("Recovery_Score", 50)
    acwr = f("ACWR", 1.0)
    inj = f("Injury_Risk_Score", 0)
    hrv = f("HRV_ms", 70)
    hrv_b = f("HRV_Media_Base", hrv)
    sono = f("Horas_Sono", 7)
    sonop = f("Sono_Profundo_h", 1.5)
    debt = f("Sleep_Debt_Acumulado_h", 0)
    assim = f("Assimetria_Percent", 0)

    r_label, _ = readiness_status(readiness)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Readiness Today", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Score: {readiness:.0f} / 100", ln=True)
    pdf.cell(0, 6, f"Status: {r_label}", ln=True)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Key Metrics", ln=True)
    pdf.set_font("Helvetica", "", 9)

    col_width = pdf.w / 3 - 2
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(col_width, 6, "Metric", border=1)
    pdf.cell(col_width * 1.5, 6, "Value", border=1)
    pdf.cell(col_width * 0.5, 6, "Status", border=1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    metrics = [
        ("Recovery", f"{recovery:.0f}%", "Good" if recovery >= 67 else "Fair" if recovery >= 50 else "Poor"),
        ("ACWR", f"{acwr:.2f}", "Safe" if acwr < 1.3 else "Caution" if acwr < 1.5 else "Danger"),
        ("HRV", f"{hrv:.0f} ms", "Above BL" if hrv >= hrv_b else "Below BL"),
        ("Sleep", f"{sono:.1f} h", f"{sonop/max(sono, 0.1)*100:.0f}% deep"),
        ("Sleep Debt", f"{debt:+.1f} h", "OK" if debt <= 1 else "Watch" if debt <= 3 else "Catch up"),
        ("Injury Risk", f"{inj:.0f}/100", "Low" if inj < 35 else "Moderate" if inj < 60 else "High"),
        ("Asymmetry", f"{assim:.1f}%", "OK" if assim <= 10 else "Alert"),
    ]

    for metric, val, status in metrics:
        pdf.cell(col_width, 6, metric, border=1)
        pdf.cell(col_width * 1.5, 6, val, border=1)
        pdf.cell(col_width * 0.5, 6, status, border=1)
        pdf.ln()

    return pdf.output(dest='S')
