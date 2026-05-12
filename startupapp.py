import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="BME · Performance AI", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

BG = "#0a0a0a"; BD = "#1e1e1e"
T2 = "#888";    T3 = "#444"
G  = "#00e676"; A  = "#ffb300"; R  = "#ff3d57"
BL = "#3d8bff"; PR = "#bf5af2"
FILL = {G:"rgba(0,230,118,.06)", A:"rgba(255,179,0,.06)",
        R:"rgba(255,61,87,.06)",  BL:"rgba(61,139,255,.06)", PR:"rgba(191,90,242,.06)"}

cg = lambda v: G if v >= 67 else A if v >= 33 else R
ca = lambda v: R if v > 1.5 else A if v > 1.3 else G if v >= 0.8 else A
ci = lambda v: R if v >= 60 else A if v >= 35 else G

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html,body,[data-testid="stApp"]{background:#0a0a0a!important;font-family:'Inter',sans-serif!important;color:#fff!important}
[data-testid="stSidebar"]{background:#050505!important;border-right:1px solid #181818!important}
[data-testid="stSidebar"] *{font-family:'Inter',sans-serif!important}
.block-container{padding:1.5rem 2rem 3rem!important;max-width:1380px!important}
#MainMenu,footer,header,.stDeployButton{visibility:hidden!important;height:0!important;display:none!important}
.mc{background:#111;border:1px solid #1e1e1e;border-radius:14px;padding:18px 20px;position:relative;overflow:hidden;margin-bottom:8px}
.mc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.ml{color:#3a3a3a;font-size:9.5px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;margin-bottom:5px}
.mv{font-size:2.15rem;font-weight:800;line-height:1}
.ms{color:#3a3a3a;font-size:11px;margin-top:4px}
.ic{background:#0d0d0d;border:1px solid #1c1c1c;border-radius:0 12px 12px 0;padding:14px 18px;font-size:12.5px;line-height:1.75;color:#666;margin:12px 0}
.ic strong{color:#ccc}
.ict{font-size:9px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#2e2e2e;margin-bottom:7px}
.sh{font-size:9.5px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#2e2e2e;margin:22px 0 10px}
.tag{display:inline-block;padding:3px 10px;border-radius:20px;font-size:9px;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
div[data-baseweb="select"]>div{background:#111!important;border-color:#222!important}
.stSlider>div>div>div{background:#1e1e1e!important}
</style>""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    if not os.path.exists('biometria_performance_startup.csv'): return None
    df = pd.read_csv('biometria_performance_startup.csv')
    if 'Dia' not in df.columns: df['Dia'] = range(1, len(df) + 1)
    return df

df = load()
if df is None:
    st.error("CSV not found. Run: python developmentday2.py"); st.stop()

RC = 'Readiness_Score' if 'Readiness_Score' in df.columns else 'Readiness'

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-size:10px;font-weight:700;color:#2a2a2a;letter-spacing:.14em;'
                'text-transform:uppercase;margin:8px 0 2px">BME PERFORMANCE</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:1.3rem;font-weight:900;color:#fff;margin:0 0 16px">AI Dashboard</p>',
                unsafe_allow_html=True)
    if 'Atleta_ID' in df.columns:
        ath = st.selectbox("Athlete", df['Atleta_ID'].unique(), label_visibility="collapsed")
        df = df[df['Atleta_ID'] == ath].reset_index(drop=True)
    dm = int(df['Dia'].max())
    dia = st.slider("Day", 1, dm, dm, label_visibility="collapsed")
    st.markdown('<div style="height:1px;background:#181818;margin:12px 0"></div>', unsafe_allow_html=True)
    page = st.radio("Nav", ["Today", "Physical", "Recovery", "Risk", "History"],
                    label_visibility="collapsed")
    st.markdown('<div style="height:1px;background:#181818;margin:12px 0"></div>', unsafe_allow_html=True)
    r0  = df[df['Dia'] == dia].iloc[0] if dia in df['Dia'].values else df.iloc[-1]
    rv  = float(r0.get(RC, 50)); av = float(r0.get('ACWR', 1.0))
    tier0 = str(r0.get('Readiness_Tier', ''))
    st.markdown(
        f'<div style="margin-bottom:14px">'
        f'<p style="font-size:9px;color:#2a2a2a;letter-spacing:.12em;text-transform:uppercase;margin:0 0 3px">Readiness</p>'
        f'<p style="font-size:1.35rem;font-weight:900;color:{cg(rv)};margin:0">{rv:.0f}'
        f'<span style="font-size:.8rem;font-weight:400;color:#333">%</span></p>'
        f'<p style="font-size:9.5px;color:#333;margin:0">{tier0}</p></div>'
        f'<div><p style="font-size:9px;color:#2a2a2a;letter-spacing:.12em;text-transform:uppercase;margin:0 0 3px">ACWR</p>'
        f'<p style="font-size:1.35rem;font-weight:900;color:{ca(av)};margin:0">{av:.2f}</p>'
        f'<p style="font-size:9.5px;color:#333;margin:0">{"Danger zone" if av>1.5 else "Optimal" if av>=0.8 else "Sub-training"}</p></div>',
        unsafe_allow_html=True)

# ── Row context ───────────────────────────────────────────────────────────────
row = df[df['Dia'] == dia].iloc[0] if dia in df['Dia'].values else df.iloc[-1]
dp  = df[df['Dia'] <= dia]

def g_(c, d=0.0):
    try: v = row.get(c, d); return d if pd.isna(v) else float(v)
    except: return d

def gs(c, d="—"):
    try: v = row.get(c, d); return d if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
    except: return d

def mc(lbl, val, sub="", c=BL):
    return (f'<div class="mc"><div style="position:absolute;top:0;left:0;right:0;height:2px;background:{c}"></div>'
            f'<div class="ml">{lbl}</div><div class="mv" style="color:{c}">{val}</div>'
            f'<div class="ms">{sub}</div></div>')

def ic(text, c=BL, title="Coach Insight"):
    return (f'<div class="ic" style="border-left:3px solid {c}">'
            f'<div class="ict">{title}</div>{text}</div>')

def ring(val, mx, col, lbl, sz=215):
    p = min(max(val / mx, 0.001), 0.999)
    fig = go.Figure(go.Pie(
        values=[p, 1 - p], hole=0.72,
        marker=dict(colors=[col, "#1c1c1c"], line=dict(width=0)),
        sort=False, direction="clockwise", rotation=90,
        textinfo="none", hoverinfo="none", showlegend=False))
    fig.add_annotation(text=f"<b>{val:.0f}</b>", x=0.5, y=0.57, showarrow=False,
                       font=dict(size=34, color=col, family="Inter"))
    fig.add_annotation(text=lbl.upper(), x=0.5, y=0.34, showarrow=False,
                       font=dict(size=9, color="#3a3a3a", family="Inter"))
    fig.update_layout(paper_bgcolor=BG, margin=dict(l=0, r=0, t=0, b=0), height=sz)
    return fig

def sf(fig, h=270):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter", color=T2, size=11),
        margin=dict(l=4, r=4, t=8, b=4), height=h,
        xaxis=dict(gridcolor="#141414", zeroline=False, showline=False,
                   tickfont=dict(size=9.5, color=T3)),
        yaxis=dict(gridcolor="#141414", zeroline=False, showline=False,
                   tickfont=dict(size=9.5, color=T3)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=T2),
                    orientation="h", y=1.08, x=0))
    return fig

PC = {"displayModeBar": False}

def hdr(title, sub=""):
    st.markdown(f'<p style="font-size:1.45rem;font-weight:900;color:#fff;margin:0 0 2px">{title}</p>'
                f'<p style="font-size:11px;color:#3a3a3a;margin:0 0 18px">{sub}</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
if page == "Today":
    readiness = g_(RC, 50);       recovery = g_('Recovery_Score', 50)
    strain    = g_('Strain_Score', 0); hrv = g_('HRV_ms', 70)
    hrv_b     = g_('HRV_Media_Base', hrv); debt = g_('Sleep_Debt_Acumulado_h', 0)
    acwr      = g_('ACWR', 1.0);  assim = g_('Assimetria_Percent', 0)
    sono      = g_('Horas_Sono', 7); sonop = g_('Sono_Profundo_h', 1.5)
    tipo      = gs('Tipo_Sessao'); tier = gs('Readiness_Tier')
    hrv_tr    = gs('HRV_Trend', 'Stable')

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">'
        f'<span style="font-size:1.5rem;font-weight:900">Day {dia}</span>'
        f'<span class="tag" style="background:{cg(readiness)}1a;color:{cg(readiness)}">{tier}</span>'
        f'<span class="tag" style="background:#181818;color:#444">{tipo}</span></div>'
        f'<p style="font-size:11px;color:#333;margin:0 0 20px">Performance Report · Athlete {gs("Atleta_ID","A001")}</p>',
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([2.5, 2, 2, 2])
    with c1:
        st.plotly_chart(ring(readiness, 100, cg(readiness), "Readiness", 230),
                        use_container_width=True, config=PC)
    with c2:
        st.plotly_chart(ring(recovery, 100, cg(recovery), "Recovery", 215),
                        use_container_width=True, config=PC)
    with c3:
        st.markdown(mc("HRV", f"{hrv:.0f} ms", f"Baseline {hrv_b:.0f} · {hrv_tr}",
                       G if hrv >= hrv_b else R), unsafe_allow_html=True)
        st.markdown(mc("Sleep", f"{sono:.1f} h",
                       f"{sonop:.1f}h deep · {sonop/max(sono,.1)*100:.0f}% eff.", BL),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(mc("Strain", f"{strain:.1f}", "of 21 · cardiovascular load",
                       A if strain > 14 else BL), unsafe_allow_html=True)
        st.markdown(mc("ACWR", f"{acwr:.2f}", "safe 0.8–1.3 · " +
                       ("danger" if acwr > 1.5 else "optimal" if acwr >= 0.8 else "low"),
                       ca(acwr)), unsafe_allow_html=True)

    rtext = "excellent" if readiness >= 67 else ("moderate" if readiness >= 33 else "low")
    atext = ("⚠ high — reduce volume 15–20% today." if acwr > 1.5
             else "✓ optimal — full intensity." if 0.8 <= acwr <= 1.3
             else "↑ sub-training — increase load gradually.")
    dtext = (f"<strong>{debt:.1f}h</strong> accumulated sleep debt — prioritize rest."
             if debt > 2 else "Sleep debt under control.")
    htext = "positive adaptation ✓" if hrv >= hrv_b else "physiological stress ⚠"
    st.markdown(ic(
        f"Readiness is <strong>{rtext}</strong> ({readiness:.0f}/100). ACWR is {atext} "
        f"HRV <strong>{hrv:.0f} ms</strong> vs baseline <strong>{hrv_b:.0f} ms</strong> — {htext}. "
        f"{dtext} Asymmetry {assim:.1f}%"
        f"{' — above 10% critical threshold.' if assim > 10 else ' — within normal range.'}",
        cg(readiness)), unsafe_allow_html=True)

    st.markdown('<div class="sh">Readiness — 30-day trend</div>', unsafe_allow_html=True)
    if RC in dp.columns:
        fig = go.Figure()
        fig.add_hrect(y0=67, y1=105, fillcolor="rgba(0,230,118,.03)", line_width=0)
        fig.add_hrect(y0=0,  y1=33,  fillcolor="rgba(255,61,87,.03)",  line_width=0)
        fig.add_trace(go.Scatter(x=dp['Dia'], y=dp[RC], mode='lines',
                                 line=dict(color=BL, width=2.5),
                                 fill='tozeroy', fillcolor=FILL[BL], name="Readiness"))
        st.plotly_chart(sf(fig, 230), use_container_width=True, config=PC)

    d_ia = gs('Diagnostico_IA'); d_sono = gs('Diagnostico_Sono')
    c1, c2 = st.columns(2)
    with c1:
        bg = R if any(x in d_ia for x in ['PERIGO', 'CRÍTICO']) else A if 'AVISO' in d_ia else G
        st.markdown(f'<div style="background:{bg}12;border:1px solid {bg}2e;border-radius:10px;'
                    f'padding:11px 15px;font-size:12px;color:{bg};line-height:1.6">{d_ia}</div>',
                    unsafe_allow_html=True)
    with c2:
        bg = R if '🔴' in d_sono else A if '🟡' in d_sono else G
        st.markdown(f'<div style="background:{bg}12;border:1px solid {bg}2e;border-radius:10px;'
                    f'padding:11px 15px;font-size:12px;color:{bg};line-height:1.6">{d_sono}</div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Physical":
    hdr("Physical Performance", "External load — distance, speed, metabolic power, HMLD")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(mc("Distance",    f"{g_('Distancia_km'):.1f} km",      "session total",        BL), unsafe_allow_html=True)
    with c2: st.markdown(mc("Max Speed",   f"{g_('Velocidade_Max_kmh'):.1f} km/h","fastest sprint",      PR), unsafe_allow_html=True)
    with c3: st.markdown(mc("HMLD",        f"{g_('HMLD_m'):.0f} m",             "metres >25.5 W/kg",    A),  unsafe_allow_html=True)
    with c4: st.markdown(mc("High Int.",   f"{g_('Minutos_High_Intensity'):.0f} min","above threshold",  G),  unsafe_allow_html=True)

    st.markdown('<div class="sh">Distance & HMLD · Max Speed</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        if 'Distancia_km' in dp.columns:
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Distancia_km'], name="Distance (km)",
                                     line=dict(color=BL, width=2.5), fill='tozeroy', fillcolor=FILL[BL]))
        if 'HMLD_m' in dp.columns:
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['HMLD_m'] / 1000, name="HMLD (km)",
                                     line=dict(color=A, width=2, dash='dot'), fill='none'))
        st.plotly_chart(sf(fig), use_container_width=True, config=PC)
    with c2:
        if 'Velocidade_Max_kmh' in dp.columns:
            clrs = [G if v > 34 else BL for v in dp['Velocidade_Max_kmh']]
            fig = go.Figure(go.Bar(x=dp['Dia'], y=dp['Velocidade_Max_kmh'],
                                   marker_color=clrs, name="Max Speed"))
            fig.add_hline(y=34, line_dash="dash", line_color=G, line_width=1)
            st.plotly_chart(sf(fig), use_container_width=True, config=PC)

    if 'Metabolic_Power_Mean_Wkg' in dp.columns:
        st.markdown('<div class="sh">Metabolic Power · Mechanical Load</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Metabolic_Power_Mean_Wkg'], name="Mean",
                                     line=dict(color=BL, width=2), fill='tozeroy', fillcolor=FILL[BL]))
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Metabolic_Power_Peak_Wkg'], name="Peak",
                                     line=dict(color=A, width=2), fill='none'))
            fig.add_hline(y=25.5, line_dash="dot", line_color=R, line_width=1)
            st.plotly_chart(sf(fig), use_container_width=True, config=PC)
        with c2:
            fig = go.Figure()
            if 'Carga_Mecanica_G_Raw' in dp.columns:
                fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Carga_Mecanica_G_Raw'], name="Raw",
                                         line=dict(color=T3, width=1, dash='dot'), opacity=0.45))
            if 'Carga_Mecanica_G' in dp.columns:
                fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Carga_Mecanica_G'], name="Smoothed",
                                         line=dict(color=A, width=2.5), fill='tozeroy', fillcolor=FILL[A]))
            st.plotly_chart(sf(fig), use_container_width=True, config=PC)

    st.markdown(ic(
        f"HMLD today: <strong>{g_('HMLD_m'):.0f} m</strong> — metres above 25.5 W/kg, where real metabolic effort begins. "
        f"Values above 300m indicate high-demand session. "
        f"Asymmetry: <strong>{g_('Assimetria_Percent'):.1f}%</strong>"
        f"{'  ⚠ Above 10% critical threshold — balance work recommended.' if g_('Assimetria_Percent') > 10 else ' — within normal range.'}",
        BL), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Recovery":
    hrv   = g_('HRV_ms', 70);       hrv_b = g_('HRV_Media_Base', hrv)
    rec   = g_('Recovery_Score', 50); sono = g_('Horas_Sono', 7)
    sonop = g_('Sono_Profundo_h', 1.5); debt = g_('Sleep_Debt_Acumulado_h', 0)
    hrv_tr = gs('HRV_Trend', 'Stable')

    hdr("Recovery & Internal State", "HRV, sleep, sleep debt, recovery score")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(mc("HRV",        f"{hrv:.0f} ms",  f"Baseline {hrv_b:.0f} · {hrv_tr}", G if hrv >= hrv_b else R), unsafe_allow_html=True)
    with c2: st.markdown(mc("Recovery",   f"{rec:.0f}%",    "3-pillar: HRV+Sleep+Load",          cg(rec)), unsafe_allow_html=True)
    with c3: st.markdown(mc("Sleep",      f"{sono:.1f} h",  f"{sonop:.1f}h deep · {sonop/max(sono,.1)*100:.0f}% eff.", BL), unsafe_allow_html=True)
    with c4: st.markdown(mc("Sleep Debt", f"{debt:+.1f} h", "7-day accumulated", R if debt > 3 else A if debt > 1 else G), unsafe_allow_html=True)

    st.markdown('<div class="sh">HRV vs Baseline · Sleep Quality</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        if 'HRV_Media_Base' in dp.columns:
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['HRV_Media_Base'], name="Baseline 30d",
                                     line=dict(color=A, width=1.5, dash='dash'), fill='none'))
        if 'HRV_ms' in dp.columns:
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['HRV_ms'], name="HRV",
                                     line=dict(color=G, width=2.5), fill='tozeroy', fillcolor=FILL[G]))
        st.plotly_chart(sf(fig), use_container_width=True, config=PC)
    with c2:
        if 'Horas_Sono' in dp.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dp['Dia'], y=dp['Horas_Sono'], name="Total",
                                 marker_color=BL, opacity=0.45))
            if 'Sono_Profundo_h' in dp.columns:
                fig.add_trace(go.Bar(x=dp['Dia'], y=dp['Sono_Profundo_h'], name="Deep",
                                     marker_color=PR))
            fig.add_hline(y=8, line_dash="dot", line_color=G, line_width=1)
            fig.update_layout(barmode='overlay')
            st.plotly_chart(sf(fig), use_container_width=True, config=PC)

    st.markdown('<div class="sh">Recovery Score · Sleep Debt</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if 'Recovery_Score' in dp.columns:
            clrs = [cg(v) for v in dp['Recovery_Score']]
            fig = go.Figure(go.Bar(x=dp['Dia'], y=dp['Recovery_Score'], marker_color=clrs))
            fig.add_hrect(y0=67, y1=105, fillcolor="rgba(0,230,118,.03)", line_width=0)
            fig.add_hrect(y0=0, y1=33, fillcolor="rgba(255,61,87,.03)", line_width=0)
            st.plotly_chart(sf(fig, 250), use_container_width=True, config=PC)
    with c2:
        if 'Sleep_Debt_Acumulado_h' in dp.columns:
            clrs = [R if v > 3 else A if v > 0 else G for v in dp['Sleep_Debt_Acumulado_h']]
            fig = go.Figure(go.Bar(x=dp['Dia'], y=dp['Sleep_Debt_Acumulado_h'], marker_color=clrs))
            fig.add_hline(y=3, line_dash="dot", line_color=R, line_width=1)
            st.plotly_chart(sf(fig, 250), use_container_width=True, config=PC)

    st.markdown(ic(
        f"Recovery score <strong>{rec:.0f}%</strong> — SNA×50% + Sleep×30% + Load×20%. "
        f"HRV <strong>{hrv:.0f} ms</strong> vs baseline <strong>{hrv_b:.0f} ms</strong> — "
        f"{'positive adaptation ✓' if hrv >= hrv_b else 'physiological stress ⚠'}. "
        f"Deep sleep: <strong>{sonop:.1f}h</strong> ({sonop/max(sono,.1)*100:.0f}% of total, target 20–25%). "
        f"{'Sleep debt elevated — prioritize rest.' if debt > 2 else 'Sleep debt under control.'}",
        cg(rec)), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Risk":
    acwr  = g_('ACWR', 1.0);     inj  = g_('Injury_Risk_Score', 0)
    zc    = g_('Z_Score_Carga', 0); fad = g_('Indice_Fadiga', 0)
    assim = g_('Assimetria_Percent', 0)

    hdr("Risk Management", "ACWR, injury risk, fatigue index, asymmetry")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(mc("ACWR",          f"{acwr:.2f}",   "safe zone 0.8–1.3",       ca(acwr)), unsafe_allow_html=True)
    with c2: st.markdown(mc("Injury Risk",   f"{inj:.0f}",    "composite score /100",    ci(inj)),  unsafe_allow_html=True)
    with c3: st.markdown(mc("Z-Score Load",  f"{zc:+.2f} σ", "deviation from baseline", R if abs(zc) > 2 else BL), unsafe_allow_html=True)
    with c4: st.markdown(mc("Fatigue Index", f"{fad:.0f}",    "accumulated impact /100", R if fad > 80 else A if fad > 50 else G), unsafe_allow_html=True)

    st.markdown('<div class="sh">ACWR with Risk Zones</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        if 'ACWR' in dp.columns:
            fig = go.Figure()
            fig.add_hrect(y0=0.8, y1=1.3, fillcolor="rgba(0,230,118,.05)",  line_width=0)
            fig.add_hrect(y0=1.3, y1=1.5, fillcolor="rgba(255,179,0,.05)",  line_width=0)
            fig.add_hrect(y0=1.5, y1=3.0, fillcolor="rgba(255,61,87,.05)",  line_width=0)
            clrs = [ca(v) for v in dp['ACWR']]
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['ACWR'], mode='lines+markers',
                                     marker=dict(color=clrs, size=5, line=dict(width=0)),
                                     line=dict(color=BL, width=2.5), name="ACWR"))
            if all(c in dp.columns for c in ['Carga_Aguda', 'Carga_Cronica']):
                fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Carga_Aguda'],
                                         name='Acute 7d', line=dict(color=A, width=1.5, dash='dot'),
                                         yaxis='y2'))
                fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Carga_Cronica'],
                                         name='Chronic 28d', line=dict(color=PR, width=1.5, dash='dot'),
                                         yaxis='y2'))
                fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False,
                                              tickfont=dict(size=9, color=T3)))
            st.plotly_chart(sf(fig, 310), use_container_width=True, config=PC)
    with c2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=inj,
            number={"font": {"size": 28, "color": ci(inj), "family": "Inter"}},
            gauge={"axis": {"range": [0, 100], "visible": False},
                   "bar": {"color": ci(inj), "thickness": 0.14},
                   "bgcolor": "#1a1a1a", "bordercolor": "rgba(0,0,0,0)",
                   "steps": [{"range": [0, 35],  "color": "rgba(0,230,118,.07)"},
                              {"range": [35, 60], "color": "rgba(255,179,0,.07)"},
                              {"range": [60, 100],"color": "rgba(255,61,87,.07)"}],
                   "threshold": {"line": {"color": R, "width": 2},
                                 "thickness": 0.75, "value": 60}}))
        fig.add_annotation(text="Injury Risk", x=0.5, y=-0.05, showarrow=False,
                           font=dict(size=9.5, color=T3, family="Inter"))
        fig.update_layout(paper_bgcolor=BG, font=dict(family="Inter", color=T2),
                          height=295, margin=dict(l=12, r=12, t=28, b=16))
        st.plotly_chart(fig, use_container_width=True, config=PC)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sh">Injury Risk History</div>', unsafe_allow_html=True)
        if 'Injury_Risk_Score' in dp.columns:
            clrs = [ci(v) for v in dp['Injury_Risk_Score']]
            fig = go.Figure(go.Bar(x=dp['Dia'], y=dp['Injury_Risk_Score'], marker_color=clrs))
            fig.add_hline(y=60, line_dash="dot", line_color=R, line_width=1)
            st.plotly_chart(sf(fig, 250), use_container_width=True, config=PC)
    with c2:
        st.markdown('<div class="sh">Muscular Asymmetry</div>', unsafe_allow_html=True)
        if 'Assimetria_Percent' in dp.columns:
            clrs = [R if v > 10 else A if v > 7 else G for v in dp['Assimetria_Percent']]
            fig = go.Figure(go.Bar(x=dp['Dia'], y=dp['Assimetria_Percent'], marker_color=clrs))
            fig.add_hline(y=10, line_dash="dot", line_color=R, line_width=1)
            fig.add_hline(y=7,  line_dash="dot", line_color=A, line_width=1)
            st.plotly_chart(sf(fig, 250), use_container_width=True, config=PC)

    st.markdown(ic(
        f"ACWR <strong>{acwr:.2f}</strong> — "
        f"{'⚠ Danger zone. Reduce volume immediately.' if acwr > 1.5 else '✓ Optimal zone. Full intensity.' if 0.8 <= acwr <= 1.3 else '↑ Sub-training. Increase gradually.'} "
        f"Injury risk: <strong>{inj:.0f}/100</strong> — ACWR (35%) + Asymmetry (25%) + Fatigue (25%) + Temp (15%). "
        f"{'🔴 Rest or active recovery recommended.' if inj > 60 else '🟡 Monitor closely.' if inj > 35 else '🟢 Low risk — good to train.'}",
        ci(inj)), unsafe_allow_html=True)

    if 'Tipo_Sessao' in dp.columns:
        st.markdown('<div class="sh">Session Type Distribution</div>', unsafe_allow_html=True)
        cnt = dp['Tipo_Sessao'].value_counts().reset_index(); cnt.columns = ['Type', 'Days']
        cmap = {'Alta Performance': G, 'Desenvolvimento': BL, 'Base': PR,
                'Recuperação': A, '⚠️ Sobrecarga': R}
        fig = px.bar(cnt, x='Type', y='Days', color='Type', color_discrete_map=cmap, text='Days')
        fig.update_layout(showlegend=False)
        st.plotly_chart(sf(fig, 240), use_container_width=True, config=PC)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "History":
    ar   = dp[RC].mean()                    if RC in dp.columns else 0
    ah   = dp['HRV_ms'].mean()              if 'HRV_ms' in dp.columns else 0
    ad   = dp['Distancia_km'].mean()        if 'Distancia_km' in dp.columns else 0
    mv   = dp['Velocidade_Max_kmh'].max()   if 'Velocidade_Max_kmh' in dp.columns else 0
    arec = dp['Recovery_Score'].mean()      if 'Recovery_Score' in dp.columns else 0

    hdr("Full History", f"All indicators · {len(dp)} days analysed")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(mc("Avg Readiness", f"{ar:.0f}%",   f"{len(dp)} days",    cg(ar)),   unsafe_allow_html=True)
    with c2: st.markdown(mc("Avg Recovery",  f"{arec:.0f}%", "3-pillar index",     cg(arec)), unsafe_allow_html=True)
    with c3: st.markdown(mc("Avg HRV",       f"{ah:.0f} ms", "cardiac variability",G),         unsafe_allow_html=True)
    with c4: st.markdown(mc("Avg Distance",  f"{ad:.1f} km", "per session",        BL),        unsafe_allow_html=True)
    with c5: st.markdown(mc("Top Sprint",    f"{mv:.1f} km/h","period record",     PR),        unsafe_allow_html=True)

    st.markdown('<div class="sh">Multi-metric overlay</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for col, name, color in [(RC, 'Readiness', BL), ('Recovery_Score', 'Recovery', G),
                              ('Injury_Risk_Score', 'Injury Risk', R)]:
        if col in dp.columns:
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp[col], name=name,
                                     line=dict(color=color, width=2), fill='none'))
    if 'Strain_Score' in dp.columns:
        fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Strain_Score'] * (100 / 21),
                                  name="Strain (norm.)", line=dict(color=A, width=1.5, dash='dot'),
                                  fill='none'))
    fig.update_layout(yaxis_range=[0, 105])
    st.plotly_chart(sf(fig, 320), use_container_width=True, config=PC)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sh">Acute vs Chronic Load</div>', unsafe_allow_html=True)
        if all(c in dp.columns for c in ['Carga_Aguda', 'Carga_Cronica']):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Carga_Aguda'], name="Acute 7d",
                                     line=dict(color=A, width=2), fill='tozeroy', fillcolor=FILL[A]))
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Carga_Cronica'], name="Chronic 28d",
                                     line=dict(color=BL, width=2), fill='none'))
            st.plotly_chart(sf(fig, 260), use_container_width=True, config=PC)
    with c2:
        st.markdown('<div class="sh">Endurance (7-day rolling avg)</div>', unsafe_allow_html=True)
        if 'Resistencia' in dp.columns:
            roll = dp['Resistencia'].rolling(7, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Resistencia'], name="Daily",
                                     line=dict(color=T3, width=1.2), fill='none', opacity=0.5))
            fig.add_trace(go.Scatter(x=dp['Dia'], y=roll, name="7d avg",
                                     line=dict(color=G, width=2.5), fill='tozeroy', fillcolor=FILL[G]))
            st.plotly_chart(sf(fig, 260), use_container_width=True, config=PC)

    with st.expander("Full data table"):
        show = [c for c in ['Dia', RC, 'Recovery_Score', 'Strain_Score', 'Injury_Risk_Score',
                             'ACWR', 'HRV_ms', 'Horas_Sono', 'Distancia_km', 'Velocidade_Max_kmh',
                             'Tipo_Sessao', 'HRV_Trend', 'Readiness_Tier'] if c in dp.columns]
        num  = [c for c in show if c not in ['Tipo_Sessao', 'HRV_Trend', 'Readiness_Tier', 'Dia']]
        st.dataframe(dp[show].set_index('Dia').style.format({c: "{:.1f}" for c in num}, na_rep="—"),
                     use_container_width=True)
