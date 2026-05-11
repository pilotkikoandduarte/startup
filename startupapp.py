import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="BME Performance AI", page_icon="🏆",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.kpi-card{background:#161b27;border-radius:14px;padding:18px 20px;border-left:4px solid #2979ff;margin:6px 0}
.kpi-label{color:#8b9ab1;font-size:.78rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase}
.kpi-value{color:#e8edf5;font-size:1.9rem;font-weight:800;line-height:1.15}
.kpi-sub{color:#8b9ab1;font-size:.75rem;margin-top:3px}
.coach-card{background:#0d1117;border:1px solid #21262d;border-radius:14px;padding:18px 22px;margin:14px 0}
.coach-title{color:#2979ff;font-size:.8rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin-bottom:8px}
.coach-text{color:#c9d1d9;font-size:.93rem;line-height:1.65}
div[data-testid="stSidebar"]{background:#0a0e1a}
</style>""", unsafe_allow_html=True)

G, Y, R = "#00c853", "#ffd600", "#ff1744"
BL, OR, PR = "#2979ff", "#ff6d00", "#9c27b0"
BG = "#0d1117"
GR = "#21262d"


def _c(acwr):
    return R if acwr > 1.5 else OR if acwr > 1.3 else G if acwr >= 0.8 else Y


def _rc(v):
    return G if v >= 67 else Y if v >= 33 else R


def _ic(v):
    return R if v >= 60 else OR if v >= 35 else G


def _style(fig, h=300):
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color="#c9d1d9",
                      margin=dict(l=6, r=6, t=36, b=6), height=h,
                      xaxis=dict(gridcolor=GR, zeroline=False, showline=False),
                      yaxis=dict(gridcolor=GR, zeroline=False, showline=False),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11))
    return fig


def kpi(label, value, sub="", color=BL):
    st.markdown(f'<div class="kpi-card" style="border-color:{color}">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value" style="color:{color}">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)


def coach(title, text):
    st.markdown(f'<div class="coach-card"><div class="coach-title">{title}</div>'
                f'<div class="coach-text">{text}</div></div>', unsafe_allow_html=True)


def gauge(val, max_val, title, color, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={"suffix": suffix, "font": {"size": 26, "color": color}},
        title={"text": title, "font": {"size": 12, "color": "#8b9ab1"}},
        gauge={"axis": {"range": [0, max_val], "tickcolor": "#8b9ab1"},
               "bar": {"color": color}, "bgcolor": GR,
               "bordercolor": "rgba(0,0,0,0)", "steps": []}))
    fig.update_layout(paper_bgcolor=BG, font_color="#c9d1d9",
                      height=195, margin=dict(l=12, r=12, t=28, b=6))
    return fig


@st.cache_data
def load_data():
    path = 'biometria_performance_startup.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'Dia' not in df.columns:
        df['Dia'] = range(1, len(df) + 1)
    return df


df = load_data()
if df is None:
    st.error("CSV não encontrado. Executa primeiro: `python developmentday2.py`")
    st.stop()

READ_COL = 'Readiness_Score' if 'Readiness_Score' in df.columns else 'Readiness'

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏆 BME Performance AI")
    st.markdown("---")
    if 'Atleta_ID' in df.columns:
        sel_a = st.selectbox("Atleta", df['Atleta_ID'].unique())
        df = df[df['Atleta_ID'] == sel_a].reset_index(drop=True)
    day_max = int(df['Dia'].max())
    dia = st.slider("📅 Dia de análise", 1, day_max, day_max)
    st.markdown("---")
    page = st.radio("Secção", ["📊 Relatório de Hoje", "🏃 Performance Física",
                               "🧬 Recuperação & HRV", "🚨 Gestão de Risco",
                               "📈 Visão Histórica"])
    st.markdown("---")
    row0 = df[df['Dia'] == dia].iloc[0] if dia in df['Dia'].values else df.iloc[-1]
    acwr0 = float(row0.get('ACWR', 1.0))
    if acwr0 > 1.5:
        st.error(f"⚠️ ACWR {acwr0:.2f} — ZONA PERIGO")
    elif acwr0 >= 0.8:
        st.success(f"✅ ACWR {acwr0:.2f} — Zona óptima")
    else:
        st.warning(f"🟡 ACWR {acwr0:.2f} — Sub-treino")
    if READ_COL in df.columns:
        r0 = float(row0.get(READ_COL, 50))
        st.caption(f"Readiness: **{r0:.0f}%** — {str(row0.get('Readiness_Tier','—'))}")
    st.caption(f"Tipo de sessão: **{str(row0.get('Tipo_Sessao','—'))}**")

row = df[df['Dia'] == dia].iloc[0] if dia in df['Dia'].values else df.iloc[-1]
dp = df[df['Dia'] <= dia]


def g_(col, default=0.0):
    try:
        v = row.get(col, default)
        return default if pd.isna(v) else float(v)
    except Exception:
        return default


def gs(col, default="—"):
    try:
        v = row.get(col, default)
        return default if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
    except Exception:
        return default


# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Relatório de Hoje":
    readiness = g_(READ_COL, 50)
    recovery  = g_('Recovery_Score', 50)
    strain    = g_('Strain_Score', 0)
    injury    = g_('Injury_Risk_Score', 0)
    tipo      = gs('Tipo_Sessao')
    tier      = gs('Readiness_Tier')
    hrv       = g_('HRV_ms', 70)
    hrv_base  = g_('HRV_Media_Base', hrv)
    hrv_trend = gs('HRV_Trend', '➡️ Estável')
    debt      = g_('Sleep_Debt_Acumulado_h', 0)
    acwr      = g_('ACWR', 1.0)
    assim     = g_('Assimetria_Percent', 0)

    st.markdown(f"## Relatório · Dia {dia} &nbsp; {tier}")
    st.caption(f"Atleta {gs('Atleta_ID','A001')} · análise gerada automaticamente")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Readiness", f"{readiness:.0f}%", "prontidão geral", _rc(readiness))
    with c2: kpi("Recovery", f"{recovery:.0f}%", "recuperação interna", _rc(recovery))
    with c3: kpi("Strain Score", f"{strain:.1f}/21", "carga cardiovascular", OR if strain > 14 else BL)
    with c4: kpi("Injury Risk", f"{injury:.0f}/100", "risco composto", _ic(injury))
    with c5: kpi("Tipo de Sessão", tipo, "classificação do dia", BL)

    st.markdown("---")

    r_txt = "excelente" if readiness >= 67 else ("moderada" if readiness >= 33 else "baixa")
    acwr_txt = ("⚠️ elevada — estás a acumular fadiga. Reduz volume 15-20% hoje." if acwr > 1.5
                else "✅ no nível óptimo — podes treinar com intensidade máxima." if 0.8 <= acwr <= 1.3
                else "🟡 baixa — bom momento para aumentar carga gradualmente.")
    debt_txt = (f"Tens <strong>{debt:.1f}h</strong> de dívida de sono acumulada — prioriza descanso esta noite."
                if debt > 2 else "O teu sono está controlado.")
    assim_txt = (f"⚠️ Assimetria de <strong>{assim:.1f}%</strong> — acima do limiar crítico de 10%. Trabalho de equilíbrio recomendado."
                 if assim > 10 else f"Assimetria dentro do normal ({assim:.1f}%).")
    hrv_st = "adaptação positiva ✅" if hrv >= hrv_base else "stress fisiológico ⚠️"

    coach("📋 Análise do Treinador",
          f"A tua prontidão hoje é <strong>{r_txt}</strong> ({readiness:.0f}/100). "
          f"A carga de treino está {acwr_txt} "
          f"O HRV ({hrv:.0f} ms vs baseline {hrv_base:.0f} ms) indica <strong>{hrv_st}</strong>, "
          f"com tendência <strong>{hrv_trend}</strong>. {debt_txt} {assim_txt}")

    c1, c2, c3 = st.columns(3)
    with c1: st.plotly_chart(gauge(readiness, 100, "Readiness Score", _rc(readiness), "%"),
                              use_container_width=True, config={"displayModeBar": False})
    with c2: st.plotly_chart(gauge(recovery, 100, "Recovery Score", _rc(recovery), "%"),
                              use_container_width=True, config={"displayModeBar": False})
    with c3: st.plotly_chart(gauge(strain, 21, "Strain Score", OR if strain > 14 else BL, "/21"),
                              use_container_width=True, config={"displayModeBar": False})

    st.markdown("#### Evolução da Prontidão")
    if READ_COL in dp.columns:
        fig = go.Figure()
        fig.add_hrect(y0=67, y1=105, fillcolor=f"{G}14", line_width=0,
                      annotation_text="Zona óptima", annotation_position="top right")
        fig.add_hrect(y0=0, y1=33, fillcolor=f"{R}14", line_width=0,
                      annotation_text="Zona crítica", annotation_position="bottom right")
        fig.add_trace(go.Scatter(x=dp['Dia'], y=dp[READ_COL], name="Readiness",
                                 line=dict(color=BL, width=2.5),
                                 fill='tozeroy', fillcolor='rgba(41,121,255,0.07)'))
        st.plotly_chart(_style(fig, 260), use_container_width=True, config={"displayModeBar": False})

    d_ia = gs('Diagnostico_IA'); d_sono = gs('Diagnostico_Sono')
    c1, c2 = st.columns(2)
    with c1:
        (st.error if any(x in d_ia for x in ['PERIGO', 'CRÍTICO']) else
         st.warning if 'AVISO' in d_ia else st.success)(f"🩺 {d_ia}")
    with c2:
        (st.error if '🔴' in d_sono else st.warning if '🟡' in d_sono else st.success)(f"🌙 {d_sono}")

# ════════════════════════════════════════════════════════════════════════════
elif page == "🏃 Performance Física":
    st.markdown("## Performance Física")
    st.caption("Carga externa — distância, velocidade, potência metabólica, HMLD")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Distância", f"{g_('Distancia_km'):.1f} km", "percorrida na sessão", BL)
    with c2: kpi("Vel. Máxima", f"{g_('Velocidade_Max_kmh'):.1f} km/h", "sprint mais rápido", PR)
    with c3: kpi("HMLD", f"{g_('HMLD_m'):.0f} m", "distância >25.5 W/kg", OR)
    with c4: kpi("Alta Intensidade", f"{g_('Minutos_High_Intensity'):.0f} min", "tempo acima do limiar", G)

    coach("📖 O que é o HMLD?",
          "O <strong>High Metabolic Load Distance</strong> (HMLD) mede os metros percorridos "
          "acima de 25,5 W/kg — o limiar onde o corpo entra em esforço metabólico real. "
          "É o indicador StatSports de exigência absoluta de sessão. Valores acima de 300m "
          "indicam sessão de alta exigência. O teu HMLD hoje: "
          f"<strong>{g_('HMLD_m'):.0f} m</strong>.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Distância (km) e HMLD")
        fig = go.Figure()
        if 'Distancia_km' in dp.columns:
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Distancia_km'], name='Distância (km)',
                                     line=dict(color=BL, width=2.5),
                                     fill='tozeroy', fillcolor='rgba(41,121,255,0.07)'))
        if 'HMLD_m' in dp.columns:
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['HMLD_m'] / 1000, name='HMLD (km)',
                                     line=dict(color=OR, width=2, dash='dot')))
        st.plotly_chart(_style(fig), use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown("#### Velocidade Máxima (km/h)")
        if 'Velocidade_Max_kmh' in dp.columns:
            clrs = [G if v > 34 else BL for v in dp['Velocidade_Max_kmh']]
            fig = go.Figure(go.Bar(x=dp['Dia'], y=dp['Velocidade_Max_kmh'],
                                   marker_color=clrs, name='Vel. Máx.'))
            fig.add_hline(y=34.0, line_dash="dash", line_color=G,
                          annotation_text="🥇 Limiar de recorde")
            st.plotly_chart(_style(fig), use_container_width=True, config={"displayModeBar": False})

    if 'Metabolic_Power_Mean_Wkg' in dp.columns:
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Potência Metabólica (W/kg)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Metabolic_Power_Mean_Wkg'],
                                     name='Média', line=dict(color=BL, width=2),
                                     fill='tozeroy', fillcolor='rgba(41,121,255,0.07)'))
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Metabolic_Power_Peak_Wkg'],
                                     name='Pico', line=dict(color=OR, width=2)))
            fig.add_hline(y=25.5, line_dash="dot", line_color=R, annotation_text="Limiar HMLD")
            st.plotly_chart(_style(fig), use_container_width=True, config={"displayModeBar": False})
        with c4:
            st.markdown("#### Carga Mecânica (G) — suavizada")
            fig = go.Figure()
            if 'Carga_Mecanica_G_Raw' in dp.columns:
                fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Carga_Mecanica_G_Raw'],
                                         name='Raw', line=dict(color="#8b9ab1", width=1, dash='dot'),
                                         opacity=0.45))
            if 'Carga_Mecanica_G' in dp.columns:
                fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Carga_Mecanica_G'],
                                         name='Suavizada', line=dict(color=OR, width=2.5),
                                         fill='tozeroy', fillcolor='rgba(255,109,0,0.07)'))
            st.plotly_chart(_style(fig), use_container_width=True, config={"displayModeBar": False})

    coach("📖 Assimetria Muscular",
          f"A assimetria hoje é <strong>{g_('Assimetria_Percent'):.1f}%</strong>. "
          "Valores acima de 10% indicam desequilíbrio significativo entre membros — fator de risco de lesão. "
          "O alvo clínico é manter abaixo de 8%. Monitorizar regularmente com testes isocinéticos.")

# ════════════════════════════════════════════════════════════════════════════
elif page == "🧬 Recuperação & HRV":
    st.markdown("## Recuperação & Estado Interno")
    st.caption("HRV, sono, dívida de sono, recovery score — carga interna")

    hrv      = g_('HRV_ms', 70)
    hrv_base = g_('HRV_Media_Base', hrv)
    recovery = g_('Recovery_Score', 50)
    sono_tot = g_('Horas_Sono', 7)
    sono_pr  = g_('Sono_Profundo_h', 1.5)
    debt     = g_('Sleep_Debt_Acumulado_h', 0)
    hrv_tr   = gs('HRV_Trend', '➡️ Estável')

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("HRV", f"{hrv:.0f} ms", f"baseline 30d: {hrv_base:.0f} ms", G if hrv >= hrv_base else R)
    with c2: kpi("Recovery Score", f"{recovery:.0f}%", "índice 3-pillar (HRV+Sono+Carga)", _rc(recovery))
    with c3: kpi("Sono", f"{sono_tot:.1f}h", f"{sono_pr:.1f}h profundo · {sono_pr/sono_tot*100:.0f}% efic.", BL)
    with c4: kpi("Dívida Sono (7d)", f"{debt:+.1f}h", "acumulada na semana", R if debt > 3 else Y if debt > 1 else G)

    coach("📖 Como é calculado o Recovery Score",
          "O Recovery Score combina <strong>SNA</strong> (HRV + SpO₂) × 50% + "
          "<strong>Sono</strong> (qualidade + quantidade) × 30% + "
          "<strong>Carga</strong> (ACWR) × 20%. "
          f"O teu HRV de <strong>{hrv:.0f} ms</strong> está "
          f"{'<strong style=\"color:{G}\">acima</strong>' if hrv >= hrv_base else '<strong style=\"color:{R}\">abaixo</strong>'} "
          f"da tua baseline pessoal de {hrv_base:.0f} ms — "
          f"indicador de {'adaptação positiva ✅' if hrv >= hrv_base else 'stress fisiológico — prioriza descanso ⚠️'}.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### HRV (ms) — Tendência: {hrv_tr}")
        fig = go.Figure()
        if 'HRV_Media_Base' in dp.columns:
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['HRV_Media_Base'], name='Baseline 30d',
                                     line=dict(color=Y, width=1.5, dash='dash'), opacity=0.8))
        if 'HRV_ms' in dp.columns:
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['HRV_ms'], name='HRV diário',
                                     line=dict(color=G, width=2.5),
                                     fill='tozeroy', fillcolor='rgba(0,200,83,0.06)'))
        st.plotly_chart(_style(fig, 300), use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown("#### Sono: Total vs Profundo (h)")
        if 'Horas_Sono' in dp.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dp['Dia'], y=dp['Horas_Sono'], name='Total', marker_color=BL, opacity=0.55))
            if 'Sono_Profundo_h' in dp.columns:
                fig.add_trace(go.Bar(x=dp['Dia'], y=dp['Sono_Profundo_h'], name='Profundo', marker_color=PR))
            fig.add_hline(y=8, line_dash="dot", line_color=G, annotation_text="Alvo: 8h")
            fig.update_layout(barmode='overlay')
            st.plotly_chart(_style(fig, 300), use_container_width=True, config={"displayModeBar": False})

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Recovery Score ao Longo do Tempo")
        if 'Recovery_Score' in dp.columns:
            clrs = [_rc(v) for v in dp['Recovery_Score']]
            fig = go.Figure(go.Bar(x=dp['Dia'], y=dp['Recovery_Score'], marker_color=clrs))
            fig.add_hrect(y0=67, y1=105, fillcolor=f"{G}12", line_width=0)
            fig.add_hrect(y0=0, y1=33, fillcolor=f"{R}12", line_width=0)
            st.plotly_chart(_style(fig, 270), use_container_width=True, config={"displayModeBar": False})

    with c4:
        st.markdown("#### Dívida de Sono Acumulada (7d)")
        if 'Sleep_Debt_Acumulado_h' in dp.columns:
            clrs = [R if v > 3 else Y if v > 0 else G for v in dp['Sleep_Debt_Acumulado_h']]
            fig = go.Figure(go.Bar(x=dp['Dia'], y=dp['Sleep_Debt_Acumulado_h'], marker_color=clrs))
            fig.add_hline(y=0, line_color="#8b9ab1", line_width=1)
            fig.add_hline(y=3, line_dash="dot", line_color=R, annotation_text="Zona de risco")
            st.plotly_chart(_style(fig, 270), use_container_width=True, config={"displayModeBar": False})

    coach("📖 Como melhorar o sono profundo",
          f"Tiveste <strong>{sono_pr:.1f}h</strong> de sono profundo ({sono_pr/sono_tot*100:.0f}% do total). "
          "O ideal são 20–25%. Para aumentar: evita écrans 1h antes de dormir, "
          "mantém temperatura abaixo de 19°C, evita cafeína após as 14h. "
          "O sono profundo é quando o corpo liberta hormona de crescimento e repara tecidos musculares.")

# ════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Gestão de Risco":
    st.markdown("## Gestão de Risco & IA Preditiva")
    st.caption("ACWR, injury risk score, fadiga e classificação de sessão")

    acwr   = g_('ACWR', 1.0)
    injury = g_('Injury_Risk_Score', 0)
    z_c    = g_('Z_Score_Carga', 0)
    fadiga = g_('Indice_Fadiga', 0)
    assim  = g_('Assimetria_Percent', 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("ACWR", f"{acwr:.2f}", "zona óptima: 0.8 – 1.3", _c(acwr))
    with c2: kpi("Injury Risk", f"{injury:.0f}/100", "risco composto de lesão", _ic(injury))
    with c3: kpi("Z-Score Carga", f"{z_c:+.2f} σ", "desvio da carga habitual", R if abs(z_c) > 2 else BL)
    with c4: kpi("Índice Fadiga", f"{fadiga:.0f}/100", "impacto acumulado", R if fadiga > 80 else Y if fadiga > 50 else G)

    coach("📖 O que é o ACWR e porque é o indicador mais importante",
          "O <strong>Acute:Chronic Workload Ratio</strong> compara a tua carga das últimas 7 dias "
          "com a média dos últimos 28 dias. É o indicador mais validado pela ciência para prever lesões. "
          f"Hoje o teu ACWR é <strong>{acwr:.2f}</strong>. "
          f"{'⚠️ ZONA PERIGOSA: carga aguda muito acima da crónica. Risco de lesão real. Reduz volume imediatamente.' if acwr > 1.5 else '✅ Zona óptima — podes treinar com intensidade máxima.' if 0.8 <= acwr <= 1.3 else '🟡 Sub-treino — bom momento para aumentar carga gradualmente.'}")

    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### ACWR com Zonas de Risco")
        if 'ACWR' in dp.columns:
            fig = go.Figure()
            fig.add_hrect(y0=0.8, y1=1.3, fillcolor=f"{G}18", line_width=0,
                          annotation_text="Zona óptima", annotation_position="top left")
            fig.add_hrect(y0=1.3, y1=1.5, fillcolor=f"{Y}18", line_width=0,
                          annotation_text="Atenção", annotation_position="top right")
            fig.add_hrect(y0=1.5, y1=3.0, fillcolor=f"{R}18", line_width=0,
                          annotation_text="Perigo", annotation_position="top right")
            clrs = [_c(v) for v in dp['ACWR']]
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['ACWR'], mode='lines+markers',
                                     marker=dict(color=clrs, size=5),
                                     line=dict(color=BL, width=2.5), name='ACWR'))
            if 'Carga_Aguda' in dp.columns:
                fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Carga_Aguda'],
                                         name='Carga Aguda 7d', line=dict(color=OR, width=1.5, dash='dot'),
                                         yaxis='y2'))
                fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Carga_Cronica'],
                                         name='Carga Crónica 28d', line=dict(color=PR, width=1.5, dash='dot'),
                                         yaxis='y2'))
                fig.update_layout(
                    yaxis2=dict(overlaying='y', side='right', showgrid=False,
                                title='Carga G', color='#8b9ab1'))
            st.plotly_chart(_style(fig, 340), use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown("#### Injury Risk Score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=injury,
            title={"text": "Risco de Lesão", "font": {"size": 13, "color": "#8b9ab1"}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": _ic(injury)},
                   "steps": [{"range": [0, 35], "color": f"{G}22"},
                              {"range": [35, 60], "color": f"{Y}22"},
                              {"range": [60, 100], "color": f"{R}22"}],
                   "threshold": {"line": {"color": R, "width": 3},
                                 "thickness": 0.75, "value": 60}}))
        fig.update_layout(paper_bgcolor=BG, font_color="#c9d1d9",
                          height=295, margin=dict(l=12, r=12, t=32, b=8))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Injury Risk ao Longo do Tempo")
        if 'Injury_Risk_Score' in dp.columns:
            clrs = [_ic(v) for v in dp['Injury_Risk_Score']]
            fig = go.Figure(go.Bar(x=dp['Dia'], y=dp['Injury_Risk_Score'], marker_color=clrs))
            fig.add_hline(y=60, line_dash="dot", line_color=R, annotation_text="Zona de alerta")
            st.plotly_chart(_style(fig, 260), use_container_width=True, config={"displayModeBar": False})

    with c4:
        st.markdown("#### Assimetria Muscular (%)")
        if 'Assimetria_Percent' in dp.columns:
            clrs = [R if v > 10 else Y if v > 7 else G for v in dp['Assimetria_Percent']]
            fig = go.Figure(go.Bar(x=dp['Dia'], y=dp['Assimetria_Percent'], marker_color=clrs))
            fig.add_hline(y=10, line_dash="dot", line_color=R, annotation_text="Limite crítico")
            fig.add_hline(y=7, line_dash="dot", line_color=Y, annotation_text="Atenção")
            st.plotly_chart(_style(fig, 260), use_container_width=True, config={"displayModeBar": False})

    if 'Tipo_Sessao' in dp.columns:
        st.markdown("#### Distribuição de Tipos de Sessão")
        cnt = dp['Tipo_Sessao'].value_counts().reset_index()
        cnt.columns = ['Tipo', 'Dias']
        cmap = {'Alta Performance': G, 'Desenvolvimento': BL,
                'Base': PR, 'Recuperação': Y, '⚠️ Sobrecarga': R}
        fig = px.bar(cnt, x='Tipo', y='Dias', color='Tipo',
                     color_discrete_map=cmap, text='Dias')
        fig.update_layout(showlegend=False)
        st.plotly_chart(_style(fig, 260), use_container_width=True, config={"displayModeBar": False})

    coach("📖 O Injury Risk Score — os 4 fatores",
          "O score combina: <strong>ACWR</strong> (35%) + <strong>Assimetria muscular</strong> (25%) + "
          "<strong>Índice de fadiga</strong> (25%) + <strong>Temperatura corporal</strong> (15%). "
          f"Hoje: <strong>{injury:.0f}/100</strong>. "
          f"{'🔴 Recomendo sessão de recuperação ativa ou descanso total.' if injury > 60 else '🟡 Mantém atenção. Aquece bem e monitoriza a assimetria.' if injury > 35 else '🟢 Baixo risco — bom estado para treinar com intensidade.'}")

# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Visão Histórica":
    st.markdown("## Visão Histórica Completa")
    st.caption(f"Todos os indicadores · {len(dp)} dias de dados analisados")

    avg_r = dp[READ_COL].mean() if READ_COL in dp.columns else 0
    avg_hrv = dp['HRV_ms'].mean() if 'HRV_ms' in dp.columns else 0
    avg_d = dp['Distancia_km'].mean() if 'Distancia_km' in dp.columns else 0
    max_v = dp['Velocidade_Max_kmh'].max() if 'Velocidade_Max_kmh' in dp.columns else 0
    avg_rec = dp['Recovery_Score'].mean() if 'Recovery_Score' in dp.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Readiness Médio", f"{avg_r:.0f}%", f"{len(dp)} dias", _rc(avg_r))
    with c2: kpi("Recovery Médio", f"{avg_rec:.0f}%", "índice 3-pillar", _rc(avg_rec))
    with c3: kpi("HRV Médio", f"{avg_hrv:.0f} ms", "variabilidade cardíaca", G)
    with c4: kpi("Distância Média", f"{avg_d:.1f} km", "por sessão", BL)
    with c5: kpi("Sprint Máximo", f"{max_v:.1f} km/h", "recorde do período", PR)

    st.markdown("---")
    st.markdown("#### Dashboard Multi-Métrica")
    fig = go.Figure()
    pairs = [(READ_COL, 'Readiness', BL), ('Recovery_Score', 'Recovery', G),
             ('Injury_Risk_Score', 'Injury Risk', R)]
    for col, name, color in pairs:
        if col in dp.columns:
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp[col], name=name,
                                     line=dict(color=color, width=2)))
    if 'Strain_Score' in dp.columns:
        fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Strain_Score'] * (100 / 21),
                                  name='Strain (norm.)', line=dict(color=OR, width=2, dash='dot')))
    fig.update_layout(yaxis_range=[0, 105])
    st.plotly_chart(_style(fig, 360), use_container_width=True, config={"displayModeBar": False})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Carga Aguda vs Crónica")
        if all(c in dp.columns for c in ['Carga_Aguda', 'Carga_Cronica']):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Carga_Aguda'],
                                     name='Aguda 7d', line=dict(color=OR, width=2)))
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Carga_Cronica'],
                                     name='Crónica 28d', line=dict(color=BL, width=2)))
            st.plotly_chart(_style(fig, 270), use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown("#### Evolução da Resistência (média 7d)")
        if 'Resistencia' in dp.columns:
            roll = dp['Resistencia'].rolling(7, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dp['Dia'], y=dp['Resistencia'], name='Diária',
                                     line=dict(color=BL, width=1.2), opacity=0.4))
            fig.add_trace(go.Scatter(x=dp['Dia'], y=roll, name='Média 7d',
                                     line=dict(color=G, width=2.5)))
            st.plotly_chart(_style(fig, 270), use_container_width=True, config={"displayModeBar": False})

    with st.expander("📋 Tabela de dados completa"):
        show = [c for c in ['Dia', READ_COL, 'Recovery_Score', 'Strain_Score',
                             'Injury_Risk_Score', 'ACWR', 'HRV_ms', 'Horas_Sono',
                             'Distancia_km', 'Velocidade_Max_kmh',
                             'Tipo_Sessao', 'HRV_Trend', 'Readiness_Tier'] if c in dp.columns]
        num_cols = [c for c in show if c not in ['Tipo_Sessao', 'HRV_Trend', 'Readiness_Tier', 'Dia']]
        st.dataframe(dp[show].set_index('Dia').style.format(
            {c: "{:.1f}" for c in num_cols}, na_rep="—"), use_container_width=True)
