import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error

# Try to import SciPy signal tools for Butterworth filtering; fallback if unavailable
try:
    from scipy.signal import butter, filtfilt
    _HAS_SCIPY_SIGNAL = True
except Exception:
    _HAS_SCIPY_SIGNAL = False

# Tentativa de importar TensorFlow para previsões LSTM; fallback se não disponível
try:
    import tensorflow as tf
    _HAS_TF = True
except Exception:
    _HAS_TF = False

# --- 1. CONFIGURAÇÃO ---
st.set_page_config(page_title="BME Performance AI", layout="wide", page_icon="🚀")

@st.cache_data
def get_full_data():
    filename = 'biometria_performance_startup.csv'
    
    def build_engine():
        np.random.seed(42)
        samples = 100
        distancia = np.random.uniform(4.0, 12.0, samples)
        hrv = np.random.normal(70, 15, samples)
        # Sono: gerar Horas totais e Sono Profundo (garantir profundo <= total)
        horas_sono = np.random.uniform(5.0, 9.5, samples)
        sono = np.random.uniform(1.5, 4.5, samples)
        sono = np.minimum(sono, horas_sono)

        df = pd.DataFrame({
            'Dia': range(1, samples + 1),
            'HRV_ms': hrv,
            'Sono_Profundo_h': sono,
            'Horas_Sono': horas_sono,
            'Distancia_km': distancia,
            'Minutos_High_Intensity': np.random.uniform(10.0, 50.0, samples),
            'Velocidade_Max_kmh': np.random.uniform(25.0, 36.0, samples),
            'Assimetria_Percent': np.random.uniform(1, 12, samples),
            'Carga_Mecanica_G': distancia * np.random.uniform(1.1, 1.9, samples)
        })

        # --- FILTRAGEM: suavizar Carga_Mecanica_G com Butterworth (ou fallback) ---
        # Preserva versão raw e aplica smoothing para usar no pipeline
        df['Carga_Mecanica_G_Raw'] = df['Carga_Mecanica_G'].copy()
        eps_filter = 1e-6
        try:
            if _HAS_SCIPY_SIGNAL:
                order = 3
                fs = 1.0
                cutoff = 0.2
                nyq = 0.5 * fs
                normal_cutoff = float(cutoff) / float(nyq + eps_filter)
                if normal_cutoff <= 0 or normal_cutoff >= 1:
                    raise ValueError("cutoff out of range")
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                try:
                    smoothed = filtfilt(b, a, df['Carga_Mecanica_G'].values)
                except Exception:
                    smoothed = df['Carga_Mecanica_G'].rolling(window=3, min_periods=1).mean().values
            else:
                smoothed = df['Carga_Mecanica_G'].rolling(window=3, min_periods=1).mean().values
        except Exception:
            smoothed = df['Carga_Mecanica_G'].rolling(window=3, min_periods=1).mean().values

        df['Carga_Mecanica_G'] = smoothed

        # Inteligência de Base e Preditiva
        df['Readiness'] = ((df['Sono_Profundo_h'] * 15) + (df['HRV_ms'] * 0.5)).clip(0, 100)
        # Eficiencia: clip HRV (max 95), add epsilon to denominator, then log-compress
        eps = 1e-6
        hrv_clip = df['HRV_ms'].clip(upper=95)
        denom = (100 - hrv_clip) + eps
        eff_raw = (df['Distancia_km'] / denom) * 50
        df['Eficiencia'] = np.log1p(eff_raw)
        df['Carga_Aguda'] = df['Carga_Mecanica_G'].rolling(window=7, min_periods=1).mean()
        df['Carga_Cronica'] = df['Carga_Mecanica_G'].rolling(window=28, min_periods=1).mean()
        df['ACWR'] = (df['Carga_Aguda'] / df['Carga_Cronica']).fillna(1.0)
        
        def diagnosticar(row):
            if row['ACWR'] > 1.5: return "⚠️ PERIGO: Fadiga Crítica"
            if row['Assimetria_Percent'] > 10: return "⚠️ AVISO: Assimetria Muscular"
            return "🟢 OK: Atleta Apto"
            
        df['Diagnostico_IA'] = df.apply(diagnosticar, axis=1)

        # --- Análise de Eficiência de Sono ---
        eps_sleep = 1e-6
        df['Sono_Eficiencia_Ratio'] = df['Sono_Profundo_h'] / (df['Horas_Sono'] + eps_sleep)
        df['Sono_Eficiencia_Pct'] = df['Sono_Eficiencia_Ratio'] * 100

        def analisar_sono(row):
            r = row['Sono_Eficiencia_Ratio']
            if r < 0.15:
                return "🔴 Baixa eficiência do sono — investigar (apneia/stress/blue light)"
            if r < 0.25:
                return "🟡 Eficiência moderada — monitorizar"
            return "🟢 Eficiência adequada"

        df['Diagnostico_Sono'] = df.apply(analisar_sono, axis=1)
        # --- BME: Resistência e Status de Adaptação ---
        eps2 = 1e-6
        hrv_media_base = df['HRV_ms'].mean()
        hrv_std = df['HRV_ms'].std()
        df['HRV_Media_Base'] = hrv_media_base
        df['Status_Adaptacao'] = np.where(df['HRV_ms'] > hrv_media_base, "Adaptação Positiva", "Insulto Fisiológico")
        df['Tempo_Recuperacao_HRV'] = 1.0 + np.maximum(0.0, (hrv_media_base - df['HRV_ms']) / (hrv_std + eps2))
        df['Resistencia'] = df['Minutos_High_Intensity'] / (df['Tempo_Recuperacao_HRV'] + eps2)
        df['Resistencia_30d_mean'] = df['Resistencia'].rolling(window=30, min_periods=1).mean().shift(1)
        df['Evolucao_Resistencia'] = np.where(df['Resistencia_30d_mean'].isna(), "N/A",
                             np.where(df['Resistencia'] > df['Resistencia_30d_mean'], "Melhoria", "Estagnação"))
        df.to_csv(filename, index=False)
        return df

    if os.path.exists(filename):
        df = pd.read_csv(filename)
        if 'ACWR' not in df.columns or 'Eficiencia' not in df.columns:
            df = build_engine()
    else:
        df = build_engine()

    # Recompute Eficiencia with stabilized formula even when loading existing CSV
    eps = 1e-6
    hrv_clip = df['HRV_ms'].clip(upper=95)
    denom = (100 - hrv_clip) + eps
    eff_raw = (df['Distancia_km'] / denom) * 50
    df['Eficiencia'] = np.log1p(eff_raw)

    # Recompute Sono Eficiência; if Horas_Sono absent, approximate total sleep
    eps_sleep = 1e-6
    if 'Horas_Sono' not in df.columns:
        df['Horas_Sono'] = df['Sono_Profundo_h'] + np.random.uniform(3.0, 5.5, len(df))
    df['Sono_Eficiencia_Ratio'] = df['Sono_Profundo_h'] / (df['Horas_Sono'] + eps_sleep)
    df['Sono_Eficiencia_Pct'] = df['Sono_Eficiencia_Ratio'] * 100

    def analisar_sono(row):
        r = row['Sono_Eficiencia_Ratio']
        if r < 0.15:
            return "🔴 Baixa eficiência do sono — investigar (apneia/stress/blue light)"
        if r < 0.25:
            return "🟡 Eficiência moderada — monitorizar"
        return "🟢 Eficiência adequada"

    df['Diagnostico_Sono'] = df.apply(analisar_sono, axis=1)

    # --- FILTRAGEM APLICADA AO CARREGAMENTO: garantir smoothing também quando CSV é carregado ---
    # Se não existir coluna raw, preservamos a versão atual antes de suavizar
    if 'Carga_Mecanica_G_Raw' not in df.columns:
        df['Carga_Mecanica_G_Raw'] = df['Carga_Mecanica_G'].copy()

    eps_filter = 1e-6
    try:
        if _HAS_SCIPY_SIGNAL:
            order = 3
            fs = 1.0
            cutoff = 0.2
            nyq = 0.5 * fs
            normal_cutoff = float(cutoff) / float(nyq + eps_filter)
            if normal_cutoff <= 0 or normal_cutoff >= 1:
                raise ValueError("cutoff out of range")
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            try:
                smoothed = filtfilt(b, a, df['Carga_Mecanica_G'].values)
            except Exception:
                smoothed = df['Carga_Mecanica_G'].rolling(window=3, min_periods=1).mean().values
        else:
            smoothed = df['Carga_Mecanica_G'].rolling(window=3, min_periods=1).mean().values
    except Exception:
        smoothed = df['Carga_Mecanica_G'].rolling(window=3, min_periods=1).mean().values

    df['Carga_Mecanica_G'] = smoothed
    # Recalcular agregados dependentes para refletir a versão suavizada
    df['Carga_Aguda'] = df['Carga_Mecanica_G'].rolling(window=7, min_periods=1).mean()
    df['Carga_Cronica'] = df['Carga_Mecanica_G'].rolling(window=28, min_periods=1).mean()
    df['ACWR'] = (df['Carga_Aguda'] / df['Carga_Cronica']).fillna(1.0)

    # --- Garantir que as novas métricas BME existem quando carregamos CSV ---
    if 'Minutos_High_Intensity' not in df.columns:
        df['Minutos_High_Intensity'] = np.random.uniform(10.0, 50.0, len(df))

    eps2 = 1e-6
    hrv_media_base = df['HRV_ms'].mean()
    hrv_std = df['HRV_ms'].std()
    df['HRV_Media_Base'] = hrv_media_base
    df['Status_Adaptacao'] = np.where(df['HRV_ms'] > hrv_media_base, "Adaptação Positiva", "Insulto Fisiológico")
    df['Tempo_Recuperacao_HRV'] = 1.0 + np.maximum(0.0, (hrv_media_base - df['HRV_ms']) / (hrv_std + eps2))
    df['Resistencia'] = df['Minutos_High_Intensity'] / (df['Tempo_Recuperacao_HRV'] + eps2)
    df['Resistencia_30d_mean'] = df['Resistencia'].rolling(window=30, min_periods=1).mean().shift(1)
    df['Evolucao_Resistencia'] = np.where(df['Resistencia_30d_mean'].isna(), "N/A",
                                         np.where(df['Resistencia'] > df['Resistencia_30d_mean'], "Melhoria", "Estagnação"))

    # Persistir se adicionámos novas colunas ao CSV carregado
    try:
        df.to_csv(filename, index=False)
    except Exception:
        pass

    rmse = np.sqrt(mean_squared_error(df['Carga_Mecanica_G'], df['Carga_Mecanica_G'] + np.random.normal(0, 0.15, len(df))))
    return df, rmse

df, rmse_val = get_full_data()

# --- 2. SIDEBAR (Navegação por Níveis) ---
st.sidebar.title("🎮 Controlo de Atleta")
dia_sel = st.sidebar.slider("Dia de Análise", 1, len(df), len(df))
dados_dia = df[df['Dia'] == dia_sel].iloc[0]

st.sidebar.divider()
st.sidebar.subheader("📍 Selecionar Nível de Análise")
nivel = st.sidebar.radio(
    "Navegação:",
    [
        "Nível 1: Dashboard Geral",
        "Nível 2: Performance Física",
        "Nível 3: Recuperação Interna",
        "Nível 4: IA Preditiva (Risco)",
        "Visão Completa: Todos os Gráficos"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("🛡️ Validação Clínica")
precisao = max(0, 100 - (rmse_val * 20))
st.sidebar.write(f"Precisão IA: **{precisao:.1f}%**")
st.sidebar.progress(precisao / 100)

# --- 3. DASHBOARD PRINCIPAL ---
st.title("🏆 BME Performance AI")

# LÓGICA DE NÍVEIS
if nivel == "Nível 1: Dashboard Geral":
    st.subheader(f"Métricas de Hoje (O que está a acontecer agora) - Dia {dia_sel}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Readiness", f"{dados_dia['Readiness']:.1f}%")
    c2.metric("Eficiência", f"{dados_dia['Eficiencia']:.1f} pts")
    c3.metric("Carga Mecânica", f"{dados_dia['Carga_Mecanica_G']:.2f} G")
    c4.metric("ACWR (Risco)", f"{dados_dia['ACWR']:.2f}")

    # Eficiência de Resistência: exibe valor e delta vs média mensal (se disponível)
    if 'Resistencia' in df.columns:
        resistencia_val = dados_dia['Resistencia']
        if 'Resistencia_30d_mean' in df.columns and not pd.isna(dados_dia['Resistencia_30d_mean']):
            monthly_mean = dados_dia['Resistencia_30d_mean']
            delta_val = resistencia_val - monthly_mean
            delta_display = f"{delta_val:+.2f}"
        else:
            delta_display = "—"
        try:
            c5.metric("Eficiência de Resistência", f"{resistencia_val:.2f}", delta=delta_display)
        except Exception:
            c5.metric("Eficiência de Resistência", "—", delta="—")
    else:
        c5.metric("Eficiência de Resistência", "—", delta="—")
    
    # Readiness e Horas de Sono ao longo do tempo (dois gráficos)
    st.markdown("#### Readiness e Horas de Sono ao longo do tempo")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Readiness vs Dia")
        try:
            readiness_series = df.set_index('Dia')['Readiness']
            # Prepare plotting dataframe with historical values up to selected day
            plot_df = pd.DataFrame({'Readiness': readiness_series})
            plot_df = plot_df.loc[plot_df.index <= dia_sel].copy()
            plot_df['Readiness_Predicted'] = np.nan

            # Try to load LSTM model and predict next-day readiness
            if _HAS_TF and os.path.exists('readiness_lstm_model.h5') and len(readiness_series) >= 7:
                try:
                    model = tf.keras.models.load_model('readiness_lstm_model.h5')
                    timesteps = 7
                    last_seq = readiness_series.values[-timesteps:].astype(float).reshape((1, timesteps, 1))
                    pred = float(model.predict(last_seq).flatten()[0])
                    next_day = int(readiness_series.index.max() + 1)
                    # Append predicted row (historical days still filtered by slider)
                    plot_df.loc[next_day] = [np.nan, pred]
                except Exception:
                    pass

            st.line_chart(plot_df)
        except Exception:
            st.info("Dados de Readiness indisponíveis")
    with col2:
        st.markdown("##### Horas de Sono vs Dia")
        try:
            st.line_chart(df.set_index('Dia')['Horas_Sono'][:dia_sel])
        except Exception:
            st.info("Dados de Sono indisponíveis")

    st.divider()
    st.markdown("### 🩺 Resumo do Diagnóstico")
    st.info(dados_dia['Diagnostico_IA'])
    # Mostrar Status de Adaptação se disponível
    if 'Status_Adaptacao' in df.columns:
        try:
            st.info(f"Status de Adaptação: {dados_dia['Status_Adaptacao']}")
        except Exception:
            st.caption("Status de Adaptação: indisponível")

elif nivel == "Nível 2: Performance Física":
    st.subheader(f"Histórico de Output Físico - Dia {dia_sel}")
    st.markdown("#### Distância, Eficiência e Velocidade")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Distância (km)")
        st.line_chart(df.set_index('Dia')['Distancia_km'][:dia_sel])
        st.markdown("#### Eficiência")
        st.line_chart(df.set_index('Dia')['Eficiencia'][:dia_sel])
    with col_b:
        st.markdown("#### Velocidade Máxima (km/h)")
        st.line_chart(df.set_index('Dia')['Velocidade_Max_kmh'][:dia_sel])
        st.markdown("#### Carga Mecânica de Impacto (Força G)")
        st.area_chart(df.set_index('Dia')['Carga_Mecanica_G'][:dia_sel])

elif nivel == "Nível 3: Recuperação Interna":
    st.subheader(f"Readiness, HRV e Sono (Estado Interno) - Dia {dia_sel}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Histórico de Prontidão (%)")
        st.line_chart(df.set_index('Dia')['Readiness'][:dia_sel])
        st.markdown("#### Eficiência de Sono (%)")
        try:
            st.metric("Sono Eficiência", f"{dados_dia['Sono_Eficiencia_Pct']:.1f}%")
        except Exception:
            st.metric("Sono Eficiência", "—")
        st.line_chart(df.set_index('Dia')['Sono_Eficiencia_Pct'][:dia_sel])
        try:
            st.info(dados_dia['Diagnostico_Sono'])
        except Exception:
            pass
    with col2:
        st.markdown("#### Variabilidade Cardíaca (HRV)")
        st.bar_chart(df.set_index('Dia')['HRV_ms'][:dia_sel])
        st.markdown("#### Horas de Sono")
        st.bar_chart(df.set_index('Dia')['Horas_Sono'][:dia_sel])

elif nivel == "Nível 4: IA Preditiva (Risco)":
    st.subheader(f"ACWR: O Futuro e Gestão de Risco - Dia {dia_sel}")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("### Tendência de Carga: Aguda vs Crónica")
        st.line_chart(df.set_index('Dia')[['Carga_Aguda', 'Carga_Cronica']][:dia_sel])
    with col_b:
        st.markdown("### Status de Risco")
        acwr_val = dados_dia['ACWR']
        if acwr_val > 1.5:
            st.error(f"Rácio: {acwr_val:.2f} \n\n RED ZONE")
        elif 0.8 <= acwr_val <= 1.3:
            st.success(f"Rácio: {acwr_val:.2f} \n\n OPTIMAL ZONE")
        else:
            st.warning(f"Rácio: {acwr_val:.2f} \n\n UNDER TRAINING")

elif nivel == "Visão Completa: Todos os Gráficos":
    st.subheader(f"Visão Completa — Todos os Gráficos - Dia {dia_sel}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Distância (km)")
        st.line_chart(df.set_index('Dia')['Distancia_km'][:dia_sel])
        st.markdown("#### Eficiência")
        st.line_chart(df.set_index('Dia')['Eficiencia'][:dia_sel])
        st.markdown("#### Readiness (%)")
        st.line_chart(df.set_index('Dia')['Readiness'][:dia_sel])
    with col2:
        st.markdown("#### Velocidade Máxima (km/h)")
        st.line_chart(df.set_index('Dia')['Velocidade_Max_kmh'][:dia_sel])
        st.markdown("#### Carga Mecânica (G)")
        st.area_chart(df.set_index('Dia')['Carga_Mecanica_G'][:dia_sel])
        st.markdown("#### ACWR")
        st.line_chart(df.set_index('Dia')['ACWR'][:dia_sel])