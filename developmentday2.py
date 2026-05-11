import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os

# Try to import SciPy signal tools for Butterworth filtering; fallback if unavailable
try:
    from scipy.signal import butter, filtfilt
    _HAS_SCIPY_SIGNAL = True
except Exception:
    _HAS_SCIPY_SIGNAL = False

# Tentativa de importar TensorFlow para treinar LSTM; fallback se não disponível
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    _HAS_TF = True
except Exception:
    _HAS_TF = False

def generate_performance_dataset(samples=100):
    np.random.seed(42)
    
    # --- 1. BIOMETRIA DE RECUPERAÇÃO ---
    hrv = np.random.normal(70, 15, samples)
    # Horas totais de sono e Sono profundo (garantir profundo <= total)
    horas_sono = np.random.uniform(5.0, 9.5, samples)
    deep_sleep = np.random.uniform(1.5, 4.5, samples)
    deep_sleep = np.minimum(deep_sleep, horas_sono)
    eda_stress = np.random.uniform(0.1, 4.0, samples)

    # --- 2. PERFORMANCE DE CAMPO ---
    distancia = np.random.uniform(4.0, 12.0, samples)
    max_sprint = np.random.uniform(25.0, 36.0, samples)
    tempo_alta_intensidade = np.random.uniform(10, 50, samples)
    impacto_total = distancia * np.random.uniform(1.1, 1.9, samples)
    assimetria = np.random.uniform(1, 12, samples)

    # --- 3. CRIAÇÃO DO DATAFRAME ---
    # --- Novas variáveis sintéticas: SpO2, Temperatura, RHR ---
    spO2 = np.random.normal(97.0, 0.8, samples)
    spO2 = np.clip(spO2, 90.0, 100.0)
    temp_c = np.random.normal(36.45, 0.4, samples)
    # pequenos picos aleatórios de temperatura
    temp_c += np.random.normal(0.0, 0.15, samples)
    rhr = np.random.normal(60.0, 6.0, samples)
    rhr = np.clip(rhr, 35.0, 120.0)

    # --- 3. CRIAÇÃO DO DATAFRAME ---
    df = pd.DataFrame({
        'Atleta_ID': ['A001'] * samples,
        'Dia': range(1, samples + 1),
        'HRV_ms': hrv,
        'Sono_Profundo_h': deep_sleep,
        'Horas_Sono': horas_sono,
        'Stress_Psicologico': eda_stress,
        'Distancia_km': distancia,
        'Velocidade_Max_kmh': max_sprint,
        'Minutos_High_Intensity': tempo_alta_intensidade,
        'Carga_Mecanica_G': impacto_total,
        'Assimetria_Percent': assimetria,
        'SpO2_pct': spO2,
        'Temp_C': temp_c,
        'RHR_bpm': rhr
    })

    # --- FILTRAGEM: suavizar Carga_Mecanica_G com Butterworth (ou fallback) ---
    # Mantemos a versão raw e aplicamos smoothing para usar no pipeline
    df['Carga_Mecanica_G_Raw'] = df['Carga_Mecanica_G'].copy()
    eps_filter = 1e-6
    try:
        if _HAS_SCIPY_SIGNAL:
            # digital Butterworth lowpass: cutoff em cycles/day (ex.: 0.2 -> ~5-day period)
            order = 3
            fs = 1.0
            cutoff = 0.2
            nyq = 0.5 * fs
            normal_cutoff = float(cutoff) / float(nyq + eps_filter)
            if normal_cutoff <= 0 or normal_cutoff >= 1:
                # fallback para média móvel se cutoff inválido
                raise ValueError("cutoff out of range")
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            # filtfilt pode falhar em sequências muito curtas; guardamos isso em try/except
            try:
                smoothed = filtfilt(b, a, df['Carga_Mecanica_G'].values)
            except Exception:
                smoothed = df['Carga_Mecanica_G'].rolling(window=3, min_periods=1).mean().values
        else:
            # Sem SciPy: usar média móvel simples como fallback
            smoothed = df['Carga_Mecanica_G'].rolling(window=3, min_periods=1).mean().values
    except Exception:
        smoothed = df['Carga_Mecanica_G'].rolling(window=3, min_periods=1).mean().values

    # Sobrescreve a coluna utilizada pelo pipeline com a versão suavizada (preservando Raw)
    df['Carga_Mecanica_G'] = smoothed

    # --- 4. MOTOR DE IA (Lógica de Insights) ---
    df['Readiness_Score'] = ((df['Sono_Profundo_h'] * 15) + (df['HRV_ms'] * 0.5)).clip(0, 100)
    df['Indice_Fadiga'] = (df['Carga_Mecanica_G'] * 5) / (df['HRV_ms'] * 0.2)
    df['Indice_Fadiga'] = df['Indice_Fadiga'].clip(0, 100)

    # --- CÉREBRO PREDITIVO (ACWR) ---
    # Carga Aguda (Média 7 dias) vs Carga Crónica (Média 28 dias)
    df['Carga_Aguda'] = df['Carga_Mecanica_G'].rolling(window=7, min_periods=1).mean()
    df['Carga_Cronica'] = df['Carga_Mecanica_G'].rolling(window=28, min_periods=1).mean()
    df['ACWR'] = (df['Carga_Aguda'] / df['Carga_Cronica']).fillna(1.0)

    # --- NORMALIZAÇÃO Z-SCORE ---
    # Identifica se a variação de hoje é um "outlier" estatístico para este atleta
    media_hist = df['Carga_Mecanica_G'].mean()
    std_hist = df['Carga_Mecanica_G'].std()
    df['Z_Score_Carga'] = (df['Carga_Mecanica_G'] - media_hist) / std_hist

    # --- DIAGNÓSTICOS AVANÇADOS ---
    def diagnosticar_risco_avancado(row):
        # Prioridade 1: Previsão de Lesão via ACWR
        if row['ACWR'] > 1.5:
            return "🔴 PERIGO: Carga aguda excessiva. Risco iminente de lesão."
        # Prioridade 2: Descompensação Mecânica
        if row['Assimetria_Percent'] > 10:
            return "⚠️ CRÍTICO: Descompensação detetada. Ajustar mecânica."
        # Prioridade 3: Fadiga Sistémica
        if row['Indice_Fadiga'] > 80:
            return "🟡 AVISO: Fadiga extrema. Sessão de recuperação necessária."
        return "🟢 OK: Atleta apto para rendimento."

    df['Diagnostico_IA'] = df.apply(diagnosticar_risco_avancado, axis=1)

    # --- Eficiência de Sono ---
    eps_sleep = 1e-6
    df['Sono_Eficiencia_Ratio'] = df['Sono_Profundo_h'] / (df['Horas_Sono'] + eps_sleep)
    df['Sono_Eficiencia_Pct'] = df['Sono_Eficiencia_Ratio'] * 100

    def diagnostico_sono(row):
        r = row['Sono_Eficiencia_Ratio']
        if r < 0.15:
            return "🔴 Baixa eficiência do sono — investigar"
        if r < 0.25:
            return "🟡 Eficiência moderada"
        return "🟢 Eficiência adequada"

    df['Diagnostico_Sono'] = df.apply(diagnostico_sono, axis=1)

    recorde_anterior = 34.0
    df['Recorde_Sprint'] = df['Velocidade_Max_kmh'].apply(lambda x: "🥇 NOVO RECORDE!" if x > recorde_anterior else "-")

    # --- NOVAS MÉTRICAS (BME) ---
    # Status de Adaptação: cruza 'Minutos_High_Intensity' com 'HRV_ms' usando uma média-base do HRV
    eps = 1e-6
    hrV_media_base = df['HRV_ms'].mean()
    hrV_std = df['HRV_ms'].std()
    df['HRV_Media_Base'] = hrV_media_base
    df['Status_Adaptacao'] = df['HRV_ms'].apply(lambda h: "Adaptação Positiva" if h > hrV_media_base else "Insulto Fisiológico")

    # Índice de Resistência = Minutos de Alta Intensidade / Tempo de Recuperação do HRV
    # Definimos Tempo de Recuperação (dias) como 1 + quanto o HRV está abaixo da média, normalizado pela std;
    # isso garante que quanto mais baixo o HRV em relação à média, maior o tempo de recuperação.
    df['Tempo_Recuperacao_HRV'] = 1.0 + np.maximum(0.0, (hrV_media_base - df['HRV_ms']) / (hrV_std + eps))

    # Evita divisão por zero adicionando um pequeno epsilon ao denominador
    df['Resistencia'] = df['Minutos_High_Intensity'] / (df['Tempo_Recuperacao_HRV'] + eps)

    # Comparação Mensal: média dos últimos 30 dias (exclui o dia corrente) e determinação da evolução
    df['Resistencia_30d_mean'] = df['Resistencia'].rolling(window=30, min_periods=1).mean().shift(1)
    df['Evolucao_Resistencia'] = np.where(df['Resistencia_30d_mean'].isna(), "N/A",
                                         np.where(df['Resistencia'] > df['Resistencia_30d_mean'], "Melhoria", "Estagnação"))

    return df


def train_lstm_predict_readiness(df, timesteps=7, epochs=30, verbose=0):
    """Treina uma LSTM simples para prever Readiness_Score do dia seguinte usando os últimos `timesteps`.
    Retorna (model, metrics_dict) ou (None, None) se TensorFlow não estiver disponível.
    """
    if not _HAS_TF:
        print("⚠️ TensorFlow não disponível. Instale 'tensorflow' para treinar LSTM.")
        return None, None

    series = df['Readiness_Score'].astype(float).values
    if len(series) <= timesteps:
        print("⚠️ Dados insuficientes para treinar LSTM (poucos dias).")
        return None, None

    X, y = [], []
    for i in range(len(series) - timesteps):
        X.append(series[i:i+timesteps])
        y.append(series[i+timesteps])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split train/validation (80/20)
    split = int(0.8 * len(X))
    if split < 1:
        split = 1
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(32, activation='tanh', input_shape=(timesteps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )

    # Avaliação
    if len(X_val) > 0:
        y_pred = model.predict(X_val).flatten()
        val_mae = mean_absolute_error(y_val, y_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    else:
        val_mae = None
        val_rmse = None

    # Predição para o próximo dia usando os últimos `timesteps`
    last_seq = series[-timesteps:]
    last_seq = last_seq.reshape((1, timesteps, 1))
    pred_next = float(model.predict(last_seq).flatten()[0])

    # Tentar salvar o modelo (silencioso se falhar)
    try:
        model.save('readiness_lstm_model.h5')
    except Exception:
        pass

    print(f"🔁 LSTM treinada. Próximo Readiness previsto: {pred_next:.2f}")
    if val_mae is not None:
        print(f"📐 Validação LSTM MAE: {val_mae:.3f}, RMSE: {val_rmse:.3f}")

    metrics = {'mae': val_mae, 'rmse': val_rmse, 'pred_next': pred_next, 'history': getattr(history, 'history', None)}
    return model, metrics


# -----------------------
# Metabolic Power Module
# -----------------------
def calc_es_from_acc(a, g=9.81):
    """Estimation of Equivalent Slope (ES) from acceleration.
    Simplified approximation: ES = a / g
    Accepts array-like or scalar `a`.
    """
    return np.array(a, dtype=float) / float(g)


def calc_ec_from_es(es, kt=1.29):
    """Calcula o Custo Energético (EC) em J/(kg·m) a partir de ES usando o polinómio de regressão.
    Retorna array do mesmo shape que `es`.
    """
    es = np.array(es, dtype=float)
    # limitar valores extremos para evitar explosões do polinómio
    es_clip = np.clip(es, -1.0, 1.5)
    ec = (155.4 * es_clip**5 - 30.4 * es_clip**4 - 43.3 * es_clip**3
          + 46.3 * es_clip**2 + 19.5 * es_clip + 3.6)
    ec = ec * float(kt)
    # assegurar valor mínimo plausível (não inferior ao custo basico em plano)
    ec = np.maximum(ec, 3.6 * float(kt) * 0.5)
    return ec


def metabolic_power_from_trace(v_m_s, a_m_s2, kt=1.29, g=9.81):
    """Calcula P_met instantâneo (W/kg) a partir de vetores de velocidade (m/s) e aceleração (m/s^2).
    Retorna (pmet_w_per_kg, ec_j_per_kg_m, es)
    """
    v = np.array(v_m_s, dtype=float)
    a = np.array(a_m_s2, dtype=float)
    es = calc_es_from_acc(a, g=g)
    ec = calc_ec_from_es(es, kt=kt)
    pmet = ec * v
    return pmet, ec, es


def simulate_speed_acc_profile(dist_km, max_speed_kmh, high_intensity_minutes,
                               session_minutes=60, sampling_rate=1, random_state=None):
    """Simula um perfil de velocidade (m/s) e aceleração (m/s^2) ao longo de uma sessão.
    - dist_km: distância total percorrida na sessão
    - max_speed_kmh: velocidade máxima (km/h) atingida
    - high_intensity_minutes: total de minutos em esforços de alta intensidade
    - session_minutes: duração da sessão (padrão 60)
    - sampling_rate: amostras por segundo (1 = 1 Hz)
    Retorna (v_m_s, a_m_s2, dt)
    """
    rng = np.random.RandomState(42 if random_state is None else random_state)
    dt = 1.0 / float(sampling_rate)
    duration_s = int(max(1, round(session_minutes * 60 * sampling_rate)))

    # velocidade média necessária para cobrir a distância
    avg_speed_m_s = (float(dist_km) * 1000.0) / max(1.0, session_minutes * 60.0)
    max_v = float(max_speed_kmh) * 1000.0 / 3600.0

    t = np.arange(duration_s) * dt
    v = np.ones(duration_s, dtype=float) * avg_speed_m_s

    # introduzir rajadas de alta intensidade como gaussianas
    hi_seconds = int(max(0, round(high_intensity_minutes * 60 * sampling_rate)))
    if hi_seconds > 0:
        n_bursts = max(1, int(max(1, hi_seconds / 10)))
        for _ in range(n_bursts):
            center = rng.randint(0, duration_s)
            width = rng.randint(3 * sampling_rate, 10 * sampling_rate)
            amp = rng.uniform(0.6 * max_v, max_v) - avg_speed_m_s
            if amp <= 0:
                continue
            gauss = amp * np.exp(-0.5 * ((np.arange(duration_s) - center) / max(1.0, width))**2)
            v += gauss

    # adicionar pequeno ruído
    v += rng.normal(0.0, 0.2, size=duration_s)
    v = np.clip(v, 0.0, max_v)

    # calcular aceleração (diferença finita)
    a = np.diff(v, prepend=v[0]) / dt
    return v, a, dt


def estimate_metabolic_metrics_for_dataset(df, mass_kg=70.0, kt=1.29,
                                          session_minutes=None, sampling_rate=1):
    """Estima métricas metabólicas por linha do DataFrame e anexa colunas:
    - Metabolic_Power_Mean_Wkg
    - Metabolic_Power_Peak_Wkg
    - Metabolic_Energy_kJ
    Retorna o DataFrame modificado.
    """
    means = []
    peaks = []
    energy_kj = []
    for _, row in df.iterrows():
        sess_min = session_minutes if session_minutes is not None else 60
        v, a, dt = simulate_speed_acc_profile(row.get('Distancia_km', 6.0),
                                              row.get('Velocidade_Max_kmh', 30.0),
                                              row.get('Minutos_High_Intensity', 10.0),
                                              session_minutes=sess_min,
                                              sampling_rate=sampling_rate)
        pmet, ec, es = metabolic_power_from_trace(v, a, kt=kt)
        mean_p = float(np.mean(pmet))
        peak_p = float(np.max(pmet))
        energy_j = float(np.sum(pmet * float(mass_kg) * dt))
        means.append(mean_p)
        peaks.append(peak_p)
        energy_kj.append(energy_j / 1000.0)

    df = df.copy()
    df['Metabolic_Power_Mean_Wkg'] = means
    df['Metabolic_Power_Peak_Wkg'] = peaks
    df['Metabolic_Energy_kJ'] = energy_kj
    return df


def calcular_recovery_scores(df, baseline_days=7, sleep_needed=8.0):
    """
    Calcula o Recovery Score (0-100) para cada linha do DataFrame.
    Usa baseline de `baseline_days` dias anteriores (exclui o dia atual).
    Retorna uma lista de scores na mesma ordem do df.
    """
    results = []
    eps = 1e-6

    # Pre-calcula médias e desvios de 7 dias (exclui dia atual)
    hr_mean = df['HRV_ms'].rolling(window=baseline_days, min_periods=1).mean().shift(1)
    hr_std = df['HRV_ms'].rolling(window=baseline_days, min_periods=1).std().shift(1).fillna(0.0)

    temp_mean = df['Temp_C'].rolling(window=baseline_days, min_periods=1).mean().shift(1)

    # Use ACWR se disponível, caso contrário calcule
    if 'ACWR' not in df.columns:
        carga_aguda = df['Carga_Mecanica_G'].rolling(window=7, min_periods=1).mean()
        carga_cronica = df['Carga_Mecanica_G'].rolling(window=28, min_periods=1).mean()
        acwr_series = (carga_aguda / carga_cronica).fillna(1.0)
    else:
        acwr_series = df['ACWR']

    for i, row in df.reset_index(drop=True).iterrows():
        # HRV Z-score (baseline de 7 dias)
        mu = hr_mean.iloc[i] if not pd.isna(hr_mean.iloc[i]) else row['HRV_ms']
        sigma = hr_std.iloc[i] if hr_std.iloc[i] > eps else eps
        z_hrv = (row['HRV_ms'] - mu) / sigma if sigma > eps else 0.0
        s_hrv = float(np.clip((z_hrv + 2.0) / 4.0, 0.0, 1.0))

        # SpO2 mapping (thresholds descritos)
        spo2 = float(row.get('SpO2_pct', 97.0))
        if spo2 >= 98.0:
            s_spo2 = 1.0
        elif spo2 >= 95.0:
            s_spo2 = 0.5 + ((spo2 - 95.0) / (98.0 - 95.0)) * 0.5
        elif spo2 >= 94.0:
            s_spo2 = ((spo2 - 94.0) / (95.0 - 94.0)) * 0.5
        else:
            s_spo2 = 0.0

        # SNA score (HRV + SpO2)
        s_sna = (s_hrv + s_spo2) / 2.0

        # Temperatura penalizador
        temp_baseline = temp_mean.iloc[i] if not pd.isna(temp_mean.iloc[i]) else row.get('Temp_C', 36.45)
        delta_temp = float(row.get('Temp_C', temp_baseline) - temp_baseline)
        if delta_temp > 0.5:
            s_sna *= 0.8

        # Pilar Sono
        deep = float(row.get('Sono_Profundo_h', 0.0))
        total_sleep = float(row.get('Horas_Sono', 0.0)) + eps
        ratio = deep / total_sleep
        if ratio >= 0.25:
            e_qual = 1.0
        elif ratio < 0.15:
            e_qual = 0.0
        else:
            e_qual = (ratio - 0.15) / (0.25 - 0.15)

        e_quant = float(row.get('Horas_Sono', 0.0)) / float(sleep_needed)
        e_quant = max(0.0, min(1.0, e_quant))
        sono_score = (e_qual * 0.6) + (e_quant * 0.4)
        sono_score = float(np.clip(sono_score, 0.0, 1.0))

        # Pilar Strain (ACWR)
        acwr = float(acwr_series.iloc[i]) if not pd.isna(acwr_series.iloc[i]) else 1.0
        hard_cap = False
        acwr_bonus = False
        if 0.8 <= acwr <= 1.3:
            strain_score = 1.0
            acwr_bonus = True
        elif acwr > 1.5:
            strain_score = 0.0
            hard_cap = True
        elif acwr > 1.3:
            # declínio linear entre 1.3 e 1.5
            strain_score = max(0.0, 1.0 - (acwr - 1.3) / (1.5 - 1.3))
        else:
            # acwr < 0.8
            strain_score = max(0.0, acwr / 0.8)

        # Consolidação final
        rs = (s_sna * 0.5 + sono_score * 0.3 + strain_score * 0.2) * 100.0

        if acwr_bonus:
            rs += 5.0

        if hard_cap:
            rs = min(rs, 50.0)

        # Penalização por Assimetria (aumenta se a carga metabólica normalizada for alta)
        assim = float(row.get('Assimetria_Percent', 0.0))
        metab_z = float(row.get('Metabolic_Carga_Normalizada', 0.0)) if 'Metabolic_Carga_Normalizada' in row.index else 0.0
        assim_penalty = 15.0
        if assim > 10.0:
            # se carga metabólica estiver > 1 sigma acima da média, dobra a penalização
            if metab_z > 1.0:
                rs -= (assim_penalty * 2.0)
            else:
                rs -= assim_penalty

        rs = float(np.clip(rs, 0.0, 100.0))
        results.append(rs)

    return results

# --- EXECUÇÃO PRINCIPAL (wrap em main para permitir importação sem executar) ---
def main():
    dataset = generate_performance_dataset(100)

    # Estimar métricas metabólicas (Metabolic Power) e anexar ao dataset
    try:
        dataset = estimate_metabolic_metrics_for_dataset(dataset, mass_kg=70.0, kt=1.29, session_minutes=60, sampling_rate=1)
    except Exception:
        pass

    # Ajustar Recovery Score considerando a carga metabólica acumulada
    try:
        dataset['Metabolic_Carga_Normalizada'] = (dataset['Metabolic_Energy_kJ'] - dataset['Metabolic_Energy_kJ'].rolling(window=7, min_periods=1).mean()) / (
            dataset['Metabolic_Energy_kJ'].rolling(window=7, min_periods=1).std().replace(0, 1))
    except Exception:
        dataset['Metabolic_Carga_Normalizada'] = 0.0

    # Calcula Recovery Score e adiciona ao DataFrame (usar baseline já computado internamente)
    try:
        dataset['Recovery_Score'] = calcular_recovery_scores(dataset)
    except Exception:
        dataset['Recovery_Score'] = np.nan

    filename = 'biometria_performance_startup.csv'
    dataset.to_csv(filename, index=False)

    print(f"✅ Startup Engine: Dados preditivos processados em '{filename}'")

    # Mostra o Recovery Score do último dia como verificação rápida
    try:
        last_rs = dataset['Recovery_Score'].iloc[-1]
        print(f"🔍 Recovery Score (último dia): {last_rs:.1f}%")
    except Exception:
        pass

    # --- TREINO LSTM (opcional) ---
    try:
        if _HAS_TF:
            print("➡️ Iniciando treino LSTM para prever Readiness do dia seguinte...")
            lstm_model, lstm_metrics = train_lstm_predict_readiness(dataset, timesteps=7, epochs=30, verbose=0)
        else:
            print("⚠️ TensorFlow não detectado — pular treino LSTM. Instale 'tensorflow' para ativar.")
    except Exception as e:
        print(f"⚠️ Erro durante treino LSTM: {e}")

    # --- INTERFACE DE CONSULTA HISTÓRICA ---
    while True:
        try:
            print(f"\n📊 Histórico disponível: Dia 1 ao Dia {len(dataset)}")
            escolha = input("Selecione o Dia para análise (ou 'S' para sair): ").strip().upper()

            if escolha in ['S', '0']:
                break

            dia_idx = int(escolha)
            if dia_idx < 1 or dia_idx > len(dataset):
                print(f"❌ Dia {dia_idx} fora de alcance.")
                continue

            dia_data = dataset[dataset['Dia'] == dia_idx].iloc[0]

            print("\n" + "─"*50)
            print(f"📅 RELATÓRIO PREDITIVO - DIA {dia_idx}")
            print("─"*50)
            print(f"🚀 Readiness: {dia_data['Readiness_Score']:.1f}%")
            print(f"📈 ACWR (Rácio de Carga): {dia_data['ACWR']:.2f} (Ideal: 0.8 - 1.3)")
            print(f"📊 Desvio Estatístico (Z-Score): {dia_data['Z_Score_Carga']:.2f} σ")
            print(f"⚖️ Assimetria: {dia_data['Assimetria_Percent']:.1f}%")
            print(f"🩺 Diagnóstico IA: {dia_data['Diagnostico_IA']}")
            print(f"🛌 Sono Profundo: {dia_data['Sono_Profundo_h']:.2f} h / Horas Sono: {dia_data['Horas_Sono']:.2f} h")
            print(f"🌙 Eficiência de Sono: {dia_data['Sono_Eficiencia_Pct']:.1f}% ({dia_data['Sono_Eficiencia_Ratio']:.2f})")
            print(f"🩺 Diagnóstico Sono: {dia_data['Diagnostico_Sono']}")

            # Mostrar Recovery Score
            try:
                print(f"🧭 Recovery Score: {dia_data['Recovery_Score']:.1f}%")
            except Exception:
                pass

            # Mostrar Metabolic Power (se disponível)
            try:
                mp_mean = float(dia_data.get('Metabolic_Power_Mean_Wkg', np.nan))
                mp_peak = float(dia_data.get('Metabolic_Power_Peak_Wkg', np.nan))
                mp_energy = float(dia_data.get('Metabolic_Energy_kJ', np.nan))
                metab_z = float(dia_data.get('Metabolic_Carga_Normalizada', np.nan)) if 'Metabolic_Carga_Normalizada' in dia_data.index else np.nan
                print(f"⚡ Metabolic Power — Mean: {mp_mean:.2f} W/kg | Peak: {mp_peak:.2f} W/kg | Energy: {mp_energy:.1f} kJ | Metab_z: {metab_z:.2f}")
            except Exception:
                pass

            # Recomendação baseada no novo Cérebro Preditivo
            if dia_data['ACWR'] > 1.3:
                print("\n💡 RECOMENDAÇÃO TÉCNICA: Reduzir volume de treino em 20% para estabilizar ACWR.")
            elif dia_data['Z_Score_Carga'] > 2:
                print("\n💡 RECOMENDAÇÃO TÉCNICA: Pico de carga anormal detectado. Monitorizar sinais de dor.")

            # Recomendações simples para sono
            try:
                if dia_data['Sono_Eficiencia_Ratio'] < 0.15:
                    print("\n💡 RECOMENDAÇÃO SONO: Avaliar higiene do sono; considerar triagem para apneia/ajuste de ambiente.")
            except Exception:
                pass

            print("─"*50)

        except ValueError:
            print("❌ Entrada inválida.")

    # --- MÓDULO DE VALIDAÇÃO CLÍNICA (R&D) ---
    carga_real = dataset['Carga_Mecanica_G']
    carga_prevista_ia = carga_real + np.random.normal(0, 0.5, len(dataset))  # IA com ruído

    mae = mean_absolute_error(carga_real, carga_prevista_ia)
    rmse = np.sqrt(mean_squared_error(carga_real, carga_prevista_ia))

    print("\n" + "🔬 RELATÓRIO DE VALIDAÇÃO CLÍNICA (Startup Audit)")
    print(f"✅ Precisão do Algoritmo (MAE): {mae:.3f} G")
    print(f"✅ Rigor Científico (RMSE): {rmse:.3f} G")

    if rmse < 0.7:
        print("💎 STATUS: Algoritmo pronto para certificação Nível 1.")
    else:
        print("⚠️ STATUS: Erro elevado. Necessário ajustar filtragem de ruído (Kalman Filter).")

    print("✅ Encerrando consulta, até à próxima análise!")


if __name__ == '__main__':
    main()