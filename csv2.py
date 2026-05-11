import pandas as pd
import numpy as np

def generate_biomedical_dataset(samples=100, filename='biometria_performance_startup_2.csv'):
    np.random.seed(42) # Para que os resultados sejam consistentes
    
    # 1. GERAÇÃO DE VARIÁVEIS BASE (Sensores)
    dias = range(1, samples + 1)
    hrv = np.random.normal(70, 15, samples)          # Variabilidade Cardíaca (ms)
    sono_profundo = np.random.uniform(1.5, 4.5, samples) # Horas de sono profundo
    horas_sono_total = np.random.uniform(6.0, 9.5, samples) # NOVA VARIÁVEL
    distancia = np.random.uniform(4.0, 12.0, samples) # Distância (km)
    vel_max = np.random.uniform(25.0, 36.0, samples)  # Velocidade Máxima
    assimetria = np.random.uniform(1, 12, samples)    # % de diferença entre pernas
    
    # Carga Mecânica (G) baseada na distância com ruído biomecânico
    carga_mecanica = distancia * np.random.uniform(1.1, 1.9, samples)

    # 2. CRIAÇÃO DO DATAFRAME
    df = pd.DataFrame({
        'Dia': dias,
        'HRV_ms': hrv,
        'Horas_Sono': horas_sono_total.round(1),
        'Sono_Profundo_h': sono_profundo,
        'Distancia_km': distancia,
        'Velocidade_Max_kmh': vel_max,
        'Carga_Mecanica_G': carga_mecanica,
        'Assimetria_Percent': assimetria
    })

    # 3. MOTOR DE IA (Cálculos Derivados)
    # Readiness Score (Prontidão)
    df['Readiness'] = ((df['Sono_Profundo_h'] * 15) + (df['HRV_ms'] * 0.5)).clip(0, 100)
    
    # Eficiência com REGULARIZAÇÃO (+1 no denominador para evitar divisão por zero/negativos)
    df['Eficiencia'] = (df['Distancia_km'] / (abs(100 - df['HRV_ms']) + 1)) * 50
    
    # Algoritmo ACWR (Acute:Chronic Workload Ratio)
    df['Carga_Aguda'] = df['Carga_Mecanica_G'].rolling(window=7, min_periods=1).mean()
    df['Carga_Cronica'] = df['Carga_Mecanica_G'].rolling(window=28, min_periods=1).mean()
    df['ACWR'] = (df['Carga_Aguda'] / df['Carga_Cronica']).fillna(1.0)

    # Diagnóstico Clínico
    def get_status(row):
        if row['ACWR'] > 1.5: return "⚠️ PERIGO: Fadiga Crítica"
        if row['Assimetria_Percent'] > 10: return "⚠️ AVISO: Risco Biomecânico"
        return "🟢 OK: Apto"

    df['Diagnostico_IA'] = df.apply(get_status, axis=1)

    # 4. EXPORTAÇÃO
    df.to_csv(filename, index=False)
    print(f"✅ Novo dataset '{filename}' gerado com sucesso!")
    return df

# Executar a criação
novo_df = generate_biomedical_dataset()