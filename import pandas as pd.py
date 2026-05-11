import pandas as pd
import numpy as np

def gerar_dataset_avancado_bme(filename='biometria_performance_startup_v3.csv'):
    """
    Gera um dataset biomédico completo integrando métricas de Performance (GPS)
    com métricas de Saúde (Wearable), estilo WHOOP/Catapult.
    """
    np.random.seed(42)
    samples = 100
    dias = range(1, samples + 1)

    # 1. MÉTRICAS DE SAÚDE (PULSO - Estilo WHOOP)
    hrv = np.random.normal(70, 15, samples)
    horas_sono = np.random.uniform(6.0, 9.5, samples).round(1)
    deep_sleep = np.random.uniform(1.5, 4.5, samples)
    
    # NOVAS MÉTRICAS DE SAÚDE [CITE: 1, 4]
    spo2 = np.random.uniform(94, 99, samples).round(1)        # Oxigénio no sangue
    temp_corporal = np.random.uniform(36.2, 37.5, samples).round(1) # Temperatura overnight
    calorias_queimadas = np.random.uniform(2200, 4500, samples).round(0) # Gasto diário

    # 2. MÉTRICAS DE PERFORMANCE (GPS/DORSO)
    distancia = np.random.uniform(4.0, 12.0, samples)
    vel_max = np.random.uniform(25.0, 36.0, samples)
    minutos_alta_intensidade = np.random.uniform(15, 60, samples).round(0)
    tempo_recuperacao_hrv = np.random.uniform(12, 48, samples).round(1)
    carga_mecanica = distancia * np.random.uniform(1.1, 1.9, samples)
    assimetria = np.random.uniform(1, 12, samples)

    # 3. CONSTRUÇÃO DO DATAFRAME
    df = pd.DataFrame({
        'Dia': dias,
        'HRV_ms': hrv,
        'Horas_Sono': horas_sono,
        'Sono_Profundo_h': deep_sleep,
        'SpO2_Percent': spo2,
        'Temp_Corporal_C': temp_corporal,
        'Calorias_Burned': calorias_queimadas,
        'Tempo_Recuperacao_HRV': tempo_recuperacao_hrv,
        'Distancia_km': distancia,
        'Minutos_Alta_Intensidade': minutos_alta_intensidade,
        'Velocidade_Max_kmh': vel_max,
        'Carga_Mecanica_G': carga_mecanica,
        'Assimetria_Percent': assimetria
    })

    # 4. INTELIGÊNCIA DERIVADA
    # Readiness Score atualizado para incluir SpO2 (simplificado)
    df['Readiness_Score'] = ((df['Sono_Profundo_h'] * 10) + (df['HRV_ms'] * 0.4) + (df['SpO2_Percent'] * 0.5)).clip(0, 100)
    
    # ACWR (Acute:Chronic Workload Ratio)
    df['Carga_Aguda'] = df['Carga_Mecanica_G'].rolling(window=7, min_periods=1).mean()
    df['Carga_Cronica'] = df['Carga_Mecanica_G'].rolling(window=28, min_periods=1).mean()
    df['ACWR'] = (df['Carga_Aguda'] / df['Carga_Cronica']).fillna(1.0)

    # Diagnóstico IA
    df['Diagnostico_IA'] = df.apply(
        lambda x: "⚠️ PERIGO: Fadiga Crítica" if x['ACWR'] > 1.5 or x['Temp_Corporal_C'] > 37.4 else "🟢 OK: Apto", 
        axis=1
    )

    # 5. EXPORTAÇÃO
    df.to_csv(filename, index=False)
    print(f"✅ Dataset V3 gerado: '{filename}' com Calorias, SpO2 e Temperatura.")

if __name__ == "__main__":
    gerar_dataset_avancado_bme()