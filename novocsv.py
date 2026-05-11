import pandas as pd
import numpy as np

def gerar_dataset_startup_bme(filename='biometria_performance_startup_2_2.csv'):
    """
    Gera um dataset sintético de alta fidelidade para treino de modelos de IA
    em performance desportiva e engenharia biomédica.
    """
    np.random.seed(42)  # Semente fixa para consistência de dados
    samples = 100
    
    # 1. GERAÇÃO DE BIOMETRIA BASE (Sensores)
    dias = range(1, samples + 1)
    hrv = np.random.normal(70, 15, samples)           # Variabilidade Cardíaca (ms)
    sono_profundo = np.random.uniform(1.5, 4.5, samples)  # Janela de reparação tecidual
    horas_sono_total = np.random.uniform(6.0, 9.5, samples).round(1)
    distancia = np.random.uniform(4.0, 12.0, samples)  # Volume de treino (km)
    vel_max = np.random.uniform(25.0, 36.0, samples)   # Output de potência
    assimetria = np.random.uniform(1, 12, samples)     # % Diferença de carga bilateral
    minutos_alta_intensidade = np.random.uniform(15, 60, samples).round(0)
    
    # NOVA VARIÁVEL ESTRATÉGICA: Tempo de Recuperação do HRV (em horas)
    # Fundamental para o cálculo de Eficiência de Resistência
    tempo_recuperacao_hrv = np.random.uniform(12, 48, samples).round(1)
    
    # Carga Mecânica (G) - Simulação de impacto captado por acelerómetros
    carga_mecanica = distancia * np.random.uniform(1.1, 1.9, samples)

    # 2. CONSTRUÇÃO DO DATAFRAME
    df = pd.DataFrame({
        'Dia': dias,
        'HRV_ms': hrv,
        'Horas_Sono': horas_sono_total,
        'Sono_Profundo_h': sono_profundo,
        'Tempo_Recuperacao_HRV': tempo_recuperacao_hrv,
        'Distancia_km': distancia,
        'Minutos_Alta_Intensidade': minutos_alta_intensidade,
        'Velocidade_Max_kmh': vel_max,
        'Carga_Mecanica_G': carga_mecanica,
        'Assimetria_Percent': assimetria
    })

    # 3. ENGINE DE CÁLCULO IA (V0.1)
    # Readiness: Balanço entre recuperação biológica e estado autonómico
    df['Readiness'] = ((df['Sono_Profundo_h'] * 15) + (df['HRV_ms'] * 0.5)).clip(0, 100)
    
    # Eficiência: Relação entre output mecânico e custo cardíaco (Regularizado)
    df['Eficiencia'] = (df['Distancia_km'] / (abs(100 - df['HRV_ms']) + 1)) * 50
    
    # Algoritmo ACWR: Acute:Chronic Workload Ratio para prevenção de lesões
    df['Carga_Aguda'] = df['Carga_Mecanica_G'].rolling(window=7, min_periods=1).mean()
    df['Carga_Cronica'] = df['Carga_Mecanica_G'].rolling(window=28, min_periods=1).mean()
    df['ACWR'] = (df['Carga_Aguda'] / df['Carga_Cronica']).fillna(1.0)

    # Diagnóstico Base para Dashboard
    df['Diagnostico_IA'] = df.apply(
        lambda x: "⚠️ PERIGO: Fadiga Crítica" if x['ACWR'] > 1.5 else "🟢 OK: Atleta Apto", 
        axis=1
    )

    # 4. EXPORTAÇÃO FINAL
    df.to_csv(filename, index=False)
    print(f"✅ Sucesso! Ficheiro '{filename}' gerado com todas as variáveis biomédicas.")

if __name__ == "__main__":
    gerar_dataset_startup_bme()