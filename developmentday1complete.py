import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.cluster import KMeans # O motor de IA para Clustering

print("--- ⚽ BME STARTUP AI: UNSUPERVISED CLUSTERING ENGINE ---")

# --- FUNÇÃO TÉCNICA: FILTRO DE BUTTERWORTH ---
def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# 1. LOAD DATA
try:
    df = pd.read_csv('iot_sports_dataset.csv') 
    print("SUCCESS: Real Dataset Loaded.")
except:
    print("NOTICE: Generating Multi-Athlete Dataset...")
    time = np.arange(0, 1000)
    ids = ['A001', 'A002', 'A003']
    data_list = []
    for athlete in ids:
        data_list.append(pd.DataFrame({
            'Timestamp': time, 'Speed': np.random.uniform(2, 32, 1000), 
            'Heart_Rate': np.random.uniform(110, 195, 1000), 'Athlete_ID': athlete,
            'Acc_X': np.random.normal(0, 0.5, 1000), 'Acc_Y': np.random.normal(0, 0.5, 1000),
            'Acc_Z': np.random.normal(9.8, 0.5, 1000), 'Jump_Height': np.random.uniform(25, 60, 1000)
        }))
    df = pd.concat(data_list)

# 2. SELECÇÃO
atleta_selecionado = input(f"Escolha o Atleta {sorted(df['Athlete_ID'].unique())}: ").strip()
df_final = df[df['Athlete_ID'] == atleta_selecionado].copy().reset_index(drop=True)

# 3. BIOMEDICAL SIGNAL PROCESSING
raw_eff = df_final['Speed'] / df_final['Heart_Rate']
df_final['eff_smooth'] = butter_lowpass_filter(raw_eff.fillna(raw_eff.mean()), 0.1, 10.0)
df_final['impact_mag'] = np.abs(np.sqrt(df_final['Acc_X']**2 + df_final['Acc_Y']**2 + df_final['Acc_Z']**2) - 9.8)
df_final['impact_smooth'] = df_final['impact_mag'].rolling(window=15).mean().fillna(0)
df_final['jump_trend'] = df_final['Jump_Height'].rolling(window=50).mean().fillna(df_final['Jump_Height'].mean())

# --- 4. MOTOR DE IA: K-MEANS CLUSTERING (A SOLUÇÃO DEFINITIVA) ---
# Ensinamos a IA a agrupar dados usando Velocidade e Impacto
X = df_final[['Speed', 'impact_smooth']].values
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_final['Cluster'] = kmeans.fit_predict(X)

# Mapeamos os clusters para nomes baseados na velocidade média de cada um
# O cluster com maior velocidade média é sempre 'Alta Intensidade'
cluster_speeds = df_final.groupby('Cluster')['Speed'].mean().sort_values()
mapping = {
    cluster_speeds.index[0]: 'Recuperação/Parado',
    cluster_speeds.index[1]: 'Moderado (Trote)',
    cluster_speeds.index[2]: 'Alta Intensidade (Sprint/Duelo)'
}
df_final['Estado'] = df_final['Cluster'].map(mapping)

# 5. READINESS SCORE ADAPTATIVO
tempo_alta = (df_final['Estado'] == 'Alta Intensidade (Sprint/Duelo)').mean()
curr_eff = df_final['eff_smooth'].iloc[-1]
avg_eff = df_final['eff_smooth'].mean()
last_jump = df_final['jump_trend'].iloc[-1]
avg_jump = df_final['jump_trend'].mean()

readiness = ( (curr_eff/avg_eff)*50 + (last_jump/avg_jump)*50 ) - (tempo_alta * 100)
readiness = max(5, min(100, readiness))

print(f"\n⭐ IA CLUSTERING READINESS: {readiness:.1f}%")

# 6. VISUALIZAÇÃO
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
colors = {'Recuperação/Parado': '#95a5a6', 'Moderado (Trote)': '#2ecc71', 'Alta Intensidade (Sprint/Duelo)': '#e74c3c'}

for estado, cor in colors.items():
    mask = df_final['Estado'] == estado
    ax1.scatter(df_final.index[mask], df_final['eff_smooth'][mask], c=cor, s=12, label=estado)

ax1.set_title(f"IA Clustering Dashboard: {atleta_selecionado} (Readiness: {readiness:.1f}%)")
ax1.legend(loc='upper right')
ax2.plot(df_final['impact_smooth'], color='orange', label='Carga Mecânica')
ax3.plot(df_final['jump_trend'], color='purple', linewidth=3, label='Capacidade Explosiva')

plt.tight_layout()
plt.show()