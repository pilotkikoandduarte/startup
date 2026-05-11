import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

print("--- ⚽ FOOTBALL PERFORMANCE AI: TRIPLE-THREAT DASHBOARD ---")

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
    print("NOTICE: CSV not found. Generating Full Synthetic Dataset...")
    time = np.arange(0, 1000)
    # Geramos dados para 3 atletas diferentes para testares a escolha
    ids = ['A001', 'A002', 'A003']
    data_list = []
    for athlete in ids:
        temp_df = pd.DataFrame({
            'Timestamp': time, 
            'Speed': np.random.uniform(5, 28, 1000), 
            'Heart_Rate': np.random.uniform(120, 190, 1000), 
            'Athlete_ID': athlete,
            'Acc_X': np.random.normal(0, 0.5, 1000),
            'Acc_Y': np.random.normal(0, 0.5, 1000),
            'Acc_Z': np.random.normal(9.8, 0.5, 1000),
            'Jump_Height': np.random.uniform(30, 55, 1000)
        })
        data_list.append(temp_df)
    df = pd.concat(data_list)

# 2. SELECÇÃO INTERATIVA DO ATLETA
print("\n" + "="*30)
lista_atletas = sorted(df['Athlete_ID'].unique())
print(f"ATLETAS DISPONÍVEIS NO SISTEMA: {lista_atletas}")
print("="*30)

while True:
    atleta_selecionado = input("Digite o ID do atleta que deseja analisar: ").strip()
    if atleta_selecionado in lista_atletas:
        print(f"\n✅ A carregar dados de {atleta_selecionado}...")
        break
    else:
        print(f"❌ Erro: '{atleta_selecionado}' não encontrado. Tente novamente (Ex: {lista_atletas[0]})")

df_final = df[df['Athlete_ID'] == atleta_selecionado].copy().reset_index(drop=True)

# 3. BIOMEDICAL LOGIC
# A) Eficiência (Coração)
raw_eff = df_final['Speed'] / df_final['Heart_Rate']
df_final['eff_smooth'] = butter_lowpass_filter(raw_eff.fillna(raw_eff.mean()), 0.1, 10.0)

# B) Impactos (Acelerómetro)
df_final['impact_mag'] = np.abs(np.sqrt(df_final['Acc_X']**2 + df_final['Acc_Y']**2 + df_final['Acc_Z']**2) - 9.8)
df_final['impact_smooth'] = df_final['impact_mag'].rolling(window=15).mean()

# C) Saltos
df_final['jump_trend'] = df_final['Jump_Height'].rolling(window=50).mean()

# 4. AI DECISION AGENT
avg_jump = df_final['Jump_Height'].mean()
last_jump_trend = df_final['jump_trend'].iloc[-1]
avg_eff = df_final['eff_smooth'].mean()
curr_eff = df_final['eff_smooth'].iloc[-1]

print("\n--- AI AGENT MULTI-REPORT ---")
neuromuscular = "🔴 ALERTA: Fadiga Central." if last_jump_trend < (avg_jump * 0.90) else "🟢 ESTÁVEL: Neuromuscular OK."
metabolic = "🔴 ALERTA: Fadiga Cardiovascular." if curr_eff < (avg_eff * 0.85) else "🟢 ESTÁVEL: Cardiovascular OK."

print(neuromuscular)
print(metabolic)

# 5. VISUALIZAÇÃO TRIPLA
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

ax1.plot(df_final['eff_smooth'], color='blue', linewidth=2, label='Eficiência (Coração)')
ax1.axhline(y=avg_eff, color='blue', linestyle='--', alpha=0.3)
ax1.set_ylabel("Eficiência (Vel/HR)")
ax1.set_title(f"Monitorização de Performance 360º: Atleta {atleta_selecionado}")
ax1.legend(loc='upper right')

ax2.fill_between(range(len(df_final)), df_final['impact_smooth'], color='orange', alpha=0.3)
ax2.plot(df_final['impact_smooth'], color='darkorange', label='Carga Mecânica (Impacto)')
ax2.set_ylabel("Impacto (G)")
ax2.legend(loc='upper right')

ax3.scatter(df_final.index, df_final['Jump_Height'], color='gray', s=5, alpha=0.2, label='Saltos Individuais')
ax3.plot(df_final['jump_trend'], color='purple', linewidth=3, label='Tendência de Salto')
ax3.axhline(y=avg_jump, color='purple', linestyle='--', alpha=0.3)
ax3.set_ylabel("Altura do Salto (cm)")
ax3.set_xlabel("Tempo (Amostras)")
ax3.legend(loc='upper right')

plt.tight_layout()
plt.show()