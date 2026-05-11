import pandas as pd

# Carrega o dataset original
df = pd.read_csv('wesad_100percent.csv')

# Extrai 1% de forma aleatória
df_sample = df.sample(frac=0.01, random_state=42)

# Guarda