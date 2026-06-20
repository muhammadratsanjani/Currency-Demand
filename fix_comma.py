import pandas as pd

df = pd.read_csv('dataset_jimf_ready.csv')
for col in df.columns:
    if col not in ['Periode', 'Dummy_Ramadan', 'Dummy_IdulFitri']:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(',', '.').astype(float)

df.to_csv('dataset_jimf_ready.csv', index=False)
print("Berhasil memperbaiki format desimal koma ke titik.")
