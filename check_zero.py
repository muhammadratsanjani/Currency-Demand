import pandas as pd
df = pd.read_csv('dataset_jimf_ready.csv')
print('Jumlah nilai 0 pada M1:', (df['Uang_Kartal_M1_miliar'] == 0).sum())
print('Baris terakhir M1:')
print(df[['Periode', 'Uang_Kartal_M1_miliar']].tail(15))
