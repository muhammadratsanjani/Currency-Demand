import pandas as pd
from arch.unitroot import ZivotAndrews

df = pd.read_csv('dataset_jimf_ready.csv')
cpi = df['IHK'].dropna()

za_level = ZivotAndrews(cpi, trend='ct')
print("--- Zivot-Andrews Test on CPI (IHK) Level (trend='ct') ---")
print(za_level.summary())

cpi_diff = cpi.diff().dropna()
za_diff = ZivotAndrews(cpi_diff, trend='c')
print("\n--- Zivot-Andrews Test on CPI (IHK) 1st Diff (trend='c') ---")
print(za_diff.summary())
