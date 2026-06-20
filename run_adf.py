import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('dataset_jimf_ready.csv')
df['Periode'] = pd.to_datetime(df['Periode'])
df = df.sort_values('Periode').reset_index(drop=True)

y = df['Uang_Kartal_M1_miliar'].values

print("--- ADF Test on Raw M1 (Level) ---")
result_raw = adfuller(y)
print(f"ADF Statistic: {result_raw[0]:.4f}")
print(f"p-value: {result_raw[1]:.4f}")
if result_raw[1] > 0.05:
    print("Result: Non-Stationary")
else:
    print("Result: Stationary")

print("\n--- Fitting Prophet ---")
# Prepare data for Prophet
prophet_df = df[['Periode', 'Uang_Kartal_M1_miliar', 'E_Money_volume_ribu', 'IHK', 'Suku_Bunga', 'IPR', 'Dummy_Ramadan', 'Dummy_IdulFitri']].copy()
prophet_df.columns = ['ds', 'y', 'emoney', 'ihk', 'sb', 'ipr', 'ramadan', 'idulfitri']

m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m.add_regressor('emoney')
m.add_regressor('ihk')
m.add_regressor('sb')
m.add_regressor('ipr')
m.add_regressor('ramadan')
m.add_regressor('idulfitri')

m.fit(prophet_df)
forecast = m.predict(prophet_df)

# Calculate residuals
residuals = prophet_df['y'] - forecast['yhat']

print("\n--- ADF Test on Prophet Residuals ---")
result_res = adfuller(residuals.dropna())
print(f"ADF Statistic: {result_res[0]:.4f}")
print(f"p-value: {result_res[1]:.4f}")
if result_res[1] > 0.05:
    print("Result: Non-Stationary")
else:
    print("Result: Stationary")
