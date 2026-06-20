import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load data
df = pd.read_csv('dataset_jimf_ready.csv')
df['Periode'] = pd.to_datetime(df['Periode'])
df = df.sort_values('Periode').reset_index(drop=True)

y = df['Uang_Kartal_M1_miliar'].values
train_y = y[:-12]
test_y = y[-12:]

# Simple ARIMA (no seasonality) to avoid explosion
model = SARIMAX(train_y, order=(1, 1, 1))
results = model.fit(disp=False)
preds = results.forecast(steps=12)

# Calculate RMSE and MAPE manually to avoid sklearn weirdness
rmse = np.sqrt(np.mean((test_y - preds)**2))
# avoid zero division
mape = np.mean(np.abs((test_y - preds) / (test_y + 1e-9))) * 100

print(f'SARIMA RMSE: {rmse:.2f}')
print(f'SARIMA MAPE: {mape:.2f}%')
