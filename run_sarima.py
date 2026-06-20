import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load data
df = pd.read_csv('dataset_jimf_ready.csv')
# Sort by Periode just in case
df['Periode'] = pd.to_datetime(df['Periode'])
df = df.sort_values('Periode').reset_index(drop=True)

# Target
y = df['Uang_Kartal_M1_miliar'].values

# Train-test split (last 12 months is test)
train_y = y[:-12]
test_y = y[-12:]

print(f'Train size: {len(train_y)}, Test size: {len(test_y)}')

# Fit SARIMA. Data is monthly, so seasonal period is 12.
# A standard baseline is SARIMA(1,1,1)(0,1,1,12) (airline model)
model = SARIMAX(train_y, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)

# Predict
preds = results.forecast(steps=12)

# Metrics
rmse = np.sqrt(mean_squared_error(test_y, preds))
mape = mean_absolute_percentage_error(test_y, preds) * 100

print(f'SARIMA RMSE: {rmse:.2f}')
print(f'SARIMA MAPE: {mape:.2f}%')
