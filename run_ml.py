import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
try:
    import xgboost as xgb
    has_xgb = True
except ImportError:
    has_xgb = False

df = pd.read_csv('dataset_jimf_ready.csv')
target = 'Uang_Kartal_M1_miliar'
features = ['E_Money_volume_ribu', 'IHK', 'Suku_Bunga', 'IPR', 'Dummy_Ramadan', 'Dummy_IdulFitri']

train = df.iloc[:-12]
test = df.iloc[-12:]

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

if has_xgb:
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    name = "XGBoost"
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    name = "Random Forest"

model.fit(X_train, y_train)
preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
mape = np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100

print(f"{name} Results:")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")
