import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from statsmodels.tsa.ardl import ARDL

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('dataset_jimf_ready.csv')
df['Periode'] = pd.to_datetime(df['Periode'])
df = df.sort_values('Periode').reset_index(drop=True)

target_col = 'Uang_Kartal_M1_miliar'
exog_cols = ['E_Money_volume_ribu', 'IHK', 'Suku_Bunga', 'IPR']

y = df[target_col]
X = df[exog_cols]

train_y = y[:-12]
test_y = y[-12:]
train_X = X[:-12]
test_X = X[-12:]

def get_metrics(test, preds, name):
    test_vals = np.array(test)
    pred_vals = np.array(preds)
    mask = test_vals != 0
    test_vals = test_vals[mask]
    pred_vals = pred_vals[mask]
    rmse = np.sqrt(np.mean((test_vals - pred_vals)**2))
    mape = np.mean(np.abs((test_vals - pred_vals) / test_vals)) * 100
    print(f'{name} RMSE: {rmse:.2f}')
    print(f'{name} MAPE: {mape:.2f}%')
    return rmse, mape

print("--- Computing Econometric Baselines ---")

# 1. ARIMAX (using SARIMAX without seasonal order)
try:
    arimax_model = SARIMAX(train_y, exog=train_X, order=(1, 1, 1))
    arimax_res = arimax_model.fit(disp=False)
    arimax_preds = arimax_res.forecast(steps=12, exog=test_X)
    get_metrics(test_y, arimax_preds, "ARIMAX")
except Exception as e:
    print(f"ARIMAX Failed: {e}")

# 2. VECM (Vector Error Correction Model)
vecm_data = df[[target_col] + exog_cols]
vecm_train = vecm_data[:-12]
vecm_test = vecm_data[-12:]
try:
    lag_order = select_order(vecm_train, maxlags=5, deterministic="ci")
    k_ar_diff = lag_order.aic
    rank = select_coint_rank(vecm_train, det_order=0, k_ar_diff=k_ar_diff, signif=0.05).rank
    vecm_model = VECM(vecm_train, deterministic="ci", k_ar_diff=max(1, k_ar_diff), coint_rank=max(1, rank))
    vecm_res = vecm_model.fit()
    vecm_preds = vecm_res.predict(steps=12)
    vecm_y_preds = vecm_preds[:, 0]
    get_metrics(test_y, vecm_y_preds, "VECM")
except Exception as e:
    print(f"VECM Failed: {e}")

# 3. ARDL (Autoregressive Distributed Lag)
try:
    ardl_model = ARDL(train_y, 2, train_X, {"E_Money_volume_ribu": 1, "IHK": 1, "Suku_Bunga": 1, "IPR": 1})
    ardl_res = ardl_model.fit()
    ardl_preds = ardl_res.forecast(steps=12, exog=test_X)
    get_metrics(test_y, ardl_preds, "ARDL")
except Exception as e:
    print(f"ARDL Failed: {e}")

print("--- Done ---")
