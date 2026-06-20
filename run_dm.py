import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.ardl import ARDL
import tensorflow as tf
import scipy.stats
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

def dm_test(actual, pred1, pred2, h=1):
    e1 = (actual - pred1)**2
    e2 = (actual - pred2)**2
    d = e1 - e2
    mean_d = np.mean(d)
    
    def autocovariance(Xi, N, k):
        autoCov = 0
        mean_Xi = np.mean(Xi)
        for i in np.arange(0, N-k):
            autoCov += ((Xi[i] - mean_Xi) * (Xi[i+k] - mean_Xi))
        return (1/N) * autoCov
        
    gamma = []
    for lag in range(0, h):
        gamma.append(autocovariance(d, len(d), lag))
        
    V_d = gamma[0] + 2 * sum(gamma[1:])
    
    if V_d <= 0:
        return 0, 1.0
        
    DM_stat = V_d**(-0.5) * mean_d * np.sqrt(len(d))
    
    # Harvey-Leybourne-Newbold (HLN) small-sample correction
    n = len(d)
    hln_factor = np.sqrt((n + 1 - 2*h + (h/n)*(h-1)) / n)
    DM_stat_hln = DM_stat * hln_factor
    
    # Use Student's t-distribution with n-1 degrees of freedom
    p_value_hln = 2 * (1 - scipy.stats.t.cdf(abs(DM_stat_hln), df=n-1))
    
    return DM_stat_hln, p_value_hln

# Load data
df = pd.read_csv('dataset_jimf_ready.csv')
df = df[df['Uang_Kartal_M1_miliar'] > 0].copy()
df.drop_duplicates(subset=['Periode'], keep='first', inplace=True)
df['Periode'] = pd.to_datetime(df['Periode'])
df = df.sort_values('Periode').reset_index(drop=True)

test_size = 12
y = df['Uang_Kartal_M1_miliar'].values
actual_y = y[-test_size:]

# 1. Prophet
prophet_df = df[['Periode', 'Uang_Kartal_M1_miliar', 'E_Money_volume_ribu', 'E_Money_nilai_miliar', 'IHK', 'Suku_Bunga', 'IPR', 'Dummy_Ramadan', 'Dummy_IdulFitri']].copy()
prophet_df.columns = ['ds', 'y', 'E_Money_volume_ribu', 'E_Money_nilai_miliar', 'IHK', 'Suku_Bunga', 'IPR', 'Dummy_Ramadan', 'Dummy_IdulFitri']
m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
for reg in ['E_Money_volume_ribu', 'E_Money_nilai_miliar', 'IHK', 'Suku_Bunga', 'IPR', 'Dummy_Ramadan', 'Dummy_IdulFitri']:
    m.add_regressor(reg)
m.fit(prophet_df.iloc[:-test_size])
forecast = m.predict(prophet_df.drop(columns='y'))
prophet_yhat = forecast['yhat'].values[-test_size:]
prophet_df['yhat'] = forecast['yhat']
prophet_df['residual'] = prophet_df['y'] - prophet_df['yhat']

# 2. LSTM
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_residuals = scaler.fit_transform(prophet_df[['residual']].values)
seq_length = 3
X, y_res = [], []
for i in range(len(scaled_residuals) - seq_length):
    X.append(scaled_residuals[i:(i + seq_length)])
    y_res.append(scaled_residuals[i + seq_length])
X = np.array(X)
y_res = np.array(y_res)
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y_res[:-test_size], y_res[-test_size:]

model = Sequential([
    LSTM(150, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.4),
    LSTM(150, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
model.fit(X_train, y_train, epochs=200, batch_size=4, verbose=0)
lstm_pred_scaled = model.predict(X_test, verbose=0)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
hybrid_yhat = prophet_yhat + lstm_pred.flatten()

# 3. ARDL
exog_cols = ['E_Money_volume_ribu', 'IHK', 'Suku_Bunga', 'IPR']
train_y_ardl = y[:-test_size]
train_X_ardl = df[exog_cols].iloc[:-test_size]
test_X_ardl = df[exog_cols].iloc[-test_size:]
ardl_model = ARDL(train_y_ardl, 2, train_X_ardl, {"E_Money_volume_ribu": 1, "IHK": 1, "Suku_Bunga": 1, "IPR": 1})
ardl_res = ardl_model.fit()
ardl_preds = ardl_res.forecast(steps=test_size, exog=test_X_ardl)

print("--- DIEBOLD-MARIANO TEST RESULTS ---")
dm_prophet, p_prophet = dm_test(actual_y, prophet_yhat, hybrid_yhat)
print(f"Hybrid vs Prophet -> DM Stat: {dm_prophet:.4f}, p-value: {p_prophet:.4f}")

dm_ardl, p_ardl = dm_test(actual_y, ardl_preds, hybrid_yhat)
print(f"Hybrid vs ARDL    -> DM Stat: {dm_ardl:.4f}, p-value: {p_ardl:.4f}")
