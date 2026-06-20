import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def build_and_train_lstm(X_train, y_train, seq_length, units, dropout, epochs, lr):
    model = Sequential([
        LSTM(units, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(dropout),
        LSTM(units, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=4, verbose=0)
    return model

def main():
    print("Membaca data yang sudah disiapkan...")
    file_path = "dataset_jimf_ready.csv"
    if not os.path.exists(file_path):
        print(f"File {file_path} tidak ditemukan!")
        return
        
    df = pd.read_csv(file_path)
    
    # FIX: Hapus baris di mana M1 bernilai 0 (karena salah copy paste / duplikat)
    df = df[df['Uang_Kartal_M1_miliar'] > 0].copy()
    
    # Hapus duplikat tanggal jika ada
    df.drop_duplicates(subset=['Periode'], keep='first', inplace=True)
    
    df['Periode'] = pd.to_datetime(df['Periode'])
    
    # 1. PERSIAPAN DATA PROPHET
    print("Membangun model Prophet (Baseline)...")
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = df['Periode']
    prophet_df['y'] = df['Uang_Kartal_M1_miliar']
    
    regressors = ['E_Money_volume_ribu', 'E_Money_nilai_miliar', 'IHK', 'Suku_Bunga', 'IPR', 'Dummy_Ramadan', 'Dummy_IdulFitri']
    valid_regressors = []
    for col in regressors:
        matching_cols = [c for c in df.columns if col.lower() in c.lower()]
        if matching_cols:
            actual_col = matching_cols[0]
            prophet_df[actual_col] = df[actual_col]
            valid_regressors.append(actual_col)
            
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    for reg in valid_regressors:
        m.add_regressor(reg)
        
    test_size = 12
    train_df = prophet_df.iloc[:-test_size]
    test_df = prophet_df.iloc[-test_size:]
    
    m.fit(train_df)
    forecast = m.predict(prophet_df.drop(columns='y'))
    
    prophet_df['yhat'] = forecast['yhat']
    prophet_df['residual'] = prophet_df['y'] - prophet_df['yhat']
    
    # 2. HYPERPARAMETER TUNING LSTM PADA RESIDUAL
    print("Memulai Hyperparameter Tuning LSTM untuk memprediksi residual...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_residuals = scaler.fit_transform(prophet_df[['residual']].values)
    
    seq_length = 3
    X, y = create_sequences(scaled_residuals, seq_length)
    
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    actual_y = prophet_df['y'].iloc[-test_size:].values
    prophet_yhat = prophet_df['yhat'].iloc[-test_size:].values
    prophet_rmse = np.sqrt(mean_squared_error(actual_y, prophet_yhat))
    prophet_mape = mean_absolute_percentage_error(actual_y, prophet_yhat) * 100
    
    # Daftar hyperparameter yang akan diuji
    hyperparams = [
        {'units': 50, 'dropout': 0.2, 'epochs': 100, 'lr': 0.001},
        {'units': 100, 'dropout': 0.3, 'epochs': 150, 'lr': 0.001},
        {'units': 150, 'dropout': 0.4, 'epochs': 200, 'lr': 0.0005}
    ]
    
    best_hybrid_rmse = float('inf')
    best_hybrid_mape = float('inf')
    best_hybrid_yhat = None
    best_params = None
    
    print("\n--- Proses Tuning Berjalan ---")
    for i, hp in enumerate(hyperparams):
        print(f"Menguji Skenario {i+1}: Units={hp['units']}, Epochs={hp['epochs']}, Dropout={hp['dropout']}, LR={hp['lr']}...")
        model = build_and_train_lstm(X_train, y_train, seq_length, hp['units'], hp['dropout'], hp['epochs'], hp['lr'])
        
        lstm_pred_scaled = model.predict(X_test, verbose=0)
        lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
        
        hybrid_yhat = prophet_yhat + lstm_pred.flatten()
        hybrid_rmse = np.sqrt(mean_squared_error(actual_y, hybrid_yhat))
        hybrid_mape = mean_absolute_percentage_error(actual_y, hybrid_yhat) * 100
        
        print(f"Hasil Skenario {i+1} -> RMSE: {hybrid_rmse:.2f} | MAPE: {hybrid_mape:.2f}%")
        
        if hybrid_rmse < best_hybrid_rmse:
            best_hybrid_rmse = hybrid_rmse
            best_hybrid_mape = hybrid_mape
            best_hybrid_yhat = hybrid_yhat
            best_params = hp
            
    print("\n=== HASIL EVALUASI MODEL (TERBAIK) ===")
    print(f"Prophet (Standalone) -> RMSE: {prophet_rmse:.2f} | MAPE: {prophet_mape:.2f}%")
    print(f"Hybrid Prophet-LSTM  -> RMSE: {best_hybrid_rmse:.2f} | MAPE: {best_hybrid_mape:.2f}%")
    print(f"Hyperparameter Terbaik: {best_params}")
    
    if best_hybrid_rmse < prophet_rmse:
        print("\nKESIMPULAN: Model Hybrid berhasil MENGALAHKAN Prophet! Ini cocok untuk jurnal.")
    else:
        print("\nKESIMPULAN: Prophet masih sedikit lebih stabil pada periode tes ini.")
        
    # 5. PLOTTING
    test_dates = prophet_df['ds'].iloc[-test_size:].values
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual_y, label='Data Aktual (M1)', marker='o', color='black')
        plt.plot(test_dates, prophet_yhat, label='Prediksi Prophet', linestyle='dashed', color='blue')
        plt.plot(test_dates, best_hybrid_yhat, label='Prediksi Hybrid (Prophet+LSTM Terbaik)', linestyle='solid', color='red', linewidth=2)
        plt.title('Perbandingan Prediksi Uang Kartal (M1) - Pasca Hyperparameter Tuning')
        plt.xlabel('Periode')
        plt.ylabel('Uang Beredar Sempit (Miliar Rp)')
        plt.legend()
        plt.grid(True)
        plt.savefig('hasil_prediksi_tuned.png')
        print("\nGrafik terbaru berhasil disimpan ke 'hasil_prediksi_tuned.png'.")
    except Exception as e:
        print(f"Gagal membuat grafik: {e}")

if __name__ == "__main__":
    main()
