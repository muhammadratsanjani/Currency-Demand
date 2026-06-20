import pandas as pd
from hijri_converter import Gregorian
import os
import glob

def check_ramadan_eid(date_obj):
    try:
        hijri_date = Gregorian(date_obj.year, date_obj.month, 15).to_hijri()
        is_ramadan = 1 if hijri_date.month == 9 else 0
        is_eid = 1 if hijri_date.month == 10 else 0
        return pd.Series([is_ramadan, is_eid])
    except Exception as e:
        return pd.Series([0, 0])

def main():
    print("Mulai proses penyiapan data...")
    
    # Cari file raw baik .csv maupun .csc
    raw_files = glob.glob("dataset_jimf_raw.cs*")
    
    if not raw_files:
        print("ERROR: File dataset_jimf_raw.csv (atau .csc) tidak ditemukan di direktori proyek!")
        return

    raw_file = raw_files[0]
    print(f"Membaca file: {raw_file}")
    
    # Baca data mentah, menggunakan pemisah koma (bisa disesuaikan jika menggunakan semicolon)
    # Jika ada error format IHK dan Suku_Bunga yang tergabung, kita tangani
    try:
        df = pd.read_csv(raw_file, sep=None, engine='python')
    except Exception as e:
        print(f"Error membaca CSV: {e}")
        return
        
    print(f"Kolom yang terdeteksi: {list(df.columns)}")
    
    if 'Periode' not in df.columns:
        print("ERROR: Kolom 'Periode' tidak ditemukan. Pastikan huruf besar/kecil sesuai.")
        return
        
    # Konversi kolom Periode menjadi datetime
    df['Periode'] = pd.to_datetime(df['Periode'])
    
    # Buat variabel dummy Ramadan dan Idul Fitri
    print("Menghitung (*mapping*) kalender Hijriah...")
    df[['Dummy_Ramadan', 'Dummy_IdulFitri']] = df['Periode'].apply(check_ramadan_eid)
    
    # Pastikan data diurutkan berdasarkan waktu
    df = df.sort_values('Periode').reset_index(drop=True)
    
    # Menangani Missing Values (Interpolasi linear jika ada yang kosong)
    print("Menangani data kosong (Missing Values)...")
    df.interpolate(method='linear', inplace=True)
    
    # Simpan dataset yang sudah siap ke file baru
    output_file = "dataset_jimf_ready.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\nProses Selesai! Data siap dilatih. Tersimpan di: {output_file}")
    print("\nTampilan sekilas dataset:")
    print(df.head())

if __name__ == "__main__":
    main()
