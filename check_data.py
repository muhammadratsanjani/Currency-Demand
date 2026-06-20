import pandas as pd
import glob
import os

files = {
    'M1': r'C:\Users\Acer\Documents\2026\11.JIMF(kebutuhanUang)\Data_Clean\Raw_Extracted\SEKI_MEI_2026\TABEL1_1.xls',
    'IHK': r'C:\Users\Acer\Documents\2026\11.JIMF(kebutuhanUang)\Data_Clean\Raw_Extracted\SEKI_MEI_2026\TABEL8_1.xls',
    'Suku_Bunga': r'C:\Users\Acer\Documents\2026\11.JIMF(kebutuhanUang)\Data_Clean\Raw_Extracted\SEKI_MEI_2026\TABEL1_25_1.xls',
    'SPIP': r'C:\Users\Acer\Documents\2026\11.JIMF(kebutuhanUang)\Data_Clean\Raw_Extracted\SPIP-Mei-2026\SPIP-Mei-2026.xlsx',
}

ipr_files = glob.glob(r'C:\Users\Acer\Documents\2026\11.JIMF(kebutuhanUang)\Data_Clean\Raw_Extracted\Data-Series-SPE-Maret-2026\*.xls*')
if ipr_files:
    files['IPR'] = ipr_files[0]

for name, path in files.items():
    print(f'\n--- {name} ---')
    try:
        if name == 'SPIP':
            df = pd.read_excel(path, sheet_name='5e', nrows=15)
        else:
            df = pd.read_excel(path, nrows=15)
        print(df.iloc[:10, :5].to_string())
    except Exception as e:
        print(f'Error reading {name}: {e}')
