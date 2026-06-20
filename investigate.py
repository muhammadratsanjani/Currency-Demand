import pandas as pd

path_m1 = r'C:\Users\Acer\Documents\2026\11.JIMF(kebutuhanUang)\Data_Clean\Raw_Extracted\SEKI_MEI_2026\TABEL1_1.xls'
df = pd.read_excel(path_m1, header=None)

for i, row in df.iterrows():
    row_str = ' '.join([str(x) for x in row.values])
    if 'Sempit' in row_str or '1985' in row_str or '2010' in row_str or '2020' in row_str:
        print(f"Row {i}: {row.values[:10]}")
