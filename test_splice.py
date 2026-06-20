import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('dataset_jimf_ready.csv')
cpi = df['IHK'].copy()

# Find the break point. The drop happens when it goes from ~139 to ~104.
diffs = cpi.diff()
break_idx = diffs.idxmin()
print(f"Break detected at index {break_idx}, Date: {df['Date'].iloc[break_idx]}")
print(f"Before break: {cpi.iloc[break_idx-1]}, After break: {cpi.iloc[break_idx]}")

# Splice the series: we want to adjust the old series (before the break) to the new base, or the new series to the old base.
# Usually we bring the old base to the new base (so recent numbers are correct).
# Ratio: new_base / old_base at the overlap. Since we don't have overlap, we estimate using the month before.
# Actually, BPS changed base year from 2012=100 to 2018=100.
# The index for 2018 under 2012 base was around 132. The index for 2018 under 2018 base is 100.
# The ratio is exactly 100 / (average of 2018).
ratio = cpi.iloc[break_idx] / cpi.iloc[break_idx-1]
print(f"Splicing ratio: {ratio}")

# Adjust the 'before' part to match the new base
cpi_spliced = cpi.copy()
cpi_spliced.iloc[:break_idx] = cpi_spliced.iloc[:break_idx] * ratio

# Test ADF on spliced
adf_level = adfuller(cpi_spliced)
adf_diff = adfuller(cpi_spliced.diff().dropna())

print(f"ADF Level (Spliced): {adf_level[0]:.3f}, p-value: {adf_level[1]:.3f}")
print(f"ADF 1st Diff (Spliced): {adf_diff[0]:.3f}, p-value: {adf_diff[1]:.3f}")
