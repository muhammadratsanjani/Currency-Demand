import pandas as pd
df = pd.read_csv('dataset_jimf_ready.csv')
df.fillna(0, inplace=True)
df.to_csv('dataset_jimf_ready.csv', index=False)
