import pandas as pd

# 1. Load Data
df = pd.read_csv('Flood_Datasets.csv')

# 2. Convert Date column to Datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# 3. Create a 'Year' column helper
df['Year'] = df['Date'].dt.year

# 4. SPLIT: Create a dictionary of DataFrames
# This loops through every unique year and creates a separate table for it
yearly_datasets = {year: df[df['Year'] == year] for year in df['Year'].unique()}

# --- HOW TO USE THEM ---
# Access specific years using the key (e.g., 2016)
df_2016 = yearly_datasets[2016]
df_2017 = yearly_datasets[2017]
df_2018 = yearly_datasets[2018]
df_2019 = yearly_datasets[2019]
df_2020 = yearly_datasets[2020]

print(f"2016 Data size: {df_2016.shape}")
print(f"2017 Data size: {df_2017.shape}")
print(f"2018 Data size: {df_2018.shape}")
print(f"2019 Data size: {df_2019.shape}")
print(f"2020 Data size: {df_2020.shape}")

df_2017 = df[df['Date'].dt.year == 2017]
df_2017.to_csv('df_2017.csv', index=False)
df_2018 = df[df['Date'].dt.year == 2018]
df_2018.to_csv('df_2018.csv', index=False)
df_2019 = df[df['Date'].dt.year == 2019]
df_2019.to_csv('df_2019.csv', index=False)
df_2020 = df[df['Date'].dt.year == 2020]
df_2020.to_csv('df_2020.csv', index=False)
