import numpy as np
import pandas as pd

DATA_DIR = 'data/'
OUTPUT_PATH = DATA_DIR + 'cambodia_merged.csv'

def load_and_merge():
    # --- Load FAO rice data and pivot from long to wide ---
    fao = pd.read_csv(DATA_DIR + 'cambodia_rice_faostat.csv')
    fao = fao[['Year', 'Element', 'Value']]
    fao = fao.pivot(index='Year', columns='Element', values='Value').reset_index()
    fao.columns.name = None
    fao = fao.rename(columns={
        'Year': 'year',
        'Area harvested': 'area_harvested',
        'Production': 'production',
        'Yield': 'yield_kg_ha'
    })

    # --- Load temperature data ---
    temp = pd.read_csv(DATA_DIR + 'cambodia_temperature_annual.csv')
    temp.columns = ['year', 'avg_temp']

    # --- Load rainfall data ---
    rain = pd.read_csv(DATA_DIR + 'cambodia_rainfall_annual.csv')
    rain = rain[['Category', 'Precipitation (Annual Mean mm)']].copy()
    rain.columns = ['year', 'rainfall_mm']

    # --- Merge all three on year ---
    df = fao.merge(temp, on='year').merge(rain, on='year')

    # --- Add binary classification label (1 = high yield, 0 = low yield) ---
    median_yield = df['yield_kg_ha'].median()
    df['high_yield'] = (df['yield_kg_ha'] >= median_yield).astype(int)

    df = df.sort_values('year').reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f'Merged dataset saved to {OUTPUT_PATH}')
    print(f'Shape: {df.shape}')
    print(f'Years: {df["year"].min()} - {df["year"].max()}')
    print(f'Yield median (threshold): {median_yield:.1f} kg/ha')
    print(f'\nSample:\n{df.head()}')
    return df

def load_merged():
    return pd.read_csv(OUTPUT_PATH)

def get_features_and_targets(df):
    feature_cols = ['area_harvested', 'production', 'avg_temp', 'rainfall_mm']
    X = df[feature_cols].to_numpy(dtype=float)
    y_reg = df['yield_kg_ha'].to_numpy(dtype=float)
    y_cls = df['high_yield'].to_numpy(dtype=float)

    # Feature scaling
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std

    # Train/test split — 80/20 (28 train, 7 test)
    # Note: small dataset (35 rows), results should be interpreted with caution
    m_train = int(0.8 * len(X_scaled))
    X_train, X_test   = X_scaled[:m_train], X_scaled[m_train:]
    yr_train, yr_test = y_reg[:m_train], y_reg[m_train:]
    yc_train, yc_test = y_cls[:m_train], y_cls[m_train:]

    return X_train, X_test, yr_train, yr_test, yc_train, yc_test, X_mean, X_std

if __name__ == '__main__':
    df = load_and_merge()
