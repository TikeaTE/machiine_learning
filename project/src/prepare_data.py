import numpy as np
import pandas as pd

np.random.seed(42)

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
    df = df.sort_values('year').reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f'Merged dataset saved to {OUTPUT_PATH}')
    print(f'Shape: {df.shape}')
    print(f'Years: {df["year"].min()} - {df["year"].max()}')
    print(f'\nSample:\n{df.head()}')
    return df

def load_merged():
    return pd.read_csv(OUTPUT_PATH)

FEATURE_COLS = ['area_harvested', 'avg_temp', 'rainfall_mm']

def get_features_and_targets(df):
    # 'production' is excluded: yield ≈ production / area × 1000, so including
    # it causes direct target leakage in the regression task.
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y_reg = df['yield_kg_ha'].to_numpy(dtype=float)

    # Train/test split first — 80/20 (28 train, 7 test)
    # Note: small dataset (35 rows), results should be interpreted with caution
    m_train = int(0.8 * len(X))
    X_train_raw, X_test_raw = X[:m_train], X[m_train:]
    yr_train, yr_test = y_reg[:m_train], y_reg[m_train:]

    # Classification threshold: median computed on TRAIN only — using the full
    # dataset would leak future yields into the threshold (look-ahead bias).
    train_median = float(np.median(yr_train))
    yc_train = (yr_train >= train_median).astype(float)
    yc_test  = (yr_test  >= train_median).astype(float)

    # Fit scaler on training set only to prevent test-set contamination
    X_mean = X_train_raw.mean(axis=0)
    X_std  = X_train_raw.std(axis=0)
    X_train = (X_train_raw - X_mean) / X_std
    X_test  = (X_test_raw  - X_mean) / X_std

    return X_train, X_test, yr_train, yr_test, yc_train, yc_test, X_mean, X_std

def get_full_xy(df):
    """Return raw (un-scaled) features and targets for cross-validation."""
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y_reg = df['yield_kg_ha'].to_numpy(dtype=float)
    return X, y_reg

if __name__ == '__main__':
    df = load_and_merge()
