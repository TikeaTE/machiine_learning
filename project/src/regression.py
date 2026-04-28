import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from prepare_data import load_merged, get_features_and_targets

PLOTS_DIR = 'outputs/plots/'

FEATURE_NAMES = ['area_harvested', 'avg_temp', 'rainfall_mm']

def run_linear_regression():
    df = load_merged()
    X_train, X_test, yr_train, yr_test, _, _, _, _ = get_features_and_targets(df)

    model = LinearRegression()
    model.fit(X_train, yr_train)

    y_pred = model.predict(X_test)

    mse  = np.mean((y_pred - yr_test) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((yr_test - y_pred) ** 2)
    ss_tot = np.sum((yr_test - yr_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Naive baseline: always predict the training-set mean
    baseline_rmse = float(np.sqrt(np.mean((yr_train.mean() - yr_test) ** 2)))

    print('=== Linear Regression ===')
    print(f'MSE:  {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'R²:   {r2:.4f}')
    print(f'Naive baseline RMSE (train mean): {baseline_rmse:.2f}')
    print('\nStandardised coefficients:')
    for name, coef in zip(FEATURE_NAMES, model.coef_):
        print(f'  {name}: {coef:+.4f}')

    # Plot predicted vs actual
    plt.figure(figsize=(6, 4))
    plt.plot(y_pred,  'o--', label='Predicted', color='#d65f5f')
    plt.plot(yr_test, 's-',  label='Actual',    color='#4878cf')
    plt.title('Linear Regression — Predicted vs Actual Yield')
    plt.xlabel('Test sample index')
    plt.ylabel('Yield (kg/ha)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR + 'linear_regression.png', dpi=150)
    plt.close()
    print(f'Plot saved to {PLOTS_DIR}linear_regression.png')

    return {'MSE': mse, 'RMSE': rmse, 'R2': r2}

if __name__ == '__main__':
    run_linear_regression()
