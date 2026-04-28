import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from prepare_data import load_merged, get_features_and_targets, get_full_xy, FEATURE_COLS

PLOTS_DIR = 'outputs/plots/'

def _metrics(y_true, y_pred):
    mse  = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return mse, rmse, r2

def cross_validate_linear(X, y, k=5, seed=42):
    """5-fold CV for Linear Regression. Standardises within each fold."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    rmses, r2s = [], []
    for tr, te in kf.split(X):
        Xt, Xv = X[tr], X[te]
        yt, yv = y[tr], y[te]
        mu, sd = Xt.mean(axis=0), Xt.std(axis=0)
        Xt = (Xt - mu) / sd
        Xv = (Xv - mu) / sd
        m = LinearRegression().fit(Xt, yt)
        _, rmse, r2 = _metrics(yv, m.predict(Xv))
        rmses.append(rmse); r2s.append(r2)
    return float(np.mean(rmses)), float(np.std(rmses)), float(np.mean(r2s)), float(np.std(r2s))

def run_linear_regression():
    df = load_merged()
    X_train, X_test, yr_train, yr_test, _, _, _, _ = get_features_and_targets(df)

    model = LinearRegression()
    model.fit(X_train, yr_train)

    y_pred = model.predict(X_test)
    mse, rmse, r2 = _metrics(yr_test, y_pred)

    # Naive baseline: always predict the training-set mean
    base_pred = np.full_like(yr_test, yr_train.mean(), dtype=float)
    base_mse, base_rmse, base_r2 = _metrics(yr_test, base_pred)

    # 5-fold CV on the full dataset
    X_full, y_full = get_full_xy(df)
    cv_rmse_mean, cv_rmse_std, cv_r2_mean, cv_r2_std = cross_validate_linear(X_full, y_full)

    print('=== Linear Regression — Chronological Split (28 train / 7 test) ===')
    print(f'MSE:  {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'R²:   {r2:.4f}')
    print(f'\nNaive baseline (predict train mean):')
    print(f'  RMSE: {base_rmse:.2f}   R²: {base_r2:.4f}')
    print(f'\n=== Linear Regression — 5-fold CV (random shuffled) ===')
    print(f'RMSE: {cv_rmse_mean:.2f} ± {cv_rmse_std:.2f}')
    print(f'R²:   {cv_r2_mean:.4f} ± {cv_r2_std:.4f}')
    print('\nStandardised coefficients (chronological split):')
    for name, coef in zip(FEATURE_COLS, model.coef_):
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

    return {
        'MSE': mse, 'RMSE': rmse, 'R2': r2,
        'baseline_RMSE': base_rmse, 'baseline_R2': base_r2,
        'cv_RMSE_mean': cv_rmse_mean, 'cv_RMSE_std': cv_rmse_std,
        'cv_R2_mean': cv_r2_mean, 'cv_R2_std': cv_r2_std,
        'coefs': dict(zip(FEATURE_COLS, [float(c) for c in model.coef_])),
    }

if __name__ == '__main__':
    run_linear_regression()
