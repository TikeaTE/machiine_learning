import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, StratifiedKFold

from prepare_data import load_merged, get_features_and_targets, get_full_xy
from classification import compute_PRF

PLOTS_DIR = 'outputs/plots/'
N_FEATURES = 3

# ── Regression NN: 3 → 8 → 1 (no final activation) ──────────────────────────
class RegressionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(N_FEATURES, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layers(x).flatten()

# ── Classification NN: 3 → 8 → 1 (Sigmoid output) ───────────────────────────
class ClassificationNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(N_FEATURES, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x).flatten()

def train_model(model, tx, ty, cost_func, num_epochs=500, lr=0.1,
                tx_val=None, ty_val=None):
    opt = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    for i in range(num_epochs):
        output = model(tx)
        loss = cost_func(output, ty)
        loss.backward()
        opt.step()
        opt.zero_grad()
        train_losses.append(loss.item())
        if tx_val is not None and ty_val is not None:
            with torch.no_grad():
                val_losses.append(cost_func(model(tx_val), ty_val).item())
    return train_losses, val_losses

def _metrics_reg(y_true, y_pred):
    mse  = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return mse, rmse, r2

def cv_nn_regression(X, y, k=5, seed=42, num_epochs=500, lr=0.1):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    rmses, r2s = [], []
    for tr, te in kf.split(X):
        torch.manual_seed(seed)
        Xt, Xv = X[tr], X[te]
        yt, yv = y[tr], y[te]
        mu, sd = Xt.mean(axis=0), Xt.std(axis=0)
        Xt = (Xt - mu) / sd; Xv = (Xv - mu) / sd
        tx = torch.tensor(Xt, dtype=torch.float32)
        tv = torch.tensor(Xv, dtype=torch.float32)
        ty = torch.tensor(yt, dtype=torch.float32)
        m = RegressionNN()
        train_model(m, tx, ty, nn.MSELoss(), num_epochs=num_epochs, lr=lr)
        with torch.no_grad():
            pred = m(tv).numpy()
        _, rmse, r2 = _metrics_reg(yv, pred)
        rmses.append(rmse); r2s.append(r2)
    return float(np.mean(rmses)), float(np.std(rmses)), float(np.mean(r2s)), float(np.std(r2s))

def cv_nn_classification(X, y_reg, k=5, seed=42, num_epochs=500, lr=0.1):
    strat = (y_reg >= np.median(y_reg)).astype(int)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    precs, recs, f1s = [], [], []
    for tr, te in skf.split(X, strat):
        torch.manual_seed(seed)
        Xt, Xv = X[tr], X[te]
        yrt, yrv = y_reg[tr], y_reg[te]
        thr = float(np.median(yrt))
        yt = (yrt >= thr).astype(np.float32)
        yv = (yrv >= thr).astype(np.float32)
        if len(np.unique(yt)) < 2:
            continue
        mu, sd = Xt.mean(axis=0), Xt.std(axis=0)
        Xt = (Xt - mu) / sd; Xv = (Xv - mu) / sd
        tx = torch.tensor(Xt, dtype=torch.float32)
        tv = torch.tensor(Xv, dtype=torch.float32)
        ty = torch.tensor(yt, dtype=torch.float32)
        m = ClassificationNN()
        train_model(m, tx, ty, nn.BCELoss(), num_epochs=num_epochs, lr=lr)
        with torch.no_grad():
            pred = (m(tv).numpy() > 0.5).astype(int)
        p, r, f = compute_PRF(pred, yv.astype(int))
        precs.append(p); recs.append(r); f1s.append(f)
    return (float(np.mean(precs)), float(np.std(precs)),
            float(np.mean(recs)),  float(np.std(recs)),
            float(np.mean(f1s)),   float(np.std(f1s)))

def run_neural_network():
    torch.manual_seed(42)
    np.random.seed(42)

    df = load_merged()
    X_train, X_test, yr_train, yr_test, yc_train, yc_test, _, _ = get_features_and_targets(df)

    tx_train = torch.tensor(X_train, dtype=torch.float32)
    tx_test  = torch.tensor(X_test,  dtype=torch.float32)

    # ── Regression ────────────────────────────────────────────────────────────
    tyr_train = torch.tensor(yr_train, dtype=torch.float32)
    tyr_test  = torch.tensor(yr_test,  dtype=torch.float32)

    torch.manual_seed(42)
    reg_model = RegressionNN()
    reg_train_losses, reg_val_losses = train_model(
        reg_model, tx_train, tyr_train, nn.MSELoss(),
        tx_val=tx_test, ty_val=tyr_test
    )

    with torch.no_grad():
        yr_pred = reg_model(tx_test).numpy()

    mse, rmse, r2 = _metrics_reg(yr_test, yr_pred)

    print('=== Neural Network — Regression (Chronological Split) ===')
    print(f'MSE:  {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'R²:   {r2:.4f}')

    # ── Classification ────────────────────────────────────────────────────────
    tyc_train = torch.tensor(yc_train, dtype=torch.float32)
    tyc_test  = torch.tensor(yc_test,  dtype=torch.float32)

    torch.manual_seed(42)
    cls_model = ClassificationNN()
    cls_train_losses, cls_val_losses = train_model(
        cls_model, tx_train, tyc_train, nn.BCELoss(),
        tx_val=tx_test, ty_val=tyc_test
    )

    with torch.no_grad():
        yc_pred = (cls_model(tx_test) > 0.5).int().numpy()

    precision, recall, f_score = compute_PRF(yc_pred, yc_test)

    print('=== Neural Network — Classification (Chronological Split) ===')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1-score:  {f_score:.4f}')

    # ── 5-fold CV ─────────────────────────────────────────────────────────────
    X_full, y_reg_full = get_full_xy(df)
    cv_rmse_m, cv_rmse_s, cv_r2_m, cv_r2_s = cv_nn_regression(X_full, y_reg_full)
    cv_p_m, cv_p_s, cv_r_m, cv_r_s, cv_f_m, cv_f_s = cv_nn_classification(X_full, y_reg_full)
    print(f'\n=== Neural Network — 5-fold CV (Regression) ===')
    print(f'RMSE: {cv_rmse_m:.2f} ± {cv_rmse_s:.2f}')
    print(f'R²:   {cv_r2_m:.4f} ± {cv_r2_s:.4f}')
    print(f'\n=== Neural Network — 5-fold Stratified CV (Classification) ===')
    print(f'Precision: {cv_p_m:.4f} ± {cv_p_s:.4f}')
    print(f'Recall:    {cv_r_m:.4f} ± {cv_r_s:.4f}')
    print(f'F1:        {cv_f_m:.4f} ± {cv_f_s:.4f}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Neural Network Results', fontweight='bold')

    axes[0].plot(reg_train_losses, color='#d65f5f', lw=1.5, label='Train (MSE)')
    axes[0].plot(reg_val_losses,   color='#4878cf', lw=1.5, label='Test (MSE)',  linestyle='--')
    axes[0].set_title('Regression Training Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(yr_pred, 'o--', color='#d65f5f', label='Predicted')
    axes[1].plot(yr_test, 's-',  color='#4878cf', label='Actual')
    axes[1].set_title('Regression — Predicted vs Actual')
    axes[1].set_xlabel('Test sample index')
    axes[1].set_ylabel('Yield (kg/ha)')
    axes[1].legend()

    metrics = [precision, recall, f_score]
    axes[2].bar(['Precision', 'Recall', 'F1'], metrics,
                color=['#5aaa5a', '#4878cf', '#d65f5f'], edgecolor='k')
    for i, v in enumerate(metrics):
        axes[2].text(i, v + 0.01, f'{v:.2f}', ha='center', fontweight='bold')
    axes[2].set_ylim(0, 1.2)
    axes[2].set_title('Classification Metrics')
    axes[2].set_ylabel('Score')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR + 'neural_network.png', dpi=150)
    plt.close()
    print(f'Plot saved to {PLOTS_DIR}neural_network.png')

    # Standalone training-curve figure for the report
    plt.figure(figsize=(6, 4))
    plt.plot(reg_train_losses, color='#d65f5f', lw=1.5, label='Train MSE')
    plt.plot(reg_val_losses,   color='#4878cf', lw=1.5, label='Test MSE', linestyle='--')
    plt.title('Neural Network — Regression Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR + 'nn_training_curve.png', dpi=150)
    plt.close()
    print(f'Plot saved to {PLOTS_DIR}nn_training_curve.png')

    return {
        'reg': {'MSE': mse, 'RMSE': rmse, 'R2': r2,
                'cv_RMSE_mean': cv_rmse_m, 'cv_RMSE_std': cv_rmse_s,
                'cv_R2_mean': cv_r2_m, 'cv_R2_std': cv_r2_s},
        'cls': {'Precision': precision, 'Recall': recall, 'F1': f_score,
                'cv_P_mean': cv_p_m, 'cv_P_std': cv_p_s,
                'cv_R_mean': cv_r_m, 'cv_R_std': cv_r_s,
                'cv_F_mean': cv_f_m, 'cv_F_std': cv_f_s},
    }

if __name__ == '__main__':
    run_neural_network()
