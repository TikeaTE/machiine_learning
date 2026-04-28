import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from prepare_data import load_merged, get_features_and_targets
from classification import compute_PRF

PLOTS_DIR = 'outputs/plots/'

# ── Regression NN: 4 → 8 → 1 (no final activation) ──────────────────────────
class RegressionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layers(x).flatten()

# ── Classification NN: 4 → 8 → 1 (Sigmoid output) ───────────────────────────
class ClassificationNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x).flatten()

def train_model(model, tx, ty, cost_func, num_epochs=500, lr=0.1):
    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for i in range(num_epochs):
        output = model(tx)
        loss = cost_func(output, ty)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return losses

def run_neural_network():
    df = load_merged()
    X_train, X_test, yr_train, yr_test, yc_train, yc_test, _, _ = get_features_and_targets(df)

    tx_train = torch.tensor(X_train, dtype=torch.float32)
    tx_test  = torch.tensor(X_test,  dtype=torch.float32)

    # ── Regression ────────────────────────────────────────────────────────────
    tyr_train = torch.tensor(yr_train, dtype=torch.float32)
    tyr_test  = torch.tensor(yr_test,  dtype=torch.float32)

    reg_model = RegressionNN()
    reg_losses = train_model(reg_model, tx_train, tyr_train, nn.MSELoss())

    with torch.no_grad():
        yr_pred = reg_model(tx_test).numpy()

    mse  = float(np.mean((yr_pred - yr_test) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((yr_test - yr_pred) ** 2)
    ss_tot = np.sum((yr_test - yr_test.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    print('=== Neural Network (Regression) ===')
    print(f'MSE:  {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'R²:   {r2:.4f}')

    # ── Classification ────────────────────────────────────────────────────────
    tyc_train = torch.tensor(yc_train, dtype=torch.float32)
    tyc_test  = torch.tensor(yc_test,  dtype=torch.float32)

    cls_model = ClassificationNN()
    cls_losses = train_model(cls_model, tx_train, tyc_train, nn.BCELoss())

    with torch.no_grad():
        yc_pred = (cls_model(tx_test) > 0.5).int().numpy()

    precision, recall, f_score = compute_PRF(yc_pred, yc_test)

    print('=== Neural Network (Classification) ===')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1-score:  {f_score:.4f}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Neural Network Results', fontweight='bold')

    # Loss curves
    axes[0].plot(reg_losses, color='#d65f5f', lw=1.5, label='Regression (MSE)')
    axes[0].plot(cls_losses, color='#4878cf', lw=1.5, label='Classification (BCE)')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Predicted vs actual (regression)
    axes[1].plot(yr_pred, 'o--', color='#d65f5f', label='Predicted')
    axes[1].plot(yr_test, 's-',  color='#4878cf', label='Actual')
    axes[1].set_title('Regression — Predicted vs Actual')
    axes[1].set_xlabel('Test sample index')
    axes[1].set_ylabel('Yield (kg/ha)')
    axes[1].legend()

    # Classification metrics bar
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

    return {
        'reg':  {'MSE': mse, 'RMSE': rmse, 'R2': r2},
        'cls':  {'Precision': precision, 'Recall': recall, 'F1': f_score},
        'reg_losses': reg_losses,
        'cls_losses': cls_losses,
        'yr_pred': yr_pred,
        'yr_test': yr_test
    }

if __name__ == '__main__':
    run_neural_network()
