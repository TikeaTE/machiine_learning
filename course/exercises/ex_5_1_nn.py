import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Data (same preprocessing as ex_5_1.py) ────────────────────────────────────
data = pd.read_csv('/home/tikea/ecam_ws/ecam_courses/machine_learning/data_3_1_2.csv').to_numpy()
x = data[:, :-1]
y = data[:, -1].reshape(-1, 1)          # (m, 1)

x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

m_train = int(0.6 * x.shape[0])
x_train, y_train = x[:m_train].T, y[:m_train].T   # shape: (n, m)
x_test,  y_test  = x[m_train:].T, y[m_train:].T

# ── Architecture: 2 → 8 → 1 ──────────────────────────────────────────────────
#   Input layer  : 2 neurons  (x1, x2)
#   Hidden layer : 8 neurons  (ReLU)
#   Output layer : 1 neuron   (Sigmoid → binary class)

np.random.seed(42)
n0, n1, n2 = 2, 8, 1

W1 = np.random.randn(n1, n0) * np.sqrt(2.0 / n0)   # He init
b1 = np.zeros((n1, 1))
W2 = np.random.randn(n2, n1) * np.sqrt(2.0 / n1)
b2 = np.zeros((n2, 1))

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(x, W1, b1, W2, b2):
    Z1 = W1 @ x + b1          # (n1, m)
    A1 = relu(Z1)              # (n1, m)
    Z2 = W2 @ A1 + b2          # (1,  m)
    A2 = sigmoid(Z2)           # (1,  m)  ← output probability
    cache = (x, Z1, A1, Z2, A2)
    return A2, cache

def compute_loss(A2, y):
    m = y.shape[1]
    eps = 1e-9
    return -np.mean(y * np.log(A2 + eps) + (1 - y) * np.log(1 - A2 + eps))

def backward(y, cache, W1, W2):
    x, Z1, A1, Z2, A2 = cache
    m = y.shape[1]

    dZ2 = A2 - y                          # (1, m)
    dW2 = dZ2 @ A1.T / m
    db2 = np.mean(dZ2, axis=1, keepdims=True)

    dA1 = W2.T @ dZ2                      # (n1, m)
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = dZ1 @ x.T / m
    db1 = np.mean(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# ── Training ──────────────────────────────────────────────────────────────────
lr       = 0.1
epochs   = 2000
losses   = []

for epoch in range(1, epochs + 1):
    A2, cache = forward(x_train, W1, b1, W2, b2)
    loss = compute_loss(A2, y_train)
    losses.append(loss)

    dW1, db1, dW2, db2 = backward(y_train, cache, W1, W2)
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

# ── Evaluation ────────────────────────────────────────────────────────────────
A2_test, _ = forward(x_test, W1, b1, W2, b2)
y_pred = (A2_test >= 0.5).astype(int)   # threshold at 0.5

TP = int(np.sum((y_pred == 1) & (y_test == 1)))
TN = int(np.sum((y_pred == 0) & (y_test == 0)))
FP = int(np.sum((y_pred == 1) & (y_test == 0)))
FN = int(np.sum((y_pred == 0) & (y_test == 1)))

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f_score   = (2 * precision * recall / (precision + recall)
             if (precision + recall) > 0 else 0)

print(f'Precision: {precision:.4f}  Recall: {recall:.4f}  F1-score: {f_score:.4f}')
print(f'Confusion matrix — TP:{TP}  TN:{TN}  FP:{FP}  FN:{FN}')

# ── Visualisation ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5))
fig.suptitle('Neural Network (2 → 8 → 1) – Binary Classification', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38)

# 1. Architecture diagram
ax0 = fig.add_subplot(gs[0])
ax0.set_xlim(-1, 5); ax0.set_ylim(-1, 10); ax0.axis('off')
ax0.set_title('Architecture', pad=8)

layer_sizes = [n0, n1, n2]
layer_labels = ['Input\n(2)', 'Hidden\n(8, ReLU)', 'Output\n(1, σ)']
layer_x = [0, 2, 4]
layer_colors = ['#4878cf', '#d65f5f', '#5aaa5a']

node_positions = []
for l, (lx, size) in enumerate(zip(layer_x, layer_sizes)):
    ys = np.linspace(1, 9, size)
    node_positions.append(list(zip([lx]*size, ys)))
    for y_node in ys:
        circle = plt.Circle((lx, y_node), 0.35, color=layer_colors[l], zorder=3, ec='k', lw=0.8)
        ax0.add_patch(circle)
    ax0.text(lx, 0.1, layer_labels[l], ha='center', va='top', fontsize=7.5)

for l in range(len(layer_sizes) - 1):
    for (x1n, y1n) in node_positions[l]:
        for (x2n, y2n) in node_positions[l+1]:
            ax0.plot([x1n, x2n], [y1n, y2n], 'k-', alpha=0.12, lw=0.6, zorder=1)

ax0.text(1, 9.6, 'He init\n+ ReLU', ha='center', fontsize=7, color='gray')
ax0.text(3, 9.6, 'BCE loss\nlr=0.1', ha='center', fontsize=7, color='gray')

# 2. Loss curve
ax1 = fig.add_subplot(gs[1])
ax1.plot(losses, color='#4878cf', lw=1.5)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Binary Cross-Entropy Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

# 3. Decision boundary
ax2 = fig.add_subplot(gs[2])
x_all = x.T   # (2, 200)
r = np.linspace(x_all[0].min() - 0.5, x_all[0].max() + 0.5, 300)
c = np.linspace(x_all[1].min() - 0.5, x_all[1].max() + 0.5, 300)
xx, yy = np.meshgrid(r, c)
grid = np.c_[xx.ravel(), yy.ravel()].T          # (2, N)
Z, _ = forward(grid, W1, b1, W2, b2)
Z = Z.reshape(xx.shape)

ax2.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.3)
ax2.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=1.5, linestyles='--')

colors_cls = {0: '#4878cf', 1: '#d65f5f'}
markers_cls = {0: 'o', 1: 's'}
y_train_flat = y_train.flatten().astype(int)
y_test_flat  = y_test.flatten().astype(int)

for cls in [0, 1]:
    mask = y_train_flat == cls
    ax2.scatter(x_train[0, mask], x_train[1, mask],
                c=colors_cls[cls], marker=markers_cls[cls],
                edgecolors='k', linewidths=0.5, s=40, alpha=0.8)
    mask = y_test_flat == cls
    ax2.scatter(x_test[0, mask], x_test[1, mask],
                c=colors_cls[cls], marker=markers_cls[cls],
                edgecolors='white', linewidths=1.2, s=60, alpha=0.9)

ax2.set_xlabel('x1 (scaled)'); ax2.set_ylabel('x2 (scaled)')
ax2.set_title('Decision Boundary')

# 4. Metrics comparison (NN vs Logistic Regression)
ax3 = fig.add_subplot(gs[3])
lr_metrics = {'Precision': 0.5625, 'Recall': 0.9730, 'F1': 0.7129}
nn_metrics = {'Precision': precision, 'Recall': recall, 'F1': f_score}

x_pos = np.arange(3)
w = 0.35
bars1 = ax3.bar(x_pos - w/2, lr_metrics.values(), w, label='Logistic Reg.', color='#aaaacc', edgecolor='k', lw=0.8)
bars2 = ax3.bar(x_pos + w/2, nn_metrics.values(), w, label='Neural Net',    color='#d65f5f', edgecolor='k', lw=0.8)

for bar, val in list(zip(bars1, lr_metrics.values())) + list(zip(bars2, nn_metrics.values())):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax3.set_xticks(x_pos)
ax3.set_xticklabels(['Precision', 'Recall', 'F1'])
ax3.set_ylim(0, 1.2)
ax3.set_ylabel('Score')
ax3.set_title('NN vs Logistic Regression\n(test set)')
ax3.legend(fontsize=8)
ax3.axhline(1.0, color='gray', linestyle='--', lw=0.8)

plt.savefig('/home/tikea/ecam_ws/ecam_courses/machine_learning/codess/ex_5_1_nn.png',
            dpi=150, bbox_inches='tight')
print('Saved ex_5_1_nn.png')
