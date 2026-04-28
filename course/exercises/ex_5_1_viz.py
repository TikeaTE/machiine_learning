import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression

# Load and prepare data (same as ex_5_1.py)
data = pd.read_csv('/home/tikea/ecam_ws/ecam_courses/machine_learning/data_3_1_2.csv').to_numpy()
x = data[:, :-1]
y = data[:, -1]

x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

m_train = int(0.6 * x.shape[0])
x_train, y_train = x[:m_train], y[:m_train]
x_test,  y_test  = x[m_train:], y[m_train:]

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

TP = np.sum((y_pred == 1) & (y_test == 1))
TN = np.sum((y_pred == 0) & (y_test == 0))
FP = np.sum((y_pred == 1) & (y_test == 0))
FN = np.sum((y_pred == 0) & (y_test == 1))

precision = TP / (TP + FP)
recall    = TP / (TP + FN)          # corrected formula
f_score   = 2 * precision * recall / (precision + recall)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5))
fig.suptitle('Logistic Regression – Binary Classification', fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# ── 1. Decision boundary ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])

x1_range = np.linspace(x[:, 0].min() - 0.5, x[:, 0].max() + 0.5, 300)
x2_range = np.linspace(x[:, 1].min() - 0.5, x[:, 1].max() + 0.5, 300)
xx1, xx2 = np.meshgrid(x1_range, x2_range)
Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)
ax1.contourf(xx1, xx2, Z, alpha=0.2, cmap='RdBu')
ax1.contour(xx1, xx2, Z, levels=[0.5], colors='k', linewidths=1.5, linestyles='--')

colors = {0: '#4878cf', 1: '#d65f5f'}
markers = {0: 'o', 1: 's'}
labels_map = {0: 'Class 0', 1: 'Class 1'}

for cls in [0, 1]:
    mask_tr = y_train == cls
    ax1.scatter(x_train[mask_tr, 0], x_train[mask_tr, 1],
                c=colors[cls], marker=markers[cls], edgecolors='k',
                linewidths=0.5, s=40, label=f'{labels_map[cls]} (train)', alpha=0.8)
    mask_te = y_test == cls
    ax1.scatter(x_test[mask_te, 0], x_test[mask_te, 1],
                c=colors[cls], marker=markers[cls], edgecolors='white',
                linewidths=1.2, s=60, label=f'{labels_map[cls]} (test)', alpha=0.9)

ax1.set_xlabel('x1 (scaled)')
ax1.set_ylabel('x2 (scaled)')
ax1.set_title('Decision Boundary')
ax1.legend(fontsize=7, loc='best')

# ── 2. Confusion matrix ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])

cm = np.array([[TN, FP],
               [FN, TP]])
im = ax2.imshow(cm, cmap='Blues', vmin=0)
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(int(cm[i, j])), ha='center', va='center',
                 fontsize=16, fontweight='bold',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')

ax2.set_xticks([0, 1]); ax2.set_xticklabels(['Predicted 0', 'Predicted 1'])
ax2.set_yticks([0, 1]); ax2.set_yticklabels(['Actual 0', 'Actual 1'])
ax2.set_title('Confusion Matrix\n(test set)')

# ── 3. Metrics bar chart ──────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])

metrics = {'Precision': precision, 'Recall': recall, 'F1-score': f_score}
bar_colors = ['#5aaa5a', '#5a8fcc', '#cc7a5a']
bars = ax3.bar(metrics.keys(), metrics.values(), color=bar_colors, edgecolor='k', linewidth=0.8)

for bar, val in zip(bars, metrics.values()):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax3.set_ylim(0, 1.15)
ax3.set_ylabel('Score')
ax3.set_title('Evaluation Metrics\n(test set)')
ax3.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)

plt.savefig('/home/tikea/ecam_ws/ecam_courses/machine_learning/codess/ex_5_1_viz.png',
            dpi=150, bbox_inches='tight')
plt.show()
print(f'Precision: {precision:.4f}  Recall: {recall:.4f}  F1-score: {f_score:.4f}')
print(f'Confusion matrix — TP:{TP}  TN:{TN}  FP:{FP}  FN:{FN}')
