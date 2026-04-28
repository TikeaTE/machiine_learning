import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from prepare_data import load_merged, get_features_and_targets

PLOTS_DIR = 'outputs/plots/'

def compute_PRF(z, y):
    TP = np.sum((z == 1) & (y == 1))
    FN = np.sum((z == 0) & (y == 1))
    FP = np.sum((z == 1) & (y == 0))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f_score   = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f_score

def run_logistic_regression():
    df = load_merged()
    X_train, X_test, _, _, yc_train, yc_test, _, _ = get_features_and_targets(df)

    model = LogisticRegression()
    model.fit(X_train, yc_train)

    y_pred = model.predict(X_test)
    precision, recall, f_score = compute_PRF(y_pred, yc_test)

    print('=== Logistic Regression ===')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1-score:  {f_score:.4f}')

    # Plot confusion matrix values as bar chart
    TP = int(np.sum((y_pred == 1) & (yc_test == 1)))
    TN = int(np.sum((y_pred == 0) & (yc_test == 0)))
    FP = int(np.sum((y_pred == 1) & (yc_test == 0)))
    FN = int(np.sum((y_pred == 0) & (yc_test == 1)))

    plt.figure(figsize=(5, 4))
    bars = plt.bar(['TP', 'TN', 'FP', 'FN'], [TP, TN, FP, FN],
                   color=['#5aaa5a', '#4878cf', '#d65f5f', '#e8a838'], edgecolor='k')
    for bar, val in zip(bars, [TP, TN, FP, FN]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 str(val), ha='center', fontweight='bold')
    plt.title('Logistic Regression — Confusion Matrix')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR + 'logistic_regression.png', dpi=150)
    plt.close()
    print(f'Plot saved to {PLOTS_DIR}logistic_regression.png')

    return {'Precision': precision, 'Recall': recall, 'F1': f_score}

if __name__ == '__main__':
    run_logistic_regression()
