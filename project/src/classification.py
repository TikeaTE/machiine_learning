import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from prepare_data import load_merged, get_features_and_targets, get_full_xy

PLOTS_DIR = 'outputs/plots/'

def compute_PRF(z, y):
    z = np.asarray(z); y = np.asarray(y)
    TP = np.sum((z == 1) & (y == 1))
    FN = np.sum((z == 0) & (y == 1))
    FP = np.sum((z == 1) & (y == 0))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f_score   = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return float(precision), float(recall), float(f_score)

def cross_validate_logistic(X, y_reg, k=5, seed=42):
    """5-fold stratified CV for Logistic Regression. Threshold is set on each
    fold's train split (no look-ahead) and features are standardised per fold."""
    # Build labels stratified across the full data using full-data median ONLY
    # for the purpose of producing balanced folds; the actual training
    # threshold is recomputed inside each fold.
    strat_labels = (y_reg >= np.median(y_reg)).astype(int)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    precs, recs, f1s = [], [], []
    for tr, te in skf.split(X, strat_labels):
        Xt, Xv = X[tr], X[te]
        yrt, yrv = y_reg[tr], y_reg[te]
        thr = float(np.median(yrt))
        yt = (yrt >= thr).astype(int)
        yv = (yrv >= thr).astype(int)
        # Skip degenerate folds where train has only one class
        if len(np.unique(yt)) < 2:
            continue
        mu, sd = Xt.mean(axis=0), Xt.std(axis=0)
        Xt = (Xt - mu) / sd
        Xv = (Xv - mu) / sd
        m = LogisticRegression().fit(Xt, yt)
        p, r, f = compute_PRF(m.predict(Xv), yv)
        precs.append(p); recs.append(r); f1s.append(f)
    return (float(np.mean(precs)), float(np.std(precs)),
            float(np.mean(recs)),  float(np.std(recs)),
            float(np.mean(f1s)),   float(np.std(f1s)))

def run_logistic_regression():
    df = load_merged()
    X_train, X_test, _, _, yc_train, yc_test, _, _ = get_features_and_targets(df)

    model = LogisticRegression()
    model.fit(X_train, yc_train)

    y_pred = model.predict(X_test)
    precision, recall, f_score = compute_PRF(y_pred, yc_test)

    # Trivial baseline: always predict the majority class of the training set.
    majority = int(round(yc_train.mean()))
    base_pred = np.full_like(yc_test, majority, dtype=int)
    base_p, base_r, base_f = compute_PRF(base_pred, yc_test)

    # 5-fold stratified CV on full dataset
    X_full, y_reg_full = get_full_xy(df)
    cv = cross_validate_logistic(X_full, y_reg_full)
    cv_p_mean, cv_p_std, cv_r_mean, cv_r_std, cv_f_mean, cv_f_std = cv

    print('=== Logistic Regression — Chronological Split ===')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1-score:  {f_score:.4f}')
    print(f'Test class balance: {int(yc_test.sum())} positive / {len(yc_test)} total')
    print(f'\nTrivial baseline (predict majority of train, class={majority}):')
    print(f'  P={base_p:.4f}  R={base_r:.4f}  F1={base_f:.4f}')
    print(f'\n=== Logistic Regression — 5-fold Stratified CV ===')
    print(f'Precision: {cv_p_mean:.4f} ± {cv_p_std:.4f}')
    print(f'Recall:    {cv_r_mean:.4f} ± {cv_r_std:.4f}')
    print(f'F1:        {cv_f_mean:.4f} ± {cv_f_std:.4f}')

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
    plt.title('Logistic Regression — Confusion Matrix (chronological split)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR + 'logistic_regression.png', dpi=150)
    plt.close()
    print(f'Plot saved to {PLOTS_DIR}logistic_regression.png')

    return {
        'Precision': precision, 'Recall': recall, 'F1': f_score,
        'baseline_P': base_p, 'baseline_R': base_r, 'baseline_F1': base_f,
        'cv_P_mean': cv_p_mean, 'cv_P_std': cv_p_std,
        'cv_R_mean': cv_r_mean, 'cv_R_std': cv_r_std,
        'cv_F_mean': cv_f_mean, 'cv_F_std': cv_f_std,
    }

if __name__ == '__main__':
    run_logistic_regression()
