import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import numpy as np
import matplotlib.pyplot as plt

from regression import run_linear_regression
from classification import run_logistic_regression
from neural_network import run_neural_network

PLOTS_DIR = 'outputs/plots/'

def run_comparison():
    print('Running all models...\n')
    lr_reg  = run_linear_regression()
    lr_cls  = run_logistic_regression()
    nn_res  = run_neural_network()

    # ── Regression: Baseline / Linear / NN ───────────────────────────────────
    reg_models = ['Predict\ntrain mean', 'Linear\nRegression', 'Neural\nNetwork']
    rmse_vals  = [lr_reg['baseline_RMSE'], lr_reg['RMSE'], nn_res['reg']['RMSE']]
    r2_vals    = [lr_reg['baseline_R2'],   lr_reg['R2'],   nn_res['reg']['R2']]

    cls_models = ['Logistic\nRegression', 'Neural\nNetwork']
    prec_vals  = [lr_cls['Precision'], nn_res['cls']['Precision']]
    rec_vals   = [lr_cls['Recall'],    nn_res['cls']['Recall']]
    f1_vals    = [lr_cls['F1'],        nn_res['cls']['F1']]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Cambodia Rice Yield — Model Comparison', fontsize=13, fontweight='bold')

    # -- Panel 1: RMSE (chronological split) --
    ax = axes[0]
    bars = ax.bar(reg_models, rmse_vals, color=['#888888', '#4878cf', '#d65f5f'], edgecolor='k')
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.0f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('RMSE (kg/ha)')
    ax.set_title('Regression Task — Test RMSE\n(chronological split, lower is better)')
    ax.grid(axis='y', alpha=0.3)

    # -- Panel 2: R² (chronological split) --
    ax = axes[1]
    bars = ax.bar(reg_models, r2_vals, color=['#888888', '#4878cf', '#d65f5f'], edgecolor='k')
    for bar, val in zip(bars, r2_vals):
        y = val + (0.5 if val < 0 else 0.05)
        ax.text(bar.get_x() + bar.get_width()/2, y,
                f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('R²')
    ax.set_title('Regression Task — Test R²\n(higher is better; negative ⇒ worse than baseline)')
    ax.grid(axis='y', alpha=0.3)

    # -- Panel 3: classification metrics --
    ax = axes[2]
    x = np.arange(2); w = 0.25
    b1 = ax.bar(x - w, prec_vals, w, label='Precision', color='#4878cf', edgecolor='k')
    b2 = ax.bar(x,     rec_vals,  w, label='Recall',    color='#d65f5f', edgecolor='k')
    b3 = ax.bar(x + w, f1_vals,   w, label='F1',        color='#5aaa5a', edgecolor='k')
    for bars, vals in [(b1, prec_vals), (b2, rec_vals), (b3, f1_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(cls_models)
    ax.set_ylim(0, 1.25)
    ax.axhline(1.0, color='gray', linestyle='--', lw=0.8)
    ax.set_ylabel('Score')
    ax.set_title('Classification Task\n(test set is all-positive; see report)')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR + 'comparison.png', dpi=150)
    plt.close()
    print(f'\nComparison plot saved to {PLOTS_DIR}comparison.png')

    # ── 5-fold CV comparison plot ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('5-fold Cross-Validation Results', fontsize=13, fontweight='bold')

    ax = axes[0]
    means = [lr_reg['cv_RMSE_mean'], nn_res['reg']['cv_RMSE_mean']]
    stds  = [lr_reg['cv_RMSE_std'],  nn_res['reg']['cv_RMSE_std']]
    ax.bar(['Linear\nRegression', 'Neural\nNetwork'], means, yerr=stds,
           color=['#4878cf', '#d65f5f'], edgecolor='k', capsize=8)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 5, f'{m:.0f} ± {s:.0f}', ha='center', fontweight='bold')
    ax.set_ylabel('RMSE (kg/ha)')
    ax.set_title('Regression — 5-fold CV RMSE')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    means = [lr_cls['cv_F_mean'], nn_res['cls']['cv_F_mean']]
    stds  = [lr_cls['cv_F_std'],  nn_res['cls']['cv_F_std']]
    ax.bar(['Logistic\nRegression', 'Neural\nNetwork'], means, yerr=stds,
           color=['#4878cf', '#d65f5f'], edgecolor='k', capsize=8)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f'{m:.2f} ± {s:.2f}', ha='center', fontweight='bold')
    ax.set_ylim(0, 1.25)
    ax.axhline(1.0, color='gray', linestyle='--', lw=0.8)
    ax.set_ylabel('F1 Score')
    ax.set_title('Classification — 5-fold Stratified CV F1')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR + 'cv_comparison.png', dpi=150)
    plt.close()
    print(f'CV comparison plot saved to {PLOTS_DIR}cv_comparison.png')

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n========== FINAL RESULTS ==========')
    print('\n--- Regression — chronological split (28 train / 7 test) ---')
    print(f'{"Model":<26} {"RMSE":>10} {"R²":>10}')
    print(f'{"Predict train mean":<26} {lr_reg["baseline_RMSE"]:>10.2f} {lr_reg["baseline_R2"]:>10.4f}')
    print(f'{"Linear Regression":<26} {lr_reg["RMSE"]:>10.2f} {lr_reg["R2"]:>10.4f}')
    print(f'{"Neural Network":<26} {nn_res["reg"]["RMSE"]:>10.2f} {nn_res["reg"]["R2"]:>10.4f}')

    print('\n--- Regression — 5-fold CV (random shuffled) ---')
    print(f'{"Model":<26} {"RMSE (mean ± std)":>22} {"R² (mean ± std)":>22}')
    print(f'{"Linear Regression":<26} {lr_reg["cv_RMSE_mean"]:>10.2f} ± {lr_reg["cv_RMSE_std"]:<8.2f} {lr_reg["cv_R2_mean"]:>10.4f} ± {lr_reg["cv_R2_std"]:<8.4f}')
    print(f'{"Neural Network":<26} {nn_res["reg"]["cv_RMSE_mean"]:>10.2f} ± {nn_res["reg"]["cv_RMSE_std"]:<8.2f} {nn_res["reg"]["cv_R2_mean"]:>10.4f} ± {nn_res["reg"]["cv_R2_std"]:<8.4f}')

    print('\n--- Classification — chronological split ---')
    print(f'{"Model":<26} {"Precision":>10} {"Recall":>10} {"F1":>10}')
    print(f'{"Logistic Regression":<26} {lr_cls["Precision"]:>10.4f} {lr_cls["Recall"]:>10.4f} {lr_cls["F1"]:>10.4f}')
    print(f'{"Neural Network":<26} {nn_res["cls"]["Precision"]:>10.4f} {nn_res["cls"]["Recall"]:>10.4f} {nn_res["cls"]["F1"]:>10.4f}')

    print('\n--- Classification — 5-fold stratified CV ---')
    print(f'{"Model":<26} {"P (mean ± std)":>18} {"R (mean ± std)":>18} {"F1 (mean ± std)":>18}')
    print(f'{"Logistic Regression":<26} {lr_cls["cv_P_mean"]:>8.3f} ± {lr_cls["cv_P_std"]:<6.3f} {lr_cls["cv_R_mean"]:>8.3f} ± {lr_cls["cv_R_std"]:<6.3f} {lr_cls["cv_F_mean"]:>8.3f} ± {lr_cls["cv_F_std"]:<6.3f}')
    print(f'{"Neural Network":<26} {nn_res["cls"]["cv_P_mean"]:>8.3f} ± {nn_res["cls"]["cv_P_std"]:<6.3f} {nn_res["cls"]["cv_R_mean"]:>8.3f} ± {nn_res["cls"]["cv_R_std"]:<6.3f} {nn_res["cls"]["cv_F_mean"]:>8.3f} ± {nn_res["cls"]["cv_F_std"]:<6.3f}')
    print('====================================')

    return {'lr_reg': lr_reg, 'lr_cls': lr_cls, 'nn': nn_res}

if __name__ == '__main__':
    run_comparison()
