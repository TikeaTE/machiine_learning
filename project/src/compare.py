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

    # ── Regression comparison: Linear Regression vs NN ───────────────────────
    reg_models  = ['Linear\nRegression', 'Neural\nNetwork']
    mse_vals    = [lr_reg['MSE'],  nn_res['reg']['MSE']]
    rmse_vals   = [lr_reg['RMSE'], nn_res['reg']['RMSE']]
    r2_vals     = [lr_reg['R2'],   nn_res['reg']['R2']]

    # ── Classification comparison: Logistic Regression vs NN ─────────────────
    cls_models  = ['Logistic\nRegression', 'Neural\nNetwork']
    prec_vals   = [lr_cls['Precision'], nn_res['cls']['Precision']]
    rec_vals    = [lr_cls['Recall'],    nn_res['cls']['Recall']]
    f1_vals     = [lr_cls['F1'],        nn_res['cls']['F1']]

    x = np.arange(2)
    w = 0.25
    colors = ['#4878cf', '#d65f5f', '#5aaa5a']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Cambodia Rice Yield — Model Comparison', fontsize=13, fontweight='bold')

    # -- Left: Regression metrics (RMSE, R²) --
    ax = axes[0]
    bars1 = ax.bar(x - w/2, rmse_vals, w, label='RMSE', color=colors[0], edgecolor='k')
    # R² on secondary axis
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + w/2, r2_vals, w, label='R²', color=colors[2], edgecolor='k', alpha=0.7)
    for bar, val in zip(bars1, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}', ha='center', fontsize=9, fontweight='bold', color=colors[0])
    for bar, val in zip(bars2, r2_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                 f'{val:.2f}', ha='center', fontsize=9, fontweight='bold', color=colors[2])
    ax.set_xticks(x)
    ax.set_xticklabels(reg_models)
    ax.set_ylabel('RMSE (kg/ha)', color=colors[0])
    ax2.set_ylabel('R²', color=colors[2])
    ax.set_title('Regression Task')
    lines = [bars1, bars2]
    ax.legend([bars1, bars2], ['RMSE', 'R²'], loc='upper right')

    # -- Right: Classification metrics (Precision, Recall, F1) --
    ax = axes[1]
    b1 = ax.bar(x - w, prec_vals, w, label='Precision', color=colors[0], edgecolor='k')
    b2 = ax.bar(x,     rec_vals,  w, label='Recall',    color=colors[1], edgecolor='k')
    b3 = ax.bar(x + w, f1_vals,   w, label='F1',        color=colors[2], edgecolor='k')
    for bars, vals in [(b1, prec_vals), (b2, rec_vals), (b3, f1_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cls_models)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel('Score')
    ax.set_title('Classification Task')
    ax.legend()
    ax.axhline(1.0, color='gray', linestyle='--', lw=0.8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR + 'comparison.png', dpi=150)
    plt.close()
    print(f'\nComparison plot saved to {PLOTS_DIR}comparison.png')

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n========== FINAL RESULTS ==========')
    print('\n--- Regression ---')
    print(f'{"Model":<22} {"MSE":>10} {"RMSE":>10} {"R²":>8}')
    print(f'{"Linear Regression":<22} {lr_reg["MSE"]:>10.2f} {lr_reg["RMSE"]:>10.2f} {lr_reg["R2"]:>8.4f}')
    print(f'{"Neural Network":<22} {nn_res["reg"]["MSE"]:>10.2f} {nn_res["reg"]["RMSE"]:>10.2f} {nn_res["reg"]["R2"]:>8.4f}')

    print('\n--- Classification ---')
    print(f'{"Model":<22} {"Precision":>10} {"Recall":>8} {"F1":>8}')
    print(f'{"Logistic Regression":<22} {lr_cls["Precision"]:>10.4f} {lr_cls["Recall"]:>8.4f} {lr_cls["F1"]:>8.4f}')
    print(f'{"Neural Network":<22} {nn_res["cls"]["Precision"]:>10.4f} {nn_res["cls"]["Recall"]:>8.4f} {nn_res["cls"]["F1"]:>8.4f}')
    print('====================================')

if __name__ == '__main__':
    run_comparison()
