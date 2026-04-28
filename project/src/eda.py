import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import numpy as np
import matplotlib.pyplot as plt

from prepare_data import load_merged, FEATURE_COLS

PLOTS_DIR = 'outputs/plots/'

def run_eda():
    df = load_merged()
    cols = FEATURE_COLS + ['yield_kg_ha']
    data = df[cols].to_numpy(dtype=float)
    corr = np.corrcoef(data, rowvar=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Cambodia Rice Data — Exploratory Overview', fontweight='bold')

    # Left: yield over time
    ax = axes[0]
    ax.plot(df['year'], df['yield_kg_ha'], 'o-', color='#4878cf', lw=1.5)
    train_end_year = df['year'].iloc[int(0.8 * len(df)) - 1]
    ax.axvline(train_end_year + 0.5, color='red', linestyle='--', lw=1,
               label=f'train/test split ({train_end_year}/{train_end_year+1})')
    ax.set_xlabel('Year')
    ax.set_ylabel('Yield (kg/ha)')
    ax.set_title('Yield over time (1990–2024)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Right: correlation heatmap
    ax = axes[1]
    short = ['area', 'temp', 'rain', 'yield']
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(short))); ax.set_xticklabels(short)
    ax.set_yticks(range(len(short))); ax.set_yticklabels(short)
    for i in range(len(short)):
        for j in range(len(short)):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                    color='white' if abs(corr[i, j]) > 0.5 else 'black',
                    fontweight='bold', fontsize=9)
    ax.set_title('Feature / target correlation')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR + 'eda.png', dpi=150)
    plt.close()
    print(f'EDA plot saved to {PLOTS_DIR}eda.png')

    print('\nCorrelation matrix:')
    header = ' ' * 8 + ''.join(f'{n:>10}' for n in short)
    print(header)
    for i, n in enumerate(short):
        row = f'{n:>8}' + ''.join(f'{corr[i, j]:>10.3f}' for j in range(len(short)))
        print(row)

if __name__ == '__main__':
    run_eda()
